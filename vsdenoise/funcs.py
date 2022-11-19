"""
This module contains general denoising functions built on top of base denoisers.
"""

from __future__ import annotations

from itertools import count, zip_longest
from typing import Any, Callable, Iterable, cast

from vskernels import Bilinear, Catrom, Scaler, ScalerT
from vsrgtools import RemoveGrainMode, removegrain
from vsrgtools.util import norm_rmode_planes
from vstools import KwargsT, PlanesT, depth, expect_bits, get_h, get_w, normalize_planes, vs

from .mvtools import MotionMode, MVTools
from .prefilters import PelType


def mlm_degrain(
    clip: vs.VideoNode, tr: int = 3, refine: int = 3, thSAD: int = 200,
    factors: Iterable[float] | range = [1 / 3, 2 / 3],
    scaler: ScalerT = Bilinear, downscaler: ScalerT = Catrom,
    mv_kwargs: KwargsT | list[KwargsT] | None = None,
    analyze_kwargs: KwargsT | list[KwargsT] | None = None,
    degrain_kwargs: KwargsT | list[KwargsT] | None = None,
    soften: Callable[..., vs.VideoNode] | bool | None = False,
    planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode:
    """
    Multi Level scaling Motion compensated Degrain. Original idea by Didée.

    The observation was that when downscaling the source to a smaller resolution,
    a normal MVTools Degrain pass can produce a much stabler result.

    The approach taken here is to first make a small-but-stable-denoised clip,
    then work our way upwards to the original resolution, averaging their differences.

    :param clip:                Clip to be denoised.
    :param tr:                  Temporal radius of the denoising.
    :param refine:              Refine param of :py:class:`MVTools`.
    :param thSAD:               Refine param of :py:attr:`MVTools.analyze`.
    :param factors:             Scaling factors.
                                 * If floats, they will be interpreted as size * factor.
                                 * If a range, it will first be normalized as a list of float with 1 / factor.
    :param scaler:              Scaler to use for scaling the downscaled clips up when diffing them.
    :param downscaler:          Scaler to use for downscaling the clip to various levels.
    :param mv_kwargs:           Keyword arguments to pass to :py:class:`MVTools`.
    :param analyze_kwargs:      Keyword arguments to pass to :py:attr:`MVTools.analyze`.
    :param degrain_kwargs:      Keyword arguments to pass to :py:attr:`MVTools.degrain`.
    :param soften:              Use a softening function to sharpen the output; recommended only for live content.
    :param planes:              Planes to process.

    :return:                    Denoised clip.
    """

    planes = normalize_planes(clip, planes)

    scaler = Scaler.ensure_obj(scaler, mlm_degrain)
    downscaler = Scaler.ensure_obj(downscaler, mlm_degrain)

    do_soft = bool(soften)

    mkwargs_def = dict[str, Any](tr=tr, refine=refine, planes=planes)
    akwargs_def = dict[str, Any](motion=MotionMode.HIGH_SAD, thSAD=thSAD, pel_type=PelType.WIENER)
    dkwargs_def = dict[str, Any](thSAD=thSAD)

    mkwargs, akwargs, dkwargs = [
        [default] if kwargs is None else [
            (default | val) for val in
            (kwargs if isinstance(kwargs, list) else [kwargs])
        ] for default, kwargs in (
            (mkwargs_def, mv_kwargs),
            (akwargs_def, analyze_kwargs),
            (dkwargs_def, degrain_kwargs)
        )
    ]

    if isinstance(factors, range):
        factors = [1 / x for x in factors if x >= 1]
    else:
        factors = list(factors)

    mkwargs_fact, akwargs_fact, dkwargs_fact = [
        cast(
            list[tuple[float, dict[str, Any]]],
            list(zip_longest(
                factors, kwargs[:len(factors)], fillvalue=kwargs[-1]
            ))
        ) for kwargs in (mkwargs, akwargs, dkwargs)
    ]

    factors = set(sorted(factors)) - {0, 1}

    norm_mkwargs, norm_akwargs, norm_dkwargs = [
        [
            next(x[1] for x in kwargs if x[0] == factor)
            for factor in factors
        ] for kwargs in (mkwargs_fact, akwargs_fact, dkwargs_fact)
    ]

    norm_mkwargs, norm_akwargs, norm_dkwargs = [
        norm_mkwargs + norm_mkwargs[-1:], norm_akwargs + norm_akwargs[-1:], norm_dkwargs + norm_dkwargs[-1:]
    ]

    def _degrain(clip: vs.VideoNode, ref: vs.VideoNode | None, idx: int) -> vs.VideoNode:
        mvtools_arg = dict(**norm_mkwargs[idx])

        if do_soft and idx in {0, last_idx}:
            if soften is True:
                softened = removegrain(clip, norm_rmode_planes(
                    clip, RemoveGrainMode.SQUARE_BLUR if clip.width < 1200 else RemoveGrainMode.BOX_BLUR, planes
                ))
            elif callable(soften):
                try:
                    softened = soften(clip, planes=planes)
                except BaseException:
                    softened = soften(clip)

            mvtools_arg |= dict(prefilter=softened)

        mvtools_arg |= dict(
            pel=2 if idx == 0 else 1, block_size=16 if clip.width > 960 else 8
        ) | norm_akwargs[idx] | kwargs

        mv = MVTools(clip, **mvtools_arg)
        mv.analyze(ref=ref)
        return mv.degrain(**norm_dkwargs[idx])

    clip, bits = expect_bits(clip, 16)
    resolutions = [
        (get_w(clip.height * factor, clip), get_h(clip.width * factor, clip))
        for factor in factors
    ]

    scaled_clips = [clip]
    for width, height in resolutions[::-1]:
        scaled_clips.insert(0, downscaler.scale(scaled_clips[0], width, height))

    diffed_clips = [
        scaler.scale(clip, nclip.width, nclip.height).std.MakeDiff(nclip)
        for clip, nclip in zip(scaled_clips[:-1], scaled_clips[1:])
    ]

    new_resolutions = [(c.width, c.height) for c in diffed_clips]

    base_denoise = _degrain(scaled_clips[0], None, 0)
    ref_den_clips = [base_denoise]
    for width, height in new_resolutions:
        ref_den_clips.append(scaler.scale(ref_den_clips[-1], width, height))

    ref_denoise = ref_den_clips[1]

    last_idx = len(diffed_clips)

    for i, diff, ref_den, ref_den_next in zip(
        count(1), diffed_clips, ref_den_clips[1:], ref_den_clips[2:] + ref_den_clips[-1:]
    ):
        ref_denoise = ref_denoise.std.MakeDiff(_degrain(diff, ref_den, i))

        if not i == last_idx:
            ref_denoise = scaler.scale(ref_denoise, ref_den_next.width, ref_den_next.height)

    return depth(ref_denoise, bits)
