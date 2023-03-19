"""
This module contains general denoising functions built on top of base denoisers.
"""

from __future__ import annotations

from itertools import count, zip_longest
from typing import Any, Callable, Iterable, cast

from vskernels import Bilinear, Catrom, Scaler, ScalerT
from vsrgtools import RemoveGrainMode, contrasharpening, contrasharpening_dehalo, removegrain
from vsrgtools.util import norm_rmode_planes
from vstools import (
    FunctionUtil, KwargsT, PlanesT, VSFunction, depth, expect_bits, fallback, get_h, get_w, normalize_planes, vs
)

from .limit import TemporalLimiter, TemporalLimiterConfig
from .mvtools import MotionMode, MVTools, MVToolsPresets, SADMode, SearchMode
from .mvtools.enums import SearchModeBase
from .mvtools.utils import normalize_thscd
from .postprocess import PostProcess, PostProcessConfig
from .prefilters import PelType, Prefilter

__all__ = [
    'mlm_degrain',

    'temporal_degrain'
]


def mlm_degrain(
    clip: vs.VideoNode, tr: int = 3, refine: int = 3, thSAD: int | tuple[int, int] = 200,
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
    :param thSAD:               thSAD param of :py:attr:`MVTools.analyze`/:py:attr:`MVTools.degrain`.
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

    if isinstance(thSAD, tuple):
        thSADA, thSADD = thSAD
    else:
        thSADA = thSADD = thSAD

    mkwargs_def = dict[str, Any](tr=tr, refine=refine, planes=planes)
    akwargs_def = dict[str, Any](motion=MotionMode.HIGH_SAD, thSAD=thSADA, pel_type=PelType.WIENER)
    dkwargs_def = dict[str, Any](thSAD=thSADD)

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

    last_idx = len(diffed_clips)

    new_resolutions = [(c.width, c.height) for c in diffed_clips]

    base_denoise = _degrain(scaled_clips[0], None, 0)
    ref_den_clips = [base_denoise]
    for width, height in new_resolutions:
        ref_den_clips.append(scaler.scale(ref_den_clips[-1], width, height))

    ref_denoise = ref_den_clips[1]

    for i, diff, ref_den, ref_den_next in zip(
        count(1), diffed_clips, ref_den_clips[1:], ref_den_clips[2:] + ref_den_clips[-1:]
    ):
        ref_denoise = ref_denoise.std.MakeDiff(_degrain(diff, ref_den, i))

        if not i == last_idx:
            ref_denoise = scaler.scale(ref_denoise, ref_den_next.width, ref_den_next.height)

    return depth(ref_denoise, bits)


def temporal_degrain(
    clip: vs.VideoNode, tr: int = 1, thSAD: int | None = None,
    block_size: int | None = None, refine: int = 0,
    post: PostProcess | PostProcessConfig | None = None,
    thSAD2: int | None = None, thSCD: int | tuple[int | None, int | None] | None = (None, 50),
    search_mode: SearchMode | SearchMode.Config = SearchMode.HEXAGON,
    sad_mode: SADMode | tuple[SADMode, SADMode] = SADMode.SPATIAL.same_recalc,
    prefilter: Prefilter | vs.VideoNode | None = None,
    truemotion: bool = False, global_motion: bool = True, chroma_motion: bool = True,
    limiter: TemporalLimiter | TemporalLimiterConfig | VSFunction = TemporalLimiter.FFT3D,
    grain_level: int = 4, contra: bool | int | float = False, planes: PlanesT = None,
    **kwargs: Any
) -> vs.VideoNode:
    func = FunctionUtil(clip, temporal_degrain, planes, (vs.GRAY, vs.YUV))

    chroma_motion = chroma_motion and func.chroma

    long_lat, short_lat = max(*(x := (func.clip.width, func.clip.height))), min(*x)

    auto_tune = next(
        (i for i, (k, j) in enumerate([(1050, 576), (1280, 720), (2048, 1152)]) if long_lat <= k and short_lat <= j), 3
    )

    thSCD = normalize_thscd(thSCD, ([192, 192, 192, 256, 320, 384][grain_level], 50), temporal_degrain, scale=False)

    if post is None:
        post = PostProcess.DFTTEST

    postConf = post if isinstance(post, PostProcessConfig) else post()

    prefilter = fallback(
        prefilter, [*([Prefilter.NONE] * 3), *([Prefilter.GAUSSBLUR2] * 3)][grain_level]  # type: ignore
    )

    if not isinstance(search_mode, SearchModeBase.Config):
        meAlgPar = 5 if refine and truemotion else 2
        meSubpel = [4, 2, 2, 1][auto_tune]

        search_mode = search_mode(meAlgPar, meSubpel, search_mode)

    block_size = fallback(block_size, [8, 8, 16, 32][auto_tune])

    motion_lambda = (1000 if truemotion else 100) * (block_size ** 2) // 64

    th_scale = 1.7 if (sad_mode[0] if isinstance(sad_mode, tuple) else sad_mode) is SADMode.SATD else 1.0

    thSAD, thSAD2 = [
        int((d * 64.0 if x is None else x) * th_scale)
        for x, d in zip((thSAD, thSAD2), list[tuple[int, int]]([
            (3, 2), (5, 4), (7, 5), (9, 6), (11, 7), (13, 8)
        ])[grain_level])
    ]

    limitVal = [6, 8, 12, 16, 32, 48][[-1, -1, 0, 0, 0, 1][grain_level] + auto_tune + 1]

    if isinstance(limiter, TemporalLimiterConfig):
        limitConf = limiter
    elif limiter is TemporalLimiter.FFT3D:
        ov = 2 * round(limitVal * 2 / [4, 4, 4, 3, 2, 2][grain_level] * 0.5)

        limitConf = limiter(sigma=limitVal, block_size=limitVal * 2, ov=ov)  # type: ignore
    elif limiter is TemporalLimiter.DFTTEST:
        limitConf = limiter(sigma_low=limitVal / 2, sigma_high=limitVal)  # type: ignore
    elif isinstance(limiter, TemporalLimiter):
        limitConf = limiter()  # type: ignore
    else:
        limitConf = TemporalLimiter.CUSTOM(limiter)

    class MotionModeCustom(MotionMode.Config):
        def block_coherence(self, block_size: int) -> int:
            return motion_lambda // 4

    motion_ref = MotionMode.from_param(truemotion)

    preset = MVToolsPresets.CUSTOM(
        tr=tr, refine=refine, prefilter=prefilter(func.work_clip) if isinstance(prefilter, Prefilter) else prefilter,
        pel=meSubpel, hpad=block_size, vpad=block_size, block_size=block_size, recalculate_args=dict(thsad=thSAD // 2),
        overlap=property(lambda self: self.block_size // 2), analyze_args=dict(chroma=chroma_motion),
        search=search_mode, super_args=dict(chroma=chroma_motion), planes=func.norm_planes, motion=MotionModeCustom(
            truemotion, motion_lambda, motion_ref.sad_limit, 50 if truemotion else 25, motion_ref.plevel, global_motion
        )
    )

    maxMV = MVTools(func.work_clip, **preset(tr=max(tr, postConf.tr), **kwargs)).analyze()

    NR2 = limitConf(
        func.work_clip, thSAD, thSAD2, thSCD,
        func.with_planes([1, 2]) if chroma_motion else func, maxMV, preset
    ) if tr > 0 else func.work_clip

    if postConf.tr > 0:
        dnWindow = MVTools(NR2, vectors=maxMV, **preset(tr=postConf.tr)).compensate(postConf, thSAD2, thSCD)
    else:
        dnWindow = postConf(NR2)

    if contra:
        if contra is True:
            contra = 3

        if isinstance(contra, int):
            sharpened = contrasharpening(dnWindow, func.work_clip, contra, 13, func.norm_planes)
        else:
            sharpened = contrasharpening_dehalo(dnWindow, func.work_clip, contra, 2.5, func.norm_planes)
    else:
        sharpened = dnWindow

    if postConf.tr > 0 and postConf.merge_strength:
        sharpened = func.work_clip.std.Merge(sharpened, postConf.merge_strength / 100)

    return func.return_clip(sharpened)
