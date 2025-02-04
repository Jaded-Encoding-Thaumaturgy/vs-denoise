"""
This module contains general denoising functions built on top of base denoisers.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, overload

from vskernels import Scaler, ScalerT, Bilinear, Catrom
from vsscale import Waifu2x
from vsscale.scale import BaseWaifu2x
from vstools import CustomIndexError, KwargsNotNone, PlanesT, VSFunction, fallback, vs, normalize_seq, mod2

from .mvtools import MotionVectors, MVTools, MVToolsPreset, MVToolsPresets

__all__ = [
    'mc_degrain',

    'mlm_degrain',

    'waifu2x_denoise'
]


@overload
def mc_degrain(
    clip: vs.VideoNode, vectors: MotionVectors | MVTools | None = None,
    prefilter: vs.VideoNode | VSFunction | None = None, mfilter: vs.VideoNode | VSFunction | None = None,
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD,
    blksize: int | tuple[int, int] = 16, refine: int = 1,
    thsad: int | tuple[int, int] = 400, thsad2: int | tuple[int | None, int | None] | None = None,
    thsad_recalc: int | None = None, limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None, export_globals: Literal[False] = ...,
    planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode:
    ...


@overload
def mc_degrain(
    clip: vs.VideoNode, vectors: MotionVectors | MVTools | None = None,
    prefilter: vs.VideoNode | VSFunction | None = None, mfilter: vs.VideoNode | VSFunction | None = None,
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD,
    blksize: int | tuple[int, int] = 16, refine: int = 1,
    thsad: int | tuple[int, int] = 400, thsad2: int | tuple[int | None, int | None] | None = None,
    thsad_recalc: int | None = None, limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None, export_globals: Literal[True] = ...,
    planes: PlanesT = None, **kwargs: Any
) -> tuple[vs.VideoNode, MVTools]:
    ...


@overload
def mc_degrain(
    clip: vs.VideoNode, vectors: MotionVectors | MVTools | None = None,
    prefilter: vs.VideoNode | VSFunction | None = None, mfilter: vs.VideoNode | VSFunction | None = None,
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD,
    blksize: int | tuple[int, int] = 16, refine: int = 1,
    thsad: int | tuple[int, int] = 400, thsad2: int | tuple[int | None, int | None] | None = None,
    thsad_recalc: int | None = None, limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None, export_globals: bool = ...,
    planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]:
    ...


def mc_degrain(
    clip: vs.VideoNode, vectors: MotionVectors | MVTools | None = None,
    prefilter: vs.VideoNode | VSFunction | None = None, mfilter: vs.VideoNode | VSFunction | None = None,
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD,
    blksize: int | tuple[int, int] = 16, refine: int = 1,
    thsad: int | tuple[int, int] = 400, thsad2: int | tuple[int | None, int | None] | None = None,
    thsad_recalc: int | None = None, limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None, export_globals: bool = False,
    planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]:

    mv_args = preset | kwargs | KwargsNotNone(search_clip=prefilter, tr=tr)

    blksize = normalize_seq(blksize, 2)
    thsad = normalize_seq(thsad, 2)

    mv = MVTools(clip, vectors=vectors, planes=planes, **mv_args)

    mv.super(mv.search_clip, rfilter=4)

    if not vectors:
        mv.analyze(blksize=blksize, overlap=[i // 2 for i in blksize])

        if refine:
            if thsad_recalc is None:
                thsad_recalc = thsad[0] // 2

            for _ in range(refine):
                blksize = [i // 2 for i in blksize]
                overlap = [i // 2 for i in blksize]

                mv.recalculate(thsad=thsad_recalc, blksize=blksize, overlap=overlap)

    mfilter = mfilter(clip) if callable(mfilter) else fallback(mfilter, mv.clip)

    den = mv.degrain(mfilter, mv.clip, None, tr, thsad, thsad2, limit, thscd)

    return (den, mv) if export_globals else den


def mlm_degrain(
    clip: vs.VideoNode, factors: Iterable[int] = [1 / 4, 1 / 3, 1 / 2],
    downsampler: ScalerT = Bilinear, upsampler: ScalerT = Catrom,
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD,
    blksize: int | tuple[int, int] = 16, refine: int = 1,
    thsad: int | tuple[int, int] = 400,
    limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None,
    planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]:

    downsampler = Scaler.ensure_obj(downsampler)
    upsampler = Scaler.ensure_obj(upsampler)

    mv_args = dict(
        tr=tr, preset=preset, blksize=blksize, refine=refine, limit=limit, thscd=thscd, planes=planes
    ) | kwargs
    
    factors = sorted(factors, reverse=True)
    downsampled_clips, residuals = [clip], list[vs.VideoNode]()

    for x, i in enumerate(factors[1:]):
        base_clip = downsampled_clips[x]
        ds_clip = downsampler.scale(base_clip, mod2(clip.width * i), mod2(clip.height * i))

        ds_up = upsampler.scale(ds_clip, base_clip.width, base_clip.height)
        ds_diff = ds_up.std.MakeDiff(base_clip)

        downsampled_clips.append(ds_clip)
        residuals.append(ds_diff)

    downsampled_clips, residuals = downsampled_clips[::-1], residuals[::-1]

    den_base = mc_degrain(downsampled_clips[0], thsad=thsad, **mv_args)

    for x in range(len(factors) - 1):
        next_base = downsampled_clips[x + 1]
        base_up = upsampler.scale(den_base, next_base.width, next_base.height)
        den_last = mc_degrain(residuals[x], prefilter=downsampled_clips[x + 1], thsad=thsad / factors[x + 1], **mv_args)
        den_base = base_up.std.MakeDiff(den_last)

    return den_base


def waifu2x_denoise(
    clip: vs.VideoNode, noise: int = 1, model: type[BaseWaifu2x] = Waifu2x.Cunet, **kwargs: Any
) -> vs.VideoNode:
    if noise < 0 or noise > 3:
        raise CustomIndexError('"noise" must be in range 0-3 (inclusive).')

    if not isinstance(model, type):
        model = model.__class__  # type: ignore

    return model(**kwargs).scale(clip, clip.width, clip.height, _static_args=dict(scale=1, noise=noise, force=True))
