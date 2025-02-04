"""
This module contains general denoising functions built on top of base denoisers.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, overload

from vskernels import Scaler, ScalerT, Bilinear, Catrom
from vsscale import Waifu2x
from vsscale.scale import BaseWaifu2x
from vstools import CustomIndexError, KwargsNotNone, PlanesT, VSFunction, fallback, vs

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
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD, refine: bool = True,
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
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD, refine: bool = True, 
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
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD, refine: bool = True,
    thsad: int | tuple[int, int] = 400, thsad2: int | tuple[int | None, int | None] | None = None,
    thsad_recalc: int | None = None, limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None, export_globals: bool = ...,
    planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]:
    ...


def mc_degrain(
    clip: vs.VideoNode, vectors: MotionVectors | MVTools | None = None,
    prefilter: vs.VideoNode | VSFunction | None = None, mfilter: vs.VideoNode | VSFunction | None = None,
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD, refine: bool = True,
    thsad: int | tuple[int, int] = 400, thsad2: int | tuple[int | None, int | None] | None = None,
    thsad_recalc: int | None = None, limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None, export_globals: bool = False,
    planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]:

    mv_args = preset | kwargs | KwargsNotNone(search_clip=prefilter, tr=tr)

    if thsad_recalc is None:
        thsad_recalc = (thsad if isinstance(thsad, int) else thsad[0]) // 2

    mv = MVTools(clip, vectors=vectors, planes=planes, **mv_args)

    if not vectors:
        mv.analyze()

        if refine:
            mv.recalculate(thsad=thsad_recalc)

    mfilter = mfilter(clip) if callable(mfilter) else fallback(mfilter, mv.clip)

    den = mv.degrain(mfilter, mv.clip, None, tr, thsad, thsad2, limit, thscd)

    return (den, mv) if export_globals else den


@overload
def mlm_degrain(
    clip: vs.VideoNode, sizes: Iterable[int] = [8, 16, 32],
    downsampler: ScalerT = Bilinear, upsampler: ScalerT = Catrom,
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD,
    thsad: int | tuple[int, int] = 400,
    limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None,
    export_globals: Literal[False] = ..., planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode:
    ...


@overload
def mlm_degrain(
    clip: vs.VideoNode, sizes: Iterable[int] = [8, 16, 32],
    downsampler: ScalerT = Bilinear, upsampler: ScalerT = Catrom,
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD,
    thsad: int | tuple[int, int] = 400,
    limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None,
    export_globals: Literal[True] = ..., planes: PlanesT = None, **kwargs: Any
) -> tuple[vs.VideoNode, MVTools]:
    ...


@overload
def mlm_degrain(
    clip: vs.VideoNode, sizes: Iterable[int] = [8, 16, 32],
    downsampler: ScalerT = Bilinear, upsampler: ScalerT = Catrom,
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD,
    thsad: int | tuple[int, int] = 400,
    limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None,
    export_globals: bool = ..., planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]:
    ...


def mlm_degrain(
    clip: vs.VideoNode, sizes: Iterable[int] = [8, 16, 32],
    downsampler: ScalerT = Bilinear, upsampler: ScalerT = Catrom,
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD,
    thsad: int | tuple[int, int] = 400,
    limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None,
    export_globals: bool = False, planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]:

    downsampler = Scaler.ensure_obj(downsampler)
    upsampler = Scaler.ensure_obj(upsampler)

    mv_args = preset | kwargs | KwargsNotNone(tr=tr)

    thsad_recalc = thsad if isinstance(thsad, int) else thsad[0]
    
    sizes = sorted(sizes)
    factors = sorted([sizes[-1] // i for i in sizes])
    downsampled_clips, residuals = [clip], list[vs.VideoNode]()

    for x, i in enumerate(factors[1:]):
        base_clip = downsampled_clips[x]
        ds_clip = downsampler.scale(base_clip, clip.width // i, clip.height // i)

        ds_up = upsampler.scale(ds_clip, base_clip.width, base_clip.height)
        ds_diff = ds_up.std.MakeDiff(base_clip)

        downsampled_clips.append(ds_clip)
        residuals.append(ds_diff)

    downsampled_clips, residuals = downsampled_clips[::-1], residuals[::-1]

    mv = MVTools(downsampled_clips[0], planes=planes, **mv_args)

    mv.analyze(blksize=sizes[0], overlap=sizes[0] // 2)
    mv.recalculate(blksize=sizes[0] // 2, overlap=sizes[0] // 4)

    den_base = mv.degrain(thsad=thsad, limit=limit, thscd=thscd)

    for x in range(len(factors) - 1):
        scale = factors[x + 1] // factors[x]
        base_up = upsampler.scale(den_base, den_base.width * scale, den_base.height * scale)

        mv.scale_vectors(scale)
        mv.recalculate(downsampled_clips[x + 1], thsad=thsad_recalc // scale, blksize=sizes[x], overlap=sizes[x] // 2)

        den_last = mv.degrain(residuals[x], thsad=thsad * factors[x], limit=limit, thscd=thscd)
        den_base = base_up.std.MakeDiff(den_last)

    return (den_base, mv) if export_globals else den_base


def waifu2x_denoise(
    clip: vs.VideoNode, noise: int = 1, model: type[BaseWaifu2x] = Waifu2x.Cunet, **kwargs: Any
) -> vs.VideoNode:
    if noise < 0 or noise > 3:
        raise CustomIndexError('"noise" must be in range 0-3 (inclusive).')

    if not isinstance(model, type):
        model = model.__class__  # type: ignore

    return model(**kwargs).scale(clip, clip.width, clip.height, _static_args=dict(scale=1, noise=noise, force=True))
