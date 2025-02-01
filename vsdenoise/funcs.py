"""
This module contains general denoising functions built on top of base denoisers.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, overload

from vskernels import Bilinear
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
    clip: vs.VideoNode, prefilter: vs.VideoNode | VSFunction | None = None,
    mfilter: vs.VideoNode | VSFunction | None = None, vectors: MotionVectors | MVTools | None = None,
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD, refine: bool = True,
    export_globals: Literal[False] = ..., planes: PlanesT = None
) -> vs.VideoNode:
    ...


@overload
def mc_degrain(
    clip: vs.VideoNode, prefilter: vs.VideoNode | VSFunction | None = None,
    mfilter: vs.VideoNode | VSFunction | None = None, vectors: MotionVectors | MVTools | None = None,
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD, refine: bool = True,
    export_globals: Literal[True] = ..., planes: PlanesT = None
) -> tuple[vs.VideoNode, MVTools]:
    ...


@overload
def mc_degrain(
    clip: vs.VideoNode, prefilter: vs.VideoNode | VSFunction | None = None,
    mfilter: vs.VideoNode | VSFunction | None = None, vectors: MotionVectors | MVTools | None = None,
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD, refine: bool = True,
    export_globals: bool = ..., planes: PlanesT = None
) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]:
    ...


def mc_degrain(
    clip: vs.VideoNode, prefilter: vs.VideoNode | VSFunction | None = None,
    mfilter: vs.VideoNode | VSFunction | None = None, vectors: MotionVectors | MVTools | None = None,
    tr: int = 1, preset: MVToolsPreset = MVToolsPresets.HQ_SAD, refine: bool = True,
    export_globals: bool = False, planes: PlanesT = None
) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]:

    mv_args = preset | KwargsNotNone(search_clip=prefilter, tr=tr)

    mv = MVTools(clip, vectors=vectors, planes=planes, **mv_args)

    if not vectors:
        mv.analyze()

        if refine:
            mv.recalculate()

    mfilter = mfilter(clip) if callable(mfilter) else fallback(mfilter, mv.clip)

    den = mv.degrain(mfilter, mv.clip, tr=tr)

    return (den, mv) if export_globals else den


def mlm_degrain(
    clip: vs.VideoNode,
    sizes: Iterable[float] = [4, 8, 16],
    downsampler = Bilinear,
    upsampler = Bilinear,
) -> vs.VideoNode:
    
    factors = sorted([max(sizes) // i for i in sizes])
    downsampled_clips, residuals = [clip], []

    for x, i in enumerate(factors[1:]):
        base_clip = downsampled_clips[x]
        ds_clip = downsampler.scale(base_clip, clip.width // i, clip.height // i)

        ds_up = upsampler.scale(ds_clip, base_clip.width, base_clip.height)
        ds_diff = ds_up.std.MakeDiff(base_clip)

        downsampled_clips.append(ds_clip)
        residuals.append(ds_diff)

    residuals = residuals[::-1]

    mv = MVTools(downsampled_clips[-1])
    mv.analyze(blksize=min(sizes))
    den_base = mv.degrain()

    for x in range(len(factors) - 1):
        scale = factors[x + 1] // factors[x]

        mv.scale_vectors(scale)
        base_up = upsampler.scale(den_base, den_base.width * scale, den_base.height * scale)

        den_last = mv.degrain(residuals[x])
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
