"""
This module contains general denoising functions built on top of base denoisers.
"""

from __future__ import annotations

from typing import Any, Literal, overload

from vsscale import Waifu2x
from vsscale.scale import BaseWaifu2x
from vstools import CustomIndexError, KwargsNotNone, PlanesT, VSFunction, fallback, normalize_seq, vs

from .mvtools import MotionVectors, MVTools, MVToolsPreset, MVToolsPresets

__all__ = [
    'mc_degrain',

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


def waifu2x_denoise(
    clip: vs.VideoNode, noise: int = 1, model: type[BaseWaifu2x] = Waifu2x.Cunet, **kwargs: Any
) -> vs.VideoNode:
    if noise < 0 or noise > 3:
        raise CustomIndexError('"noise" must be in range 0-3 (inclusive).')

    if not isinstance(model, type):
        model = model.__class__  # type: ignore

    return model(**kwargs).scale(clip, clip.width, clip.height, _static_args=dict(scale=1, noise=noise, force=True))
