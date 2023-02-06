from __future__ import annotations

from typing import cast

from vsexprtools import norm_expr
from vstools import FunctionUtil, KwargsT, PlanesT, core, vs

from .prefilters import Prefilter

__all__ = [
    'wnnm',
    'bmdegrain'
]


def _recursive_denoise(
    clip: vs.VideoNode, func: vs.Function, self_key: str | None,
    refine: int, merge_factor: float, planes: PlanesT, kwargs: KwargsT
) -> vs.VideoNode:
    denoised: vs.VideoNode = None  # type: ignore

    assert refine >= 0

    for i in range(refine + 1):
        if i == 0:
            prev = clip
        elif i == 1:
            prev = denoised
        else:
            prev = norm_expr([clip, prev, denoised], f'x y - {merge_factor} * z +', planes)

        dkwargs = (kwargs | {self_key: denoised}) if self_key and denoised else kwargs

        denoised = cast(vs.VideoNode, func(prev, **dkwargs))

    return denoised


def wnnm(
    clip: vs.VideoNode, sigma: float | list[float] = 3.0,
    refine: int = 0, radius: int = 0, rclip: vs.VideoNode | Prefilter | None = None,
    block_size: int = 8, block_step: int = 8, group_size: int = 8,
    bm_range: int = 7, ps_num: int = 2, ps_range: int = 4,
    residual: bool = False, adaptive_aggregation: bool = True,
    merge_factor: float = 0.1, self_refine: bool = False, planes: PlanesT = None
) -> vs.VideoNode:
    func = FunctionUtil(clip, wnnm, planes, bitdepth=32)

    sigma = func.norm_seq(sigma)

    if isinstance(rclip, Prefilter):
        rclip = rclip(func.work_clip, planes)

    return func.return_clip(
        _recursive_denoise(
            func.work_clip, core.wnnm.WNNM, self_refine and 'rclip' or None,
            refine, merge_factor, planes, dict(
                sigma=sigma, block_size=block_size, block_step=block_step, group_size=group_size,
                bm_range=bm_range, radius=radius, ps_num=ps_num, ps_range=ps_range, rclip=rclip,
                adaptive_aggregation=adaptive_aggregation, residual=residual
            )
        )
    )


def bmdegrain(
    clip: vs.VideoNode, sigma: float | list[float] = 3.0,
    refine: int = 0, radius: int = 0, rclip: vs.VideoNode | Prefilter | None = None,
    block_size: int = 8, block_step: int = 8, group_size: int = 8,
    bm_range: int = 7, ps_num: int = 2, ps_range: int = 4,
    merge_factor: float = 0.1, self_refine: bool = False, planes: PlanesT = None
) -> vs.VideoNode:
    func = FunctionUtil(clip, wnnm, planes, bitdepth=32)

    sigma = func.norm_seq(sigma)

    if isinstance(rclip, Prefilter):
        rclip = rclip(func.work_clip, planes)

    return func.return_clip(
        _recursive_denoise(
            func.work_clip, core.bmdegrain.BMDegrain, self_refine and 'rclip' or None,
            refine, merge_factor, planes, dict(
                th_sse=sigma, block_size=block_size, block_step=block_step, group_size=group_size,
                bm_range=bm_range, radius=radius, ps_num=ps_num, ps_range=ps_range, rclip=rclip
            )
        )
    )
