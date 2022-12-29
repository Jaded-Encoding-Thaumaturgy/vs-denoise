from __future__ import annotations

from vstools import PlanesT, vs, core, FunctionUtil
from vsexprtools import norm_expr

from .prefilters import Prefilter


__all__ = [
    'wnnm'
]


def wnnm(
    clip: vs.VideoNode, sigma: float | list[float] = 3.0,
    refine: int = 0, rclip: vs.VideoNode | Prefilter | None = None,
    block_size: int = 8, block_step: int = 8, group_size: int = 8,
    bm_range: int = 7, radius: int = 0, ps_num: int = 2, ps_range: int = 4,
    residual: bool = False, adaptive_aggregation: bool = True,
    merge_factor: float = 0.1, planes: PlanesT = None
) -> vs.VideoNode:
    func = FunctionUtil(clip, wnnm, planes, bitdepth=32)

    sigma = func.norm_seq(sigma)

    prev: vs.VideoNode
    denoised: vs.VideoNode

    if isinstance(rclip, Prefilter):
        rclip = rclip(func.work_clip, planes)

    for i in range(refine + 1):
        if i == 0:
            prev = func.work_clip
        elif i == 1:
            prev = denoised
        else:
            prev = norm_expr([func.work_clip, prev, denoised], f'x y - {merge_factor} * z +')

        denoised = core.wnnm.WNNM(
            prev, sigma,
            block_size, block_step, group_size, bm_range, radius,
            ps_num, ps_range, residual, adaptive_aggregation, rclip
        )

    return func.return_clip(denoised)
