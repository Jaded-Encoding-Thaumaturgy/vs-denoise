from __future__ import annotations

from typing import cast

from vsexprtools import norm_expr
from vstools import (
    FieldBased, FunctionUtil, KwargsT, PlanesT, UnsupportedFieldBasedError, check_ref_clip, core, depth, get_y, vs
)

from .prefilters import Prefilter

__all__ = [
    'wnnm',
    'bmdegrain'
]


def _recursive_denoise(
    clip: vs.VideoNode, func: vs.Function, ref_key: str | None,
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

        dkwargs = (kwargs | {ref_key: denoised}) if ref_key and denoised else kwargs

        denoised = cast(vs.VideoNode, func(prev, **dkwargs))

    return denoised


def wnnm(
    clip: vs.VideoNode, sigma: float | list[float] = 3.0,
    refine: int = 0, tr: int = 0, ref: vs.VideoNode | Prefilter | None = None,
    block_size: int = 8, block_step: int = 8, group_size: int = 8,
    bm_range: int = 7, ps_num: int = 2, ps_range: int = 4,
    residual: bool = False, adaptive_aggregation: bool = True,
    merge_factor: float = 0.1, self_refine: bool = False, planes: PlanesT = None
) -> vs.VideoNode:
    """
    Weighted Nuclear Norm Minimization Denoise algorithm.

    Block matching, which is popularized by BM3D, finds similar blocks and then stacks together in a 3-D group.
    The similarity between these blocks allows details to be preserved during denoising.

    In contrast to BM3D, which denoises the 3-D group based on frequency domain filtering,
    WNNM utilizes weighted nuclear norm minimization, a kind of low rank matrix approximation.
    Because of this, WNNM exhibits less blocking and ringing artifact compared to BM3D,
    but the computational complexity is much higher. This stage is called collaborative filtering in BM3D.

    For more information, see the `WNNM README <https://github.com/WolframRhodium/VapourSynth-WNNM>`_.

    :param clip:                    Clip to process.
    :param sigma:                   Strength of denoising, valid range is [0, +inf].
                                    If a float is passed, this strength will be applied to every plane.
                                    Values higher than 4.0 are not recommended. Recommended values are [0.35, 1.0].
                                    Default: 3.0.
    :param refine:                  The amount of iterations for iterative regularization.
                                    Default: 0.
    :param radius:                  Temporal radius. To enable spatial-only denoising, set this to 0.
                                    Higher values will rapidly increase filtering time and RAM usage.
                                    Default: 0.
    :param ref:                     Reference clip. Must be the same dimensions and format as input clip.
                                    Alternatively, a :py:class:`Prefilter` can be passed.
                                    Default: None.
    :param block_size:              The size of a block. Blocks are basic processing units.
                                    Larger blocks will take more time to process, but combined with `block_step`,
                                    may result in fewer blocks being processed overall.
                                    Valid ranges are [1, 64]. Default: 8.
    :param block_step:              Sliding step to process every next reference block.
                                    The total amount of blocks to process can be calculated with the following equation:
                                    `(width / block_step) * (height / block_step)`.
                                    Smaller values results in more reference blocks being processed.
                                    Default: 8.
    :param group_size:              Maximum number of similar blocks allowed per group (the 3rd dimension).
                                    Valid range is [1, 256]. By allowing more similar blocks to be grouped together,
                                    fewer blocks will be given to a transformed group,
                                    increasing the denoising strength.
                                    Setting this to 1 means no block matching will be performed.
                                    Default: 8.
    :param bm_range:                Length of the side of the searching neighborhood. Valid range is [0, +inf].
                                    The size of the search window is `(bm_range * 2 + 1) x (bm_range * 2 + 1)`.
                                    Larger values take more time to process, but increases the likelihood
                                    of finding similar patches.
                                    Default: 7.
    :param ps_num:                  The number of matched locations used for the predictive search.
                                    Valid ranges are [1, `group_size`].
                                    Larger values increases the possibility to match similar blocks,
                                    at the cost of taking more processing power.
                                    Default: 2.
    :param ps_range:                Length of the side of the search neighborhood for `pd_num`.
                                    Valid range is [1, +inf]. Default: 4.
    :param residual:                Whether to center blocks before performing collaborative filtering.
                                    Default: False.
    :param adaptive_aggregation:    Whether to aggregate similar blocks weighted by the inverse of the number
                                    of non-zero singular values after WNNM. Default: True.
    :param merge_factor:            Merge amount of the last recalculation into the new one
                                    when performing iterative regularization.
    :param self_refine:             If True, in the iterative recalculation step it will pass the
                                    last recalculation as ref clip instead of the original ``ref``.
    :param planes:                  Planes to process. If None, all planes. Default: None.

    :return:                        Denoised clip.
    """

    if (fb := FieldBased.from_video(clip, False, wnnm)).is_inter:
        raise UnsupportedFieldBasedError('Interlaced input is not supported!', wnnm, fb)

    func = FunctionUtil(clip, wnnm, planes, bitdepth=32)

    sigma = func.norm_seq(sigma)

    if isinstance(ref, Prefilter):
        ref = ref(func.work_clip, planes)
    elif ref is not None:
        ref = depth(ref, 32)
        ref = get_y(ref) if func.luma_only else ref
        check_ref_clip(func.work_clip, ref, func.func)

    return func.return_clip(
        _recursive_denoise(
            func.work_clip, core.wnnm.WNNM, self_refine and 'rclip' or None,
            refine, merge_factor, planes, dict(
                sigma=sigma, block_size=block_size, block_step=block_step, group_size=group_size,
                bm_range=bm_range, radius=tr, ps_num=ps_num, ps_range=ps_range, rclip=ref,
                adaptive_aggregation=adaptive_aggregation, residual=residual
            )
        )
    )


def bmdegrain(
    clip: vs.VideoNode, sigma: float | list[float] = 3.0,
    refine: int = 0, tr: int = 0, ref: vs.VideoNode | Prefilter | None = None,
    block_size: int = 8, block_step: int = 8, group_size: int = 8,
    bm_range: int = 7, ps_num: int = 2, ps_range: int = 4,
    merge_factor: float = 0.1, self_refine: bool = False, planes: PlanesT = None
) -> vs.VideoNode:
    """
    BM3D and mvtools inspired denoiser.

    :param clip:                    Clip to process.
    :param sigma:                   Strength of denoising, valid range is [0, +inf].
                                    If a float is passed, this strength will be applied to every plane.
                                    Values higher than 4.0 are not recommended. Recommended values are [0.35, 1.0].
                                    Default: 3.0.
    :param refine:                  The amount of iterations for iterative regularization.
                                    Default: 0.
    :param radius:                  Temporal radius. To enable spatial-only denoising, set this to 0.
                                    Higher values will rapidly increase filtering time and RAM usage.
                                    Default: 0.
    :param ref:                     Reference clip. Must be the same dimensions and format as input clip.
                                    Alternatively, a :py:class:`Prefilter` can be passed.
                                    Default: None.
    :param block_size:              The size of a block. Blocks are basic processing units.
                                    Larger blocks will take more time to process, but combined with `block_step`,
                                    may result in fewer blocks being processed overall.
                                    Valid ranges are [1, 64]. Default: 8.
    :param block_step:              Sliding step to process every next reference block.
                                    The total amount of blocks to process can be calculated with the following equation:
                                    `(width / block_step) * (height / block_step)`.
                                    Smaller values results in more reference blocks being processed.
                                    Default: 8.
    :param group_size:              Maximum number of similar blocks allowed per group (the 3rd dimension).
                                    Valid range is [1, 256]. By allowing more similar blocks to be grouped together,
                                    fewer blocks will be given to a transformed group,
                                    increasing the denoising strength.
                                    Setting this to 1 means no block matching will be performed.
                                    Default: 8.
    :param bm_range:                Length of the side of the searching neighborhood. Valid range is [0, +inf].
                                    The size of the search window is `(bm_range * 2 + 1) x (bm_range * 2 + 1)`.
                                    Larger values take more time to process, but increases the likelihood
                                    of finding similar patches.
                                    Default: 7.
    :param ps_num:                  The number of matched locations used for the predictive search.
                                    Valid ranges are [1, `group_size`].
                                    Larger values increases the possibility to match similar blocks,
                                    at the cost of taking more processing power.
                                    Default: 2.
    :param ps_range:                Length of the side of the search neighborhood for `pd_num`.
                                    Valid range is [1, +inf]. Default: 4.
    :param merge_factor:            Merge amount of the last recalculation into the new one
                                    when performing iterative regularization.
    :param self_refine:             If True, in the iterative recalculation step it will pass the
                                    last recalculation as ref clip instead of the original ``ref``.
    :param planes:                  Planes to process. If None, all planes. Default: None.

    :return:                        Denoised clip.
    """

    if (fb := FieldBased.from_video(clip, False, bmdegrain)).is_inter:
        raise UnsupportedFieldBasedError('Interlaced input is not supported!', bmdegrain, fb)

    func = FunctionUtil(clip, bmdegrain, planes, bitdepth=32)

    sigma = func.norm_seq(sigma)

    if isinstance(ref, Prefilter):
        ref = ref(func.work_clip, planes)
    elif ref is not None:
        ref = depth(ref, 32)
        ref = get_y(ref) if func.luma_only else ref
        check_ref_clip(clip, ref, func.func)

    return func.return_clip(
        _recursive_denoise(
            func.work_clip, core.bmdegrain.BMDegrain, self_refine and 'rclip' or None,
            refine, merge_factor, planes, dict(
                th_sse=sigma, block_size=block_size, block_step=block_step, group_size=group_size,
                bm_range=bm_range, radius=tr, ps_num=ps_num, ps_range=ps_range, rclip=ref
            )
        )
    )
