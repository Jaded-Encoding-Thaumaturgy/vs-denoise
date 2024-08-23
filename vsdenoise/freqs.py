from __future__ import annotations

from typing import Any, Iterable, Sequence

from vsexprtools import norm_expr
from vsrgtools import MeanMode
from vstools import (
    Align, CustomValueError, FunctionUtil, GenericVSFunction, KwargsT, PlanesT, core, fallback, flatten_vnodes,
    get_plane_sizes, get_video_format, join, normalize_planes, normalize_seq, padder, to_arr, vs
)

from .fft import DFTTest
from .mvtools import MVTools

__all__ = [
    'frequency_merge',

    'deblock_qed'
]


def frequency_merge(
    *_clips: vs.VideoNode | Iterable[vs.VideoNode], tr: int = 0,
    mode_high: MeanMode | vs.VideoNode = MeanMode.LEHMER, mode_low: MeanMode | vs.VideoNode = MeanMode.ARITHMETIC,
    mode_tr: MeanMode | None = None, lowpass: GenericVSFunction | list[GenericVSFunction] = DFTTest.denoise,
    mean_diff: bool = False, planes: PlanesT = None, mv_args: KwargsT | None = None,
    **kwargs: Any
) -> vs.VideoNode:
    """
    Merges the frequency components of the input clips.
    :param _clips:      The clips to merge.
    :param tr:          The temporal radius to use for temporal mean mode.
    :param mode_high:   The mean mode to use for the high frequency components or
                        specifying the clip with the high frequency components.
    :param mode_low:    The mean mode to use for the low frequency components or
                        specifying the clip with the low frequency components.
    :param mode_tr:     The mode to use for temporal mean.
                        If None, it defaults to the value of mode_high.
    :param lowpass:     The lowpass filter to used to extract high frequency components.
                            Example: `lowpass = lambda i: vsrgtools.box_blur(i, passes=3)`
    :param mean_diff:   Whether to use the mean of the lowpass filter and the original clip to
                        extract the low frequency components. Default is False.
    :param planes:      The planes to process. If None, all planes will be processed.
    :param mv_args:     The arguments to pass to the MVTools class.
    """

    clips = flatten_vnodes(_clips)
    n_clips = len(clips)

    planes = normalize_planes(clips[0], planes)

    mv_args = (mv_args or KwargsT()) | KwargsT(planes=planes)
    mode_tr = fallback(mode_tr, MeanMode.LEHMER if isinstance(mode_high, vs.VideoNode) else mode_high)

    if not lowpass:
        raise CustomValueError('You must pass at least one lowpass filter!', frequency_merge)

    formats = {get_video_format(clip).id for clip in clips}

    if len(formats) > 1:
        raise CustomValueError('All clips must have the same format!', frequency_merge)

    blurred_clips = []
    for clip, filt in zip(clips, normalize_seq(lowpass, n_clips)):
        try:
            blurred_clips.append(clip if not filt else filt(clip, planes=planes, **kwargs))
        except Exception:
            blurred_clips.append(clip if not filt else filt(clip, **kwargs))

    if isinstance(mode_low, vs.VideoNode):
        low_freqs = blurred_clips[clips.index(mode_low)]
    else:
        low_freqs = mode_low(blurred_clips, planes=planes, func=frequency_merge)

    diffed_clips = []
    for clip, blur in zip(clips, normalize_seq(low_freqs if mean_diff else blurred_clips, n_clips)):
        diffed_clips.append(None if clip == blur else clip.std.MakeDiff(blur, planes))

    if isinstance(mode_high, vs.VideoNode):
        high_freqs = diffed_clips[clips.index(mode_high)]
    else:
        high_freqs = mode_high([clip for clip in diffed_clips if clip], planes=planes, func=frequency_merge)

    if tr:
        mv = MVTools(clip, tr, **mv_args)
        mv.analyze()

        low_freqs = mv.degrain(ref=low_freqs)

        if mode_tr is MeanMode.ARITHMETIC:
            high_freqs = mv.degrain(ref=high_freqs)
        else:
            high_freqs = mv.compensate(mode_tr, ref=high_freqs)  # type: ignore

    return low_freqs.std.MergeDiff(high_freqs, planes)


def deblock_qed(
    clip: vs.VideoNode,
    quant_edge: int | Sequence[int] = 24,
    quant_inner: int | Sequence[int] = 26,
    alpha_edge: int = 1, beta_edge: int = 2,
    alpha_inner: int = 1, beta_inner: int = 2,
    chroma_str: int = 0,
    align: Align = Align.TOP_LEFT,
    mean_edge: MeanMode = MeanMode.ARITHMETIC,
    mean_inner: MeanMode | None = None,
    planes: PlanesT = None
) -> vs.VideoNode:
    """
    A postprocessed Deblock: Uses full frequencies of Deblock's changes on block borders,
    but DCT-lowpassed changes on block interiours.

    :param clip:            Clip to process.
    :param quant_edge:      Strength of block edge deblocking.
    :param quant_inner:     Strength of block internal deblocking.
    :param alpha_edge:      Halfway "sensitivity" and halfway a strength modifier for borders.
    :param beta_edge:       "Sensitivity to detect blocking" for borders.
    :param alpha_inner:     Halfway "sensitivity" and halfway a strength modifier for block interiors.
    :param beta_inner:      "Sensitivity to detect blocking" for block interiors.
    :param chroma_str:      Chroma deblocking behaviour/strength.
                            - 0 = use proposed method for chroma deblocking
                            - 1 = directly use chroma debl. from the normal Deblock
                            - 2 = directly use chroma debl. from the strong Deblock
    :param align:           Where to align the blocks for eventual padding.
    :param planes:          What planes to process.

    :return:                Deblocked clip
    """
    func = FunctionUtil(clip, deblock_qed, planes)
    if not func.chroma:
        chroma_str = 0

    with padder.ctx(8, align=align) as p8:
        clip = p8.MIRROR(func.work_clip)

        block = padder.COLOR(
            clip.std.BlankClip(
                width=6, height=6, length=1, color=0,
                format=func.work_clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0)
            ), 1, 1, 1, 1, True
        )
        block = core.std.StackHorizontal([block] * (clip.width // block.width))
        block = core.std.StackVertical([block] * (clip.height // block.height))

        if func.chroma:
            blockc = block.std.CropAbs(*get_plane_sizes(clip, 1))
            block = join(block, blockc, blockc)

        block = block * clip.num_frames

        normal, strong = (
            mean_mode(
                clip.deblock.Deblock(quant, alpha, beta, func.norm_planes if chroma_str == cstr else 0)
                for quant in quants
            ) for mean_mode, quants, alpha, beta, cstr in [
                (mean_edge, to_arr(quant_edge), alpha_edge, beta_edge, 1),
                (mean_inner or mean_edge, to_arr(quant_inner), alpha_inner, beta_inner, 2)
            ]
        )

        normalD2, strongD2 = (
            norm_expr([clip, dclip, block], 'z x y - 0 ? range_diff +', planes)
            for dclip in (normal, strong)
        )

        with padder.ctx(16, align=align) as p16:
            strongD2 = p16.CROP(
                norm_expr(p16.MIRROR(strongD2), 'x range_diff - 1.01 * range_diff +', planes)
                .dctf.DCTFilter([1, 1, 0, 0, 0, 0, 0, 0], planes)
            )

        strongD4 = norm_expr([strongD2, normalD2], 'y range_diff = x y ?', planes)
        deblocked = clip.std.MakeDiff(strongD4, planes)

        if func.chroma and chroma_str:
            deblocked = join([deblocked, strong if chroma_str == 2 else normal])

        deblocked = p8.CROP(deblocked)

    return func.return_clip(deblocked)
