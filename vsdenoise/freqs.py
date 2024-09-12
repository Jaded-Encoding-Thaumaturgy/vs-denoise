from __future__ import annotations

from typing import Any, Iterable

from vsrgtools import MeanMode
from vstools import (
    CustomValueError, GenericVSFunction, KwargsT, PlanesT, fallback, flatten_vnodes,
    normalize_planes, normalize_seq, vs, FormatsMismatchError
)

from .fft import DFTTest
from .mvtools import MVTools

__all__ = [
    'frequency_merge',
]


def frequency_merge(
    *_clips: vs.VideoNode | Iterable[vs.VideoNode], tr: int = 0,
    mode_high: MeanMode | vs.VideoNode = MeanMode.LEHMER, mode_low: MeanMode | vs.VideoNode = MeanMode.ARITHMETIC,
    mode_tr: MeanMode | None = None, lowpass: GenericVSFunction | list[GenericVSFunction] = DFTTest.denoise,
    planes: PlanesT = None, mv_args: KwargsT | None = None,
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

    FormatsMismatchError.check(frequency_merge, *clips)

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
    for clip, blur in zip(clips, normalize_seq(blurred_clips, n_clips)):
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
