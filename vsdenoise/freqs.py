from __future__ import annotations

from typing import Any, Iterable

from vsrgtools import MeanMode
from vstools import (
    CustomValueError, GenericVSFunction, KwargsT, PlanesT, fallback, flatten_vnodes, get_video_format, normalize_seq, vs
)

from .fft import DFTTest
from .mvtools import MVTools

__all__ = [
    'frequency_merge'
]


def frequency_merge(
    *_clips: vs.VideoNode | Iterable[vs.VideoNode], tr: int = 0,
    mode_high: MeanMode | vs.VideoNode = MeanMode.LEHMER, mode_low: MeanMode | vs.VideoNode = MeanMode.ARITHMETIC,
    mode_tr: MeanMode | None = None, lowpass: GenericVSFunction | list[GenericVSFunction] = DFTTest.denoise,
    mean_diff: bool = False, planes: PlanesT = None, mv_args: KwargsT | None = None,
    **kwargs: Any
) -> vs.VideoNode:
    clips = flatten_vnodes(_clips)
    n_clips = len(clips)

    mv_args = mv_args or KwargsT()
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
        low_freqs = mode_low(blurred_clips)

    diffed_clips = []
    for clip, blur in zip(clips, normalize_seq(low_freqs if mean_diff else blurred_clips, n_clips)):
        diffed_clips.append(None if clip == blur else clip.std.MakeDiff(blur))

    if isinstance(mode_high, vs.VideoNode):
        high_freqs = diffed_clips[clips.index(mode_high)]
    else:
        high_freqs = mode_high([clip for clip in diffed_clips if clip])

    if tr:
        mv = MVTools(clip, tr, **mv_args)
        mv.analyze()

        low_freqs = mv.degrain(ref=low_freqs)

        if mode_tr is MeanMode.ARITHMETIC:
            high_freqs = mv.degrain(ref=high_freqs)
        else:
            high_freqs = mv.compensate(mode_tr, ref=high_freqs)  # type: ignore

    return low_freqs.std.MergeDiff(high_freqs)
