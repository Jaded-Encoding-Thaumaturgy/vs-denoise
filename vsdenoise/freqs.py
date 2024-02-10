from __future__ import annotations

from typing import Any, Callable, Iterable

from vskernels import FmtConv
from vsrgtools import MeanMode
from vstools import (
    CustomValueError, GenericVSFunction, KwargsT, PlanesT, fallback, flatten_vnodes, get_video_format, normalize_seq, vs
)

from .fft import DFTTest
from .mvtools import MVTools

__all__ = [
    'frequency_merge',
    'dvd_deblur'
]


def frequency_merge(
    *_clips: vs.VideoNode | Iterable[vs.VideoNode], tr: int = 0,
    mode_high: MeanMode = MeanMode.LEHMER, mode_low: MeanMode = MeanMode.ARITHMETIC,
    mode_tr: MeanMode | None = None, lowpass: GenericVSFunction | list[GenericVSFunction] = DFTTest.denoise,
    mean_diff: bool = False, planes: PlanesT = None, mv_args: KwargsT | None = None,
    **kwargs: Any
) -> vs.VideoNode:
    clips = flatten_vnodes(_clips)
    n_clips = len(clips)

    mv_args = mv_args or KwargsT()
    mode_tr = fallback(mode_tr, mode_low)

    if not lowpass:
        raise CustomValueError('You must pass at least one lowpass filter!', frequency_merge)

    formats = {get_video_format(clip).id for clip in clips}

    if len(formats) > 1:
        raise CustomValueError('All clips must have the same format!', frequency_merge)

    blurred_clips = []
    for clip, filt in zip(clips, normalize_seq(lowpass, n_clips)):
        try:
            blurred_clips.append(filt(clip, planes=planes, **kwargs))
        except Exception:
            blurred_clips.append(filt(clip, **kwargs))

    low_freqs = mode_low(blurred_clips)

    to_diff_clips = normalize_seq(low_freqs if mean_diff else blurred_clips, n_clips)

    diffed_clips = [
        clip.std.MakeDiff(blur)
        for clip, blur in zip(clips, to_diff_clips)
    ]

    high_freqs = mode_high(diffed_clips)

    if tr:
        mv = MVTools(clip, tr, **mv_args)
        mv.analyze()

        low_freqs = mv.degrain(ref=low_freqs)

        if mode_high is MeanMode.ARITHMETIC:
            high_freqs = mv.degrain(ref=high_freqs)
        else:
            high_freqs = mv.compensate(mode_high, ref=high_freqs)  # type: ignore

    return low_freqs.std.MergeDiff(high_freqs)


def dvd_deblur(
    clip: vs.VideoNode, clean: vs.VideoNode,
    blur: tuple[float, float] | float, taps: int = 4,
    post_process: Callable[[vs.VideoNode, vs.VideoNode], vs.VideoNode] = None,
    compare: bool = False,
) -> vs.VideoNode:
    """
    Function to deblur DVDs using a reference clean source.

    Blurring on JP DVDs is oftentimes the result of horizontal resampling
    (anamorphic -> square pixels -> anamorphic) using Lanczos with a blur value.
    This is usually not the case for DVDs from other regions.

    This function works by performing the same resampling on the "clean" clip,
    and creating a diff clip to try and restore the original clip.

    :param clip:            Clip to process.
    :param clean:           "Clean" clip that does not have the resample blurring.
    :param blur:            Blur values to pass on to Lanczos. As blur values may differ
                            between the resampling steps, a tuple of floats is accepted.
                            If only 1 float is passed, it will be used for both steps.
    :param taps:            Taps param for the Lanczos kernel. Default: 4.
    :param post_process:    Function for performing post-processing on the clip prior to the diffing step.
                            The function must accept two input clips (clip, mangled) and return a clip.
                            Default: None.
    :param compare:         Return the mangled clip so it can be compared to the source clip. Default: False.

    :return:                Demangled input clip.
    """

    if not isinstance(blur, tuple):
        blur = (blur, blur)

    class FmtConvLanczos(FmtConv):
        _kernel = "lanczos"

    lpf = FmtConvLanczos(taps)

    mangle = lpf.scale(clean, clip.width, clip.height, fh=1 / blur[0])
    mangle = lpf.scale(mangle, clip.width, clip.height, fh=1 / blur[1])

    if compare:
        return mangle

    if post_process is not None:
        clip = post_process(clip, mangle)

    diff = clean.std.MakeDiff(mangle)
    return clip.std.MergeDiff(diff)
