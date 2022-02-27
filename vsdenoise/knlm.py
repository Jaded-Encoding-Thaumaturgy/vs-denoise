"""
This module implements a wrapper for KNLMeansCL
"""

from __future__ import annotations

__all__ = ['ChannelMode', 'DeviceType', 'knl_means_cl']

import warnings
from enum import Enum, auto
from vsutil import get_subsampling
from typing import Any, List, Literal, Sequence, final

from .utils import arr_to_len

import vapoursynth as vs

core = vs.core


@final
class ChannelMode(Enum):
    ALL_PLANES = auto()
    LUMA = auto()
    CHROMA = auto()


@final
class DeviceType(str, Enum):
    ACCELERATOR = 'accelerator'
    CPU = 'cpu'
    GPU = 'gpu'
    AUTO = 'auto'


DEVICETYPE = Literal['accelerator', 'cpu', 'gpu', 'auto']


def knl_means_cl(
    clip: vs.VideoNode, /, strength: float | Sequence[float] = 1.2,
    tr: int | Sequence[int] = 1, sr: int | Sequence[int] = 2, simr: int | Sequence[int] = 4,
    channels: ChannelMode = ChannelMode.ALL_PLANES, device_type: DeviceType | DEVICETYPE = DeviceType.AUTO,
    **kwargs: Any
) -> vs.VideoNode:
    """
    Convenience wrapper for KNLMeansCL.\n
    Parameters that accept Sequences will only use the first two elements of it.

    For more information, please refer to the original documentation.
    https://github.com/Khanattila/KNLMeansCL/wiki/Filter-description

    :param clip:            Source clip.
    :param strength:        Controls the strength of the filtering.
                            Larger values will remove more noise.
    :param tr:              Temporal Radius. Set the number of past and future frame that the filter uses
                            for denoising the current frame.
                            tr=0 uses 1 frame, while tr=1 uses 3 frames and so on.
                            Usually, larger it the better the result of the denoising.
                            Temporal size = (2 * tr + 1).
    :param sr:              Search Radius. Set the radius of the search window.
                            sr=1 uses 9 pixel, while sr=2 uses 25 pixels and so on.
                            Usually, larger it the better the result of the denoising.
                            Spatial size = (2 * sr + 1)^2.
    :param simr:            Similarity Radius. Set the radius of the similarity neighbourhood window.
                            The impact on performance is low, therefore it depends on the nature of the noise.
                            Similarity neighbourhood size = (2 * simr + 1)^2.
    :param channels:        Set the colour channels to be denoised
    :param device_type:     Set the OpenCL device.
    :param kwargs:          Additional settings
    :return:                Denoised clip.
    """
    if clip.format is None:
        raise ValueError("knl_means_cl: Variable format clips not supported")

    if isinstance(strength, (float, int)):
        strength = [strength]
    if isinstance(tr, int):
        tr = [tr]
    if isinstance(sr, int):
        sr = [sr]
    if isinstance(simr, int):
        simr = [simr]

    kwargs |= dict(device_type=device_type)

    params: List[Sequence[Any]] = [strength, tr, sr, simr]
    params_doc = ['strength', 'tr', 'sr', 'simr']

    # Handle GRAY and RGB format
    if clip.format.color_family in {vs.GRAY, vs.RGB}:
        for p, doc in zip(params, params_doc):
            if len(p) > 1:
                warnings.warn(
                    f'knl_means_cl: only "{doc}" first parameter will be used '
                    f'since input clip is {clip.format.color_family.name}',
                    UserWarning
                )
        return clip.knlm.KNLMeansCL(
            h=strength[0], d=tr[0], a=sr[0], s=simr[0], channels='auto', **kwargs
        )

    # Handle YUV444 here if these conditions are true
    if all(len(p) < 2 for p in params) and not get_subsampling(clip) and channels == ChannelMode.ALL_PLANES:
        return clip.knlm.KNLMeansCL(
            h=strength[0], d=tr[0], a=sr[0], s=simr[0], channels='YUV', **kwargs
        )

    if channels == ChannelMode.LUMA:
        return clip.knlm.KNLMeansCL(
            h=strength[0], d=tr[0], a=sr[0], s=simr[0], channels='Y', **kwargs
        )

    if channels == ChannelMode.CHROMA:
        return clip.knlm.KNLMeansCL(
            h=strength[1], d=tr[1], a=sr[1], s=simr[1], channels='UV', **kwargs
        )

    normalized_args = [arr_to_len(x, 2) for x in (strength, tr, sr, simr)]

    return core.std.ShufflePlanes([
        clip.knlm.KNLMeansCL(
            h=st, d=tr, a=sr, s=simr, channels=ch, **kwargs
        ) for st, tr, sr, simr, ch in zip(*normalized_args, ('Y', 'UV'))
    ], [0, 1, 2], vs.YUV)
