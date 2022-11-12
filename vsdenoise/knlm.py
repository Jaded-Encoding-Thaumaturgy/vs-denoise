"""
This module implements a wrapper for KNLMeansCL
"""

from __future__ import annotations

__all__ = ['ChannelMode', 'DeviceType', 'knl_means_cl']

import warnings
from enum import Enum, auto
from typing import Any, List, Literal, Sequence, final

from vstools import core, disallow_variable_format, vs


@final
class ChannelMode(Enum):
    """Enum representing the KNLMeansCL channel operation mode."""

    ALL_PLANES = auto()
    """Process all planes."""

    LUMA = auto()
    """Only process luma in YUV/GRAY."""

    CHROMA = auto()
    """Only process chroma in YUV."""

    @classmethod
    def from_planes(cls, planes: Sequence[int]) -> ChannelMode:
        """
        Get :py:attr:`ChannelMode` from a traditional ``planes`` param.

        :param planes:  Sequence of planes to be processed.

        :return:        :py:attr:`ChannelMode` value.
        """

        planes = list(planes)

        if planes == [0]:
            return cls.LUMA

        if 0 not in planes:
            return cls.CHROMA

        return cls.ALL_PLANES


@final
class DeviceType(str, Enum):
    """Enum representing available OpenCL device on which to run the plugin."""

    ACCELERATOR = 'accelerator'
    """Dedicated OpenCL accelerators."""

    CPU = 'cpu'
    """An OpenCL device that is the host processor."""

    GPU = 'gpu'
    """An OpenCL device that is a GPU."""

    AUTO = 'auto'
    """Automatically detect device. Priority is "accelerator" -> "gpu" -> "cpu"."""


DEVICETYPE = Literal['accelerator', 'cpu', 'gpu', 'auto']


@disallow_variable_format
def knl_means_cl(
    clip: vs.VideoNode, /, strength: float | Sequence[float] = 1.2,
    tr: int | Sequence[int] = 1, sr: int | Sequence[int] = 2, simr: int | Sequence[int] = 4,
    channels: ChannelMode = ChannelMode.ALL_PLANES, device_type: DeviceType | DEVICETYPE = DeviceType.AUTO,
    **kwargs: Any
) -> vs.VideoNode:
    """
    Convenience wrapper for KNLMeansCL.\n
    Parameters that accept Sequences will only use the first two elements of it.

    For more information, please refer to the
    `original documentation <https://github.com/Khanattila/KNLMeansCL/wiki/Filter-description>`_.

    :param clip:            Source clip.
    :param strength:        Controls the strength of the filtering.\n
                            Larger values will remove more noise.
    :param tr:              Temporal Radius. Temporal size = `(2 * tr + 1)`.\n
                            Sets the number of past and future frames to uses for denoising the current frame.\n
                            tr=0 uses 1 frame, while tr=1 uses 3 frames and so on.\n
                            Usually, larger values result in better denoising.
    :param sr:              Search Radius. Spatial size = `(2 * sr + 1)^2`.\n
                            Sets the radius of the search window.\n
                            sr=1 uses 9 pixel, while sr=2 uses 25 pixels and so on.\n
                            Usually, larger values result in better denoising.
    :param simr:            Similarity Radius. Similarity neighbourhood size = `(2 * simr + 1) ** 2`.\n
                            Sets the radius of the similarity neighbourhood window.\n
                            The impact on performance is low, therefore it depends on the nature of the noise.
    :param channels:        Set the clip channels to be denoised.
    :param device_type:     Set the OpenCL device to use for processing.
    :param kwargs:          Additional arguments to pass to knlmeansCL.

    :return:                Denoised clip.
    """

    assert clip.format

    if isinstance(strength, (float, int)):
        strength = [strength]
    if isinstance(tr, int):
        tr = [tr]
    if isinstance(sr, int):
        sr = [sr]
    if isinstance(simr, int):
        simr = [simr]

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
            h=strength[0], d=tr[0], a=sr[0], s=simr[0],
            channels='auto', device_type=device_type, **kwargs
        )

    # Handle YUV444 here if these conditions are true
    if all(len(p) < 2 for p in params) and clip.format.subsampling_w == clip.format.subsampling_h == 0 \
            and channels == ChannelMode.ALL_PLANES:
        return clip.knlm.KNLMeansCL(
            h=strength[0], d=tr[0], a=sr[0], s=simr[0],
            channels='YUV', device_type=device_type, **kwargs
        )

    strength = (list(strength) + [strength[-1]] * (2 - len(strength)))[:2]
    tr = (list(tr) + [tr[-1]] * (2 - len(tr)))[:2]
    sr = (list(sr) + [sr[-1]] * (2 - len(sr)))[:2]
    simr = (list(simr) + [simr[-1]] * (2 - len(simr)))[:2]

    if channels == ChannelMode.LUMA:
        return clip.knlm.KNLMeansCL(
            h=strength[0], d=tr[0], a=sr[0], s=simr[0],
            channels='Y', device_type=device_type, **kwargs
        )
    if channels == ChannelMode.CHROMA:
        return clip.knlm.KNLMeansCL(
            h=strength[1], d=tr[1], a=sr[1], s=simr[1],
            channels='UV', device_type=device_type, **kwargs
        )

    return core.std.ShufflePlanes(
        [
            clip.knlm.KNLMeansCL(
                h=strength[i], d=tr[i], a=sr[i], s=simr[i],
                channels=('Y', 'UV')[i], device_type=device_type, **kwargs
            ) for i in range(2)
        ], [0, 1, 2], vs.YUV
    )
