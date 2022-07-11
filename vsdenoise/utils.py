from __future__ import annotations
import vapoursynth as vs
from typing import List, Sequence, Tuple, TypeVar

from vsutil import disallow_variable_format, disallow_variable_resolution

from vsdenoise.knlm import ChannelMode

T = TypeVar('T')


def arr_to_len(array: Sequence[T], length: int = 3) -> List[T]:
    return (list(array) + [array[-1]] * length)[:length]


@disallow_variable_format
@disallow_variable_resolution
def check_ref_clip(src: vs.VideoNode, ref: vs.VideoNode | None) -> None:
    if ref is None:
        return

    assert src.format and ref.format

    if ref.format.id != src.format.id:
        raise ValueError("Ref clip format must match the source clip's!")

    if ref.width != src.width or ref.height != src.height:
        raise ValueError("Ref clip sizes must match the source clip's!")


def planes_to_channelmode(planes: Sequence[int]) -> ChannelMode:
    planes = list(planes)

    if planes == [0]:
        return ChannelMode.LUMA

    if 0 not in planes:
        return ChannelMode.CHROMA

    return ChannelMode.ALL_PLANES


def planes_to_mvtools(planes: Sequence[int]) -> Tuple[List[int], int]:
    planes = list(planes)

    if planes == [0, 1, 2]:
        mv_plane = 4
    elif len(planes) == 1 and planes[0] in {0, 1, 2}:
        mv_plane = planes[0]
    elif planes == [1, 2]:
        mv_plane = 3
    else:
        raise ValueError("Invalid planes specified!")

    return list(planes), mv_plane
