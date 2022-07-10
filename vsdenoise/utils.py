from __future__ import annotations

from typing import List, Sequence, TypeVar

from vsrgtools.util import PlanesT

from vsdenoise.knlm import ChannelMode

T = TypeVar('T')


def arr_to_len(array: Sequence[T], length: int = 3) -> List[T]:
    return (list(array) + [array[-1]] * length)[:length]


def planes_to_channelmode(planes: PlanesT) -> ChannelMode:
    if planes == [0]:
        return ChannelMode.LUMA

    if 0 not in planes:
        return ChannelMode.CHROMA

    return ChannelMode.ALL_PLANES
