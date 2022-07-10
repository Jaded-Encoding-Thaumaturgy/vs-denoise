from __future__ import annotations

from typing import List, Sequence, TypeVar

import vapoursynth as vs
from vsrgtools.util import PlanesT, normalise_planes

from vsdenoise.knlm import ChannelMode

T = TypeVar('T')


def arr_to_len(array: Sequence[T], length: int = 3) -> List[T]:
    return (list(array) + [array[-1]] * length)[:length]


