from __future__ import annotations

from typing import List, Sequence, TypeVar

import vapoursynth as vs
from vsutil import get_depth

T = TypeVar('T')


def arr_to_len(array: Sequence[T], length: int = 3) -> List[T]:
    return (list(array) + [array[-1]] * length)[:length]


# here until vsutil gets a new release
def get_peak_value(clip: vs.VideoNode, chroma: bool = False) -> float:
    assert clip.format
    return (0.5 if chroma else 1.) if clip.format.sample_type == vs.FLOAT else (1 << get_depth(clip)) - 1.
