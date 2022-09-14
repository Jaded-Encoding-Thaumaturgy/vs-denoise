from __future__ import annotations

from typing import List, Sequence, Tuple, TypeVar

import vapoursynth as vs
from vstools import disallow_variable_format, disallow_variable_resolution

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


def planes_to_mvtools(planes: Sequence[int]) -> Tuple[List[int], int]:
    planes = list(planes)

    if planes == [0, 1, 2]:
        mv_plane = 4
    elif len(planes) == 1 and planes[0] in {0, 1, 2}:
        mv_plane = planes[0]
    elif planes == [1, 2]:
        mv_plane = 3
    else:
        raise CustomValueError("Invalid planes specified!", planes_to_mvtools)

    return list(planes), mv_plane
