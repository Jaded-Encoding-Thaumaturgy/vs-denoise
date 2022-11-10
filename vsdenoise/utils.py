from __future__ import annotations

from typing import List, Sequence, Tuple

from vstools import CustomValueError


def planes_to_mvtools(planes: Sequence[int]) -> Tuple[List[int], int]:
    """
    Util function to normalize planes, and converting them to mvtools planes param.

    :param planes:  Sequence of planes to be processed.

    :return:        Tuple of planes, and mv planes.
    """

    planes = list(planes)

    if planes == [0, 1, 2]:
        mv_plane = 4
    elif len(planes) == 1 and planes[0] in {0, 1, 2}:
        mv_plane = planes[0]
    elif planes == [1, 2]:
        mv_plane = 3
    else:
        raise CustomValueError("Invalid planes specified!", planes_to_mvtools)

    return planes, mv_plane
