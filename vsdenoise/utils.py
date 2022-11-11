from __future__ import annotations

from typing import Sequence

from vstools import CustomValueError


def planes_to_mvtools(input_planes: Sequence[int]) -> int:
    planes = set(input_planes)

    if planes == {0, 1, 2}:
        return 4

    if len(planes) == 1 and planes.intersection({0, 1, 2}):
        return planes.pop()

    if planes == {1, 2}:
        return 3

    raise CustomValueError("Invalid planes specified!", planes_to_mvtools)
