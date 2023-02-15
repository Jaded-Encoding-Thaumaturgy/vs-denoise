from __future__ import annotations

from typing import Sequence

from vstools import CustomOverflowError, CustomValueError, fallback

__all__ = [
    'planes_to_mvtools',
    'normalize_thscd'
]


def planes_to_mvtools(input_planes: Sequence[int]) -> int:
    """
    Util function to normalize planes, and converting them to mvtools planes param.

    :param planes:  Sequence of planes to be processed.

    :return:        Value of planes used by mvtools.
    """

    planes = set(input_planes)

    if planes == {0, 1, 2}:
        return 4

    if len(planes) == 1 and planes.intersection({0, 1, 2}):
        return planes.pop()

    if planes == {1, 2}:
        return 3

    raise CustomValueError("Invalid planes specified!", planes_to_mvtools)

def normalize_thscd(thSCD: int | tuple[int | None, int | None] | None,
                     thSAD: int, params_curve: bool, cls):
    thSCD1, thSCD2 = thSCD if isinstance(thSCD, tuple) else (thSCD, None)

    thSCD1 = fallback(thSCD1, round(0.35 * thSAD + 300) if params_curve else 400)
    thSCD2 = fallback(thSCD2, 51)

    if not 1 <= thSCD2 <= 100:
        raise CustomOverflowError('"thSCD[1]" must be between 1 and 100 (inclusive)!', cls)

    thSCD2 = int(thSCD2 / 100 * 255)

    return (thSCD1, thSCD2)
