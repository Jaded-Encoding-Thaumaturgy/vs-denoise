from __future__ import annotations

from typing import Sequence, List, TypeVar

T = TypeVar('T')


def arr_to_len(array: Sequence[T], length: int = 3) -> List[T]:
    return (list(array) + [array[-1]] * length)[:length]
