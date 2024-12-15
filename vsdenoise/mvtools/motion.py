from __future__ import annotations

from typing import Any, NamedTuple

from vstools import vs

from .enums import MVDirection, MVToolsPlugin, FinestMode

__all__ = [
    'MotionVectors',

    'SuperClips'
]


class MotionVectors:
    vmulti: vs.VideoNode
    """Super analyzed clip."""

    kwargs: dict[str, Any]

    temporal_vectors: dict[MVDirection, dict[int, vs.VideoNode]]
    """Dict containing backwards and forwards motion vectors."""

    def __init__(self) -> None:
        self._init_vects()
        self.vmulti = None  # type: ignore
        self.kwargs = dict[str, Any]()

    def _init_vects(self) -> None:
        self.temporal_vectors = {w: {} for w in MVDirection}

    @property
    def has_vectors(self) -> bool:
        """Whether the instance uses bidirectional motion vectors."""

        return bool(
            (self.temporal_vectors[MVDirection.BACK] and self.temporal_vectors[MVDirection.FWRD]) or self.vmulti
        )

    def has_mv(self, direction: MVDirection, delta: int) -> bool:
        """
        Returns whether the motion vector exists.

        :param direction:   Which direction the motion vector was analyzed.
        :param delta:       Delta with which the motion vector was analyzed.

        :return:            Whether the motion vector exists.
        """

        return delta in self.temporal_vectors[direction]

    def get_mv(self, direction: MVDirection, delta: int) -> vs.VideoNode:
        """
        Get the motion vector.

        :param direction:   Which direction the motion vector was analyzed.
        :param delta:       Delta with which the motion vector was analyzed.

        :return:            Motion vector.
        """

        return self.temporal_vectors[direction][delta]

    def set_mv(self, direction: MVDirection, delta: int, vect: vs.VideoNode) -> None:
        """
        Sets the motion vector.

        :param direction:   Which direction the motion vector was analyzed.
        :param delta:       Delta with which the motion vector was analyzed.
        """

        self.temporal_vectors[direction][delta] = vect

    def clear(self) -> None:
        """Deletes all :py:class:`vsdenoise.mvtools.MotionVectors` attributes."""

        del self.vmulti
        self.kwargs.clear()
        self.temporal_vectors.clear()
        self._init_vects()

    def calculate_vectors(
        self, delta: int, mvtools: MVToolsPlugin, supers: SuperClips, recalc: bool, finest: FinestMode, **kwargs: Any
    ) -> None:
        for direction in MVDirection:
            if not recalc:
                vect = mvtools.Analyse(supers.search, isb=direction.isb, delta=delta, **kwargs)
                if finest.after_analyze:
                    vect = mvtools.Finest(vect)
            else:
                vect = mvtools.Recalculate(
                    supers.recalculate, self.get_mv(direction, delta), **kwargs
                )
                if finest.after_recalculate:
                    vect = mvtools.Finest(vect)

            self.set_mv(direction, delta, vect)

    def finest(self, mvtools: MVToolsPlugin) -> None:
        self.temporal_vectors = {
            direction: {
                delta: mvtools.Finest(vect)
                for delta, vect in vectors.items()
            } for direction, vectors in self.temporal_vectors.items()
        }


class SuperClips(NamedTuple):
    base: vs.VideoNode
    render: vs.VideoNode
    search: vs.VideoNode
    recalculate: vs.VideoNode
