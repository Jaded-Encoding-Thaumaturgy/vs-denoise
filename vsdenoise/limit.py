"""
This module contains general functions built on top of base denoisers
to limit the processing either spatially or temporally.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Concatenate, Literal, overload

from vsrgtools import LimitFilterMode, limit_filter
from vstools import CustomIntEnum, P, PlanesT, VSFunction, vs

from .fft import fft3d
from .mvtools import MotionVectors, MVTools, MVToolsPreset, MVToolsPresets

__all__ = [
    'TemporalLimit'
]


@dataclass
class TemporalLimitConfig:
    limiter: VSFunction

    def __call__(
        self, clip: vs.VideoNode,
        thSAD1: int | tuple[int, int], thSAD2: int | tuple[int, int],
        thSCD: int | tuple[int | None, int | None] | None = None, mv_planes: PlanesT = None,
        vectors: MotionVectors | None = None, preset: MVToolsPreset = MVToolsPresets.SMDE
    ) -> vs.VideoNode:
        preset = preset(planes=mv_planes, vectors=vectors)

        NR1 = MVTools(clip, **preset).degrain(thSAD1, 255, thSCD)

        NR1x = limit_filter(NR1, clip, self.limiter(clip), LimitFilterMode.SIMPLE_MIN, 0)

        return MVTools(NR1x, **preset).degrain(thSAD2, 255, thSCD)


class TemporalLimit(CustomIntEnum):
    CUSTOM = -1
    FFT3D = 0

    if TYPE_CHECKING:
        from .limit import TemporalLimit

        @overload
        def __call__(  # type: ignore
            self: Literal[TemporalLimit.CUSTOM],
            limiter: Callable[Concatenate[vs.VideoNode, P], vs.VideoNode], /, *args: P.args, **kwargs: P.kwargs
        ) -> TemporalLimitConfig:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[TemporalLimit.CUSTOM], limiter: vs.VideoNode, /,
        ) -> TemporalLimitConfig:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[TemporalLimit.FFT3D], *, sigma: float, block_size: int, ov: int
        ) -> TemporalLimitConfig:
            ...

        def __call__(self, *args: Any, **kwargs: Any) -> TemporalLimitConfig:  # type: ignore
            ...
    else:
        def __call__(
            self, limiter: vs.VideoNode | VSFunction | None = None, *args: Any, **kwargs: Any
        ) -> TemporalLimitConfig:
            if self is self.CUSTOM:
                if isinstance(limiter, vs.VideoNode):
                    def limit_func(clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
                        return limiter
                else:
                    if limiter is None:
                        def _limiter(clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
                            return clip

                        limiter = _limiter

                    limit_func = partial(limiter, *args, **kwargs)
            elif self is self.FFT3D:
                sigma = kwargs.get('sigma')
                block_size = kwargs.get('block_size')
                ov = kwargs.get('ov')

                def limit_func(clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
                    return fft3d(
                        clip, sigma=sigma, sigma2=sigma * 0.625,
                        sigma3=sigma * 0.375, sigma4=sigma * 0.250,
                        bt=3, bw=block_size, bh=block_size, ow=ov, oh=ov,
                        **kwargs
                    )

            return TemporalLimitConfig(limit_func)
