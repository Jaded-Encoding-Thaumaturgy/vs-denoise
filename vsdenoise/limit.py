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

from .fft import DFTTest, fft3d
from .mvtools import MotionVectors, MVTools, MVToolsPreset, MVToolsPresets

__all__ = [
    'TemporalLimiter'
]


@dataclass
class TemporalLimiterConfig:
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


class TemporalLimiter(CustomIntEnum):
    CUSTOM = -1
    FFT3D = 0
    DFTTEST = 1

    if TYPE_CHECKING:
        from .limit import TemporalLimiter

        @overload
        def __call__(  # type: ignore
            self: Literal[TemporalLimiter.CUSTOM],
            limiter: Callable[Concatenate[vs.VideoNode, P], vs.VideoNode], /, *args: P.args, **kwargs: P.kwargs
        ) -> TemporalLimiterConfig:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[TemporalLimiter.CUSTOM], limiter: vs.VideoNode, /,
        ) -> TemporalLimiterConfig:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[TemporalLimiter.FFT3D], *, sigma: float, block_size: int, ov: int
        ) -> TemporalLimiterConfig:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[TemporalLimiter.DFTTEST], *, sigma_low: float, sigma_high: float | None = None
        ) -> TemporalLimiterConfig:
            ...

        def __call__(self, *args: Any, **kwargs: Any) -> TemporalLimiterConfig:  # type: ignore
            ...
    else:
        def __call__(
            self, limiter: vs.VideoNode | VSFunction | None = None, *args: Any, **kwargs: Any
        ) -> TemporalLimiterConfig:
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
            elif self is self.DFTTEST:
                sigma_low = kwargs.get('sigma_low')
                sigma_high = kwargs.get('sigma_high', sigma_low)

                def limit_func(clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
                    return DFTTest.denoise(clip, {0: sigma_low, 1: sigma_high}, **kwargs)

            return TemporalLimiterConfig(limit_func)
