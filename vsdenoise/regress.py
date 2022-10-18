from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Concatenate, Sequence
from weakref import WeakValueDictionary

from vsexprtools import ExprOp, aka_expr_available, norm_expr
from vskernels import Catrom, Kernel, KernelT, Mitchell, Scaler, ScalerT
from vsrgtools import box_blur
from vstools import (
    P0, P1, InvalidSubsamplingError, P, check_variable, complex_hash, depth, expect_bits, get_subsampling, get_u, get_v,
    get_y, join, split, vs
)

__all__ = [
    'Regression',
    'chroma_reconstruct'
]

_cached_blurs = WeakValueDictionary[int, vs.VideoNode]()


@dataclass
class Regression:
    @dataclass
    class Linear:
        slope: vs.VideoNode
        intercept: vs.VideoNode
        correlation: vs.VideoNode

    class BlurConf:
        def __init__(
            self, func: Callable[Concatenate[vs.VideoNode, P], vs.VideoNode], /, *args: P.args, **kwargs: P.kwargs
        ) -> None:
            self.func = func
            self.args = args
            self.kwargs = kwargs

        @classmethod
        def from_param(
            self, func: Callable[Concatenate[vs.VideoNode, P1], vs.VideoNode] | Regression.BlurConf,
            *args: P1.args, **kwargs: P1.kwargs
        ) -> Regression.BlurConf:
            if isinstance(func, Regression.BlurConf):
                return func.extend(*args, **kwargs)

            return Regression.BlurConf(func, *args, **kwargs)

        def extend(self, *args: Any, **kwargs: Any) -> Regression.BlurConf:
            if args or kwargs:
                return Regression.BlurConf(
                    self.func, *(args or self.args), **(self.kwargs | kwargs)  # type: ignore[arg-type]
                )
            return self

        def __call__(
            self, clip: vs.VideoNode, chroma_only: bool = False, *args: Any, **kwargs: Any
        ) -> vs.VideoNode:
            if not args:
                args = self.args

            kwargs = self.kwargs | kwargs

            out = None

            if chroma_only:
                ckwargs = kwargs | dict(planes=[1, 2])

                key = complex_hash.hash(clip, args, ckwargs)

                got_result = _cached_blurs.get(key, False)

                if isinstance(got_result, vs.VideoNode):
                    return got_result

                try:
                    out = self.func(clip, *args, **ckwargs)  # type: ignore[arg-type]
                except Exception:
                    ...

            if not out:
                key = complex_hash.hash(clip, args, kwargs)

                got_result = _cached_blurs.get(key, False)

                if isinstance(got_result, vs.VideoNode):
                    return got_result

                out = self.func(clip, *args, **kwargs)

            return _cached_blurs.setdefault(key, out)

        def blur(self, clip: vs.VideoNode, chroma_only: bool = False, *args: Any, **kwargs: Any) -> Any:
            return self(clip, chroma_only, *args, **kwargs)

        def get_bases(
            self, clip: vs.VideoNode | Sequence[vs.VideoNode]
        ) -> tuple[list[vs.VideoNode], list[vs.VideoNode], list[vs.VideoNode]]:
            planes = split(clip) if isinstance(clip, vs.VideoNode) else clip

            blur = [self(shifted) for shifted in planes]

            variation = [
                norm_expr([
                    Ex, self(ExprOp.MUL.combine(shifted, suffix=ExprOp.DUP))
                ], 'y x dup * - 0 max')
                for Ex, shifted in zip(blur, planes)
            ]

            var_mul = [
                self(ExprOp.MUL.combine(planes[0], shifted_y))
                for shifted_y in planes[1:]
            ]

            return blur, variation, var_mul

    blur_func: BlurConf | Callable[  # type: ignore[misc]
        Concatenate[vs.VideoNode, P0], vs.VideoNode
    ] = BlurConf(box_blur, radius=2)
    eps: float = 1e-7

    def __post_init__(self) -> None:
        self.blur_conf = Regression.BlurConf.from_param(self.blur_func)

    @classmethod
    def from_param(
        self, func: Callable[Concatenate[vs.VideoNode, P1], vs.VideoNode] | Regression.BlurConf,
        eps: float = 1e-7,
        *args: P1.args, **kwargs: P1.kwargs
    ) -> Regression:
        return Regression(
            Regression.BlurConf.from_param(func, *args, **kwargs), eps
        )

    def linear(
        self, clip: vs.VideoNode | Sequence[vs.VideoNode], eps: float | None = None, *args: Any, **kwargs: Any
    ) -> list[Regression.Linear]:
        eps = eps or self.eps
        blur_conf = self.blur_conf.extend(*args, **kwargs)

        (blur_x, *blur_ys), (var_x, *var_ys), var_mul = blur_conf.get_bases(clip)

        cov_xys = [norm_expr([vm_y, blur_x, Ey], 'x y z * -') for vm_y, Ey in zip(var_mul, blur_ys)]

        slopes = [norm_expr([cov_xy, var_x], f'x y {eps} + /') for cov_xy in cov_xys]

        intercepts = [norm_expr([blur_y, slope, blur_x], 'x y z * -') for blur_y, slope in zip(blur_ys, slopes)]

        corrs = [
            norm_expr([cov_xy, var_x, var_y], f'x dup * y z * {eps} + / sqrt')
            for cov_xy, var_y in zip(cov_xys, var_ys)
        ]

        return [
            Regression.Linear(slope, intercept, correlation)
            for slope, intercept, correlation in zip(slopes, intercepts, corrs)
        ]

    def sloped_corr(
        self, clip: vs.VideoNode | Sequence[vs.VideoNode],
        eps: float | None = None, avg: bool = False, *args: Any, **kwargs: Any
    ) -> list[vs.VideoNode]:
        eps = eps or self.eps
        blur_conf = self.blur_conf.extend(*args, **kwargs)

        (blur_x, *blur_ys), (var_x, *var_ys), var_mul = blur_conf.get_bases(clip)

        corr_slopes = [
            norm_expr(
                [Exys_y, blur_x, Ex_y, var_x, var_y],
                f'x y z * - XYS! XYS@ a {eps} + / XYS@ dup * a b * {eps} + / sqrt 0.5 - 0.5 / 0 max *'
            ) if aka_expr_available else norm_expr(
                [norm_expr([Exys_y, blur_x, Ex_y], 'x y z * -'), var_x, var_y],
                f'x y {eps} + / x dup * y z * {eps} + / sqrt 0.5 - 0.5 / 0 max *'
            )
            for Exys_y, Ex_y, var_y in zip(var_mul, blur_ys, var_ys)
        ]

        if not avg:
            return corr_slopes

        return [
            blur_conf(corr_slope)
            for corr_slope in corr_slopes
        ]


def chroma_reconstruct(
    clip: vs.VideoNode, i444: bool = False,
    kernel: KernelT = Catrom, scaler: ScalerT = Mitchell, downscaler: ScalerT | None = None,
    blur_conf: Callable[
        Concatenate[vs.VideoNode, P], vs.VideoNode
    ] | Regression.BlurConf = Regression.BlurConf(box_blur, radius=2),
    eps: float = 1e-7, *args: P.args, **kwargs: P.kwargs
) -> vs.VideoNode:
    assert check_variable(clip, chroma_reconstruct)

    kernel = Kernel.ensure_obj(kernel, chroma_reconstruct)
    scaler = Scaler.ensure_obj(scaler, chroma_reconstruct)
    downscaler = kernel.ensure_obj(downscaler, chroma_reconstruct)

    if get_subsampling(clip) != '420':
        raise InvalidSubsamplingError(chroma_reconstruct, clip)

    regression = Regression.from_param(blur_conf, eps, *args, **kwargs)

    up, bits = expect_bits(clip, 32)

    y = get_y(up)

    shifted_planes = [
        scaler.scale(plane, up.width, up.height, (0, 0.25)) for plane in [
            downscaler.scale(
                y, up.width // 2, up.height // 2, (0, -.5)
            ), get_u(up), get_v(up)
        ]
    ]

    corr_slopes = regression.sloped_corr(shifted_planes, avg=True)

    y_dw = ExprOp.SUB.combine(y, shifted_planes[0])

    chroma_fix = [
        norm_expr([y_dw, corr_slope, shifted], 'x y * z +')
        for shifted, corr_slope in zip(shifted_planes[1:], corr_slopes)
    ]

    merged = join(y, *chroma_fix)

    if i444:
        return depth(merged, bits)

    return kernel.resample(merged, clip.format)
