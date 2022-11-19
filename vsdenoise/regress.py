from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Concatenate, Sequence
from weakref import WeakValueDictionary

from vsexprtools import ExprOp, aka_expr_available, norm_expr
from vskernels import Catrom, Kernel, KernelT, Mitchell, Scaler, ScalerT
from vsrgtools import box_blur
from vstools import (
    P1, CustomOverflowError, InvalidSubsamplingError, P, VSFunction, check_variable, complex_hash, depth, expect_bits,
    get_subsampling, get_u, get_v, get_y, join, split, vs
)

__all__ = [
    'Regression',
    'chroma_reconstruct'
]

_cached_blurs = WeakValueDictionary[int, vs.VideoNode]()


@dataclass
class Regression:
    """
    Class for math operation on a clip.

    For more info see `this Wikipedia article <https://en.wikipedia.org/wiki/Regression_analysis>`_.
    """

    @dataclass
    class Linear:
        """
        Representation of a Linear Regression.

        For more info see `this Wikipedia article <https://en.wikipedia.org/wiki/Linear_regression>`_.
        """

        slope: vs.VideoNode
        """
        One of the regression coefficients.

        In simple linear regression the coefficient is the regression slope.
        """

        intercept: vs.VideoNode
        """Component of :py:attr:`slope`, the intercept term."""

        correlation: vs.VideoNode
        """The relationship between the error term and the regressors."""

    class BlurConf:
        """Class for the blur (or averaging filter) used for regression."""

        def __init__(
            self, func: Callable[Concatenate[vs.VideoNode, P], vs.VideoNode], /, *args: P.args, **kwargs: P.kwargs
        ) -> None:
            """
            :param func:        Function used for blurring.
            :param args:        Positional arguments passed to the function.
            :param kwargs:      Keyword arguments passed to the function.
            """

            self.func = func
            self.args = args
            self.kwargs = kwargs

        @classmethod
        def from_param(
            self, func: Callable[Concatenate[vs.VideoNode, P1], vs.VideoNode] | Regression.BlurConf,
            *args: P1.args, **kwargs: P1.kwargs
        ) -> Regression.BlurConf:
            """
            Get a :py:attr:`BlurConf` from generic parameters.

            :param func:        Function used for blurring or already existing config.
            :param args:        Positional arguments passed to the function.
            :param kwargs:      Keyword arguments passed to the function.

            :return:            :py:attr:`BlurConf` object.
            """

            if isinstance(func, Regression.BlurConf):
                return func.extend(*args, **kwargs)

            return Regression.BlurConf(func, *args, **kwargs)

        def extend(self, *args: Any, **kwargs: Any) -> Regression.BlurConf:
            """
            Extend the current config arguments and get a new :py:attr:`BlurConf` object.

            :param args:        Positional arguments passed to the function.
            :param kwargs:      Keyword arguments passed to the function.

            :return:            :py:attr:`BlurConf` object.
            """
            if args or kwargs:
                return Regression.BlurConf(
                    self.func, *(args or self.args), **(self.kwargs | kwargs)  # type: ignore[arg-type]
                )
            return self

        def __call__(
            self, clip: vs.VideoNode, chroma_only: bool = False, *args: Any, **kwargs: Any
        ) -> vs.VideoNode:
            """
            Blur a clip with the current config.

            :param clip:            Clip to be blurred.
            :param chroma_only:     Try only processing chroma.
            :param args:            Positional arguments passed to the function.
            :param kwargs:          Keyword arguments passed to the function.

            :return:                Blurred clip.
            """

            if not args:
                args = self.args

            kwargs = self.kwargs | kwargs

            out = None

            if chroma_only:
                ckwargs = kwargs | dict(planes=[1, 2])

                key = complex_hash.hash(clip, args, ckwargs)

                got_result = _cached_blurs.get(key, None)

                if got_result is not None:
                    return got_result

                try:
                    out = self.func(clip, *args, **ckwargs)  # type: ignore[arg-type]
                except Exception:
                    ...

            if not out:
                key = complex_hash.hash(clip, args, kwargs)

                got_result = _cached_blurs.get(key, None)

                if got_result is not None:
                    return got_result

                out = self.func(clip, *args, **kwargs)

            return _cached_blurs.setdefault(key, out)

        def blur(self, clip: vs.VideoNode, chroma_only: bool = False, *args: Any, **kwargs: Any) -> Any:
            """
            Blur a clip with the current config.

            :param clip:            Clip to be blurred.
            :param chroma_only:     Try only processing chroma.
            :param args:            Positional arguments passed to the function.
            :param kwargs:          Keyword arguments passed to the function.

            :return:                Blurred clip.
            """

            return self(clip, chroma_only, *args, **kwargs)

        def get_bases(
            self, clip: vs.VideoNode | Sequence[vs.VideoNode]
        ) -> tuple[list[vs.VideoNode], list[vs.VideoNode], list[vs.VideoNode]]:
            """
            Get the base elements for a regression.

            :param clip:    Clip or individual planes to be processed.

            :return:        Tuple containing the blurred clips, variations, and relation of the two.
            """

            planes = clip if isinstance(clip, Sequence) else split(clip)

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

    blur_func: BlurConf | VSFunction = BlurConf(box_blur, radius=2)
    """Function used for blurring (averaging)."""

    eps: float = 1e-7
    """Epsilon, used in expressions to avoid division by zero."""

    def __post_init__(self) -> None:
        self.blur_conf = Regression.BlurConf.from_param(self.blur_func)

    @classmethod
    def from_param(
        self, func: Callable[Concatenate[vs.VideoNode, P1], vs.VideoNode] | Regression.BlurConf,
        eps: float = 1e-7,
        *args: P1.args, **kwargs: P1.kwargs
    ) -> Regression:
        """
        Get a :py:attr:`Regression` from generic parameters.

        :param func:        Function used for blurring or a preconfigured :py:attr:`Regression.BlurConf`.
        :param eps:         Epsilon, used in expressions to avoid division by zero.
        :param args:        Positional arguments passed to the blurring function.
        :param kwargs:      Keyword arguments passed to the blurring function.

        :return:            :py:attr:`Regression` object.
        """

        return Regression(
            Regression.BlurConf.from_param(func, *args, **kwargs), eps
        )

    def linear(
        self, clip: vs.VideoNode | Sequence[vs.VideoNode], eps: float | None = None, *args: Any, **kwargs: Any
    ) -> list[Regression.Linear]:
        """
        Perform a simple linear regression.

        :param clip:        Clip or singular planes to be processed.
        :param eps:         Epsilon, used in expressions to avoid division by zero.
        :param args:        Positional arguments passed to the blurring function.
        :param kwargs:      Keyword arguments passed to the blurring function.

        :return:            List of a :py:attr:`Regression.Linear` object for each plane.
        """

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
        self, clip: vs.VideoNode | Sequence[vs.VideoNode], weight: float = 0.5,
        eps: float | None = None, avg: bool = False, *args: Any, **kwargs: Any
    ) -> list[vs.VideoNode]:
        """
        Compute correlation of slopes of a simple regression.

        :param clip:        Clip or individual planes to be processed.
        :param eps:         Epsilon, used in expressions to avoid division by zero.
        :param avg:         Average (blur) the final result.
        :param args:        Positional arguments passed to the blurring function.
        :param kwargs:      Keyword arguments passed to the blurring function.

        :return:            List of clips representing the correlation of slopes.
        """

        eps = eps or self.eps
        blur_conf = self.blur_conf.extend(*args, **kwargs)

        (blur_x, *blur_ys), (var_x, *var_ys), var_mul = blur_conf.get_bases(clip)

        if 0.0 >= weight or weight >= 1.0:
            raise CustomOverflowError(
                '"weight" must be between 0.0 and 1.0 (exclusive)!', self.__class__.sloped_corr, weight
            )

        coeff_x, coeff_y = weight, 1.0 - weight

        corr_slopes = [
            norm_expr(
                [Exys_y, blur_x, Ex_y, var_x, var_y],
                f'x y z * - XYS! XYS@ a {eps} + / XYS@ dup * a b * {eps} + / sqrt {coeff_x} - {coeff_y} / 0 max *'
            ) if aka_expr_available else norm_expr(
                [norm_expr([Exys_y, blur_x, Ex_y], 'x y z * -'), var_x, var_y],
                f'x y {eps} + / x dup * y z * {eps} + / sqrt {coeff_x} - {coeff_y} / 0 max *'
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
    clip: vs.VideoNode, i444: bool = False, weight: float = 0.5,
    kernel: KernelT = Catrom, scaler: ScalerT = Mitchell, downscaler: ScalerT | None = None,
    blur_conf: Callable[
        Concatenate[vs.VideoNode, P], vs.VideoNode
    ] | Regression.BlurConf = Regression.BlurConf(box_blur, radius=2),
    eps: float = 1e-7, *args: P.args, **kwargs: P.kwargs
) -> vs.VideoNode:
    """
    Chroma reconstruction filter using :py:attr:`Regress`.

    This function should be used with care, and not blindly applied to anything.\n
    Ideally you should see how the function works,
    and then mangle the luma of your source to match how your chroma was mangled.

    This function can also return a 4:4:4 clip.\n
    This is not recommended except for very specific cases, like for example where you're
    dealing with a razor-sharp 1080p source with a lot of bright (mostly reddish) colours
    or a lot of high-contrast edges.

    :param clip:        Clip to process.
    :param i444:        Whether to return a 444 clip.
    :param weight:      Weight for correlation of slopes calculation cutoff.
    :param kernel:      Kernel used for resampling and general scaling operations.
    :param scaler:      Scaler used to scale up chroma planes.
    :param downscaler:  Scaler used to downscale the luma plane. Defaults to :py:attr:`kernel`
    :param func:        Function used for blurring or a preconfigured :py:attr:`Regression.BlurConf`.
    :param eps:         Epsilon, used in expressions to avoid division by zero.
    :param args:        Positional arguments passed to the blurring function.
    :param kwargs:      Keyword arguments passed to the blurring function.

    :return:            Clip with demangled chroma.
    """

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

    corr_slopes = regression.sloped_corr(shifted_planes, weight, avg=True)

    y_dw = ExprOp.SUB.combine(y, shifted_planes[0])

    chroma_fix = [
        norm_expr([y_dw, corr_slope, shifted], 'x y * z +')
        for shifted, corr_slope in zip(shifted_planes[1:], corr_slopes)
    ]

    merged = join(y, *chroma_fix)

    if i444:
        return depth(merged, bits)

    return kernel.resample(merged, clip.format)
