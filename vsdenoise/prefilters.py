"""
This module implements prefilters for denoisers
"""

from __future__ import annotations

from math import ceil, sin
from typing import TYPE_CHECKING, Any, Literal, overload

from vsaa import Nnedi3, Znedi3
from vsexprtools import ExprOp, norm_expr, aka_expr_available
from vskernels import Bicubic, BicubicZopti, Bilinear, Scaler, ScalerT, KernelT
from vsrgtools import gauss_blur, min_blur, replace_low_frequencies, blur
from vstools import (
    ColorRange, CustomRuntimeError, DitherType, PlanesT, core, depth, disallow_variable_format,
    disallow_variable_resolution, get_depth, get_neutral_value, get_peak_value, get_y, join, normalize_planes,
    scale_8bit, scale_value, split, vs, CustomEnum, CustomIntEnum, clamp, check_variable, SingleOrArrOpt, SingleOrArr,
    ConvMode
)

from .bm3d import BM3D as BM3DM
from .bm3d import BM3DCPU, AbstractBM3D, BM3DCuda, BM3DCudaRTC, Profile
from .knlm import DEVICETYPE, ChannelMode, DeviceType, knl_means_cl

__all__ = [
    'Prefilter', 'prefilter_to_full_range',
    'PelType'
]


class PrefilterBase(CustomIntEnum):
    def __call__(  # type: ignore
        self: Prefilter, clip: vs.VideoNode, /, planes: PlanesT = None, **kwargs: Any
    ) -> vs.VideoNode:
        """@@PLACEHOLDER@@"""
        assert check_variable(clip, self)

        pref_type = Prefilter.MINBLUR3 if self == Prefilter.AUTO else self

        bits = get_depth(clip)
        peak = get_peak_value(clip)
        planes = normalize_planes(clip, planes)

        if pref_type == Prefilter.NONE:
            return clip

        if pref_type.value in {0, 1, 2}:
            return min_blur(clip, pref_type.value, planes)

        if pref_type == Prefilter.MINBLURFLUX:
            temp_thr, spat_thr = kwargs.get('temp_thr', 2), kwargs.get('spat_thr', 2)
            return min_blur(clip, 2, planes).flux.SmoothST(temp_thr, spat_thr, planes)

        if pref_type == Prefilter.DFTTEST:
            dftt_args = dict[str, Any](
                tbsize=1, sbsize=12, sosize=6, swin=2, slocation=[
                    0.0, 4.0, 0.2, 9.0, 1.0, 15.0
                ]
            ) | kwargs

            dfft = clip.dfttest.DFTTest(**dftt_args)

            i, j = (scale_value(x, 8, bits, range_out=ColorRange.FULL) for x in (16, 75))

            pref_mask = norm_expr(
                get_y(clip),
                f'x {i} < {peak} x {j} > 0 {peak} x {i} - {peak} {j} {i} - / * - ? ?'
            )

            return dfft.std.MaskedMerge(clip, pref_mask, planes)

        if pref_type == Prefilter.KNLMEANSCL:
            kwargs |= dict(strength=7.0, tr=1, sr=2, simr=2) | kwargs | dict(channels=ChannelMode.from_planes(planes))
            knl = knl_means_cl(clip, **kwargs)

            return replace_low_frequencies(knl, clip, 600 * (clip.width / 1920), False, planes)

        if pref_type == Prefilter.BM3D:
            bm3d_arch: type[AbstractBM3D] = kwargs.pop('arch', BM3DCuda if kwargs.pop('gpu', False) else BM3DM)

            if bm3d_arch is BM3DM:
                sigma, profile = 10, Profile.FAST
            elif bm3d_arch is BM3DCPU:
                sigma, profile = 10, Profile.LOW_COMPLEXITY
            elif bm3d_arch in (BM3DCuda, BM3DCudaRTC):
                sigma, profile = 8, Profile.NORMAL
            else:
                raise ValueError

            sigma = kwargs.pop('sigma', sigma)

            sigmas = [sigma if 0 in planes else 0, sigma if (1 in planes or 2 in planes) else 0]

            bm3d_args = dict[str, Any](sigma=sigmas, radius=1, profile=profile) | kwargs

            return bm3d_arch(clip, **bm3d_args).clip

        if pref_type == Prefilter.SCALEDBLUR:
            scale = kwargs.pop('scale', 2)
            downscaler = Scaler.ensure_obj(kwargs.pop('downscaler', Bilinear))
            upscaler = downscaler.ensure_obj(kwargs.pop('upscaler', downscaler))

            downscale = downscaler.scale(clip, clip.width // scale, clip.height // scale)

            boxblur = blur(downscale, kwargs.pop('radius', 1), kwargs.pop('mode', ConvMode.SQUARE), planes)

            return upscaler.scale(boxblur, clip.width, clip.height)

        if pref_type == Prefilter.GAUSSBLUR:
            if 'sharp' not in kwargs and 'sigma' not in kwargs:
                kwargs |= dict(sigma=1.0)

            dgd = gauss_blur(clip, **(kwargs | dict[str, Any](planes=planes)))

            return replace_low_frequencies(dgd, clip, clip.width / 2)

        if pref_type in {Prefilter.GAUSSBLUR1, Prefilter.GAUSSBLUR2}:
            boxblur = blur(clip, kwargs.pop('radius', 1), kwargs.get('mode', ConvMode.SQUARE), planes=planes)

            if 'sharp' not in kwargs and 'sigma' not in kwargs:
                kwargs |= dict(sigma=1.75)

            strg = clamp(kwargs.pop('strenght', 50 if pref_type == Prefilter.GAUSSBLUR2 else 90), 0, 98) + 1

            gaussblur = gauss_blur(boxblur, **(kwargs | dict[str, Any](planes=planes)))

            if pref_type == Prefilter.GAUSSBLUR2:
                i2, i7 = (scale_8bit(clip, x) for x in (2, 7))

                merge_expr = f'x {i7} + y < x {i2} + x {i7} - y > x {i2} - x {strg} * y {100 - strg} * + 100 / ? ?'
            else:
                merge_expr = f'x {strg / 100} * y {(100 - strg) / 100} * +'

            return norm_expr([gaussblur, clip], merge_expr, planes)

        return clip


class Prefilter(PrefilterBase):
        """
    Enum representing available filters.
    These are mainly thought as prefilters for :py:attr:`MVTools`,
    but can be used standalone as-is.
    """

    AUTO = -2
    """Automatically decide what filter to use."""

    NONE = -1
    """Don't use any filters. Will return the clip as-is."""

    MINBLUR1 = 0
    """A gaussian/temporal median merge of radius 1."""

    MINBLUR2 = 1
    """A gaussian/temporal median merge of radius 2."""

    MINBLUR3 = 2
    """A gaussian/temporal median merge of radius 3."""

    MINBLURFLUX = 3
    """:py:attr:`MINBLUR2` with temporal/spatial average."""

    DFTTEST = 4
    """Denoising in frequency domain with dfttest with adaptive mask for retaining lineart."""

    KNLMEANSCL = 5
    """Denoising with KNLMeansCL, then postprocessed to remove low frequencies."""

    BM3D = 6
    """Normal spatio-temporal denoising with the BM3D denoiser."""

    SCALEDBLUR = 7
    """@@PLACEHOLDER@@"""

    GAUSSBLUR = 8
    """@@PLACEHOLDER@@"""

    GAUSSBLUR1 = 9
    """@@PLACEHOLDER@@"""

    GAUSSBLUR2 = 10
    """@@PLACEHOLDER@@"""


    if TYPE_CHECKING:
        from .prefilters import Prefilter

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.MINBLURFLUX], clip: vs.VideoNode, /, planes: PlanesT = None,
            *, temp_thr: int = 2, spat_thr: int = 2
        ) -> vs.VideoNode:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.DFTTEST], clip: vs.VideoNode, /, planes: PlanesT = None,
            *,
            tbsize: int = 1, sbsize: int = 12, sosize: int = 6, swin: int = 2,
            slocation: SingleOrArr[float] = [0.0, 4.0, 0.2, 9.0, 1.0, 15.0],
            ftype: int | None = None, sigma: float | None = None, sigma2: float | None = None,
            pmin: float | None = None, pmax: float | None = None, smode: int | None = None,
            tmode: int | None = None, tosize: int | None = None, twin: int | None = None,
            sbeta: float | None = None, tbeta: float | None = None, zmean: int | None = None,
            f0beta: float | None = None, nlocation: SingleOrArrOpt[int] = None, alpha: float | None = None,
            ssx: SingleOrArrOpt[float] = None, ssy: SingleOrArrOpt[float] = None, sst: SingleOrArrOpt[float] = None,
            ssystem: int | None = None, opt: int | None = None
        ) -> vs.VideoNode:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.KNLMEANSCL], clip: vs.VideoNode, /, planes: PlanesT = None,
            *, strength: SingleOrArr[float] = 7.0, tr: SingleOrArr[int] = 1, sr: SingleOrArr[int] = 2,
            simr: SingleOrArr[int] = 2, device_type: DEVICETYPE | DeviceType = DeviceType.AUTO, **kwargs: Any
        ) -> vs.VideoNode:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.BM3D], clip: vs.VideoNode, /, planes: PlanesT = None,
            *, arch: type[AbstractBM3D] = ..., gpu: bool = False,
            sigma: SingleOrArr[float] = ..., radius: SingleOrArr[int] = 1,
            profile: Profile = ..., ref: vs.VideoNode | None = None, refine: int = 1,
            yuv2rgb: KernelT = Bicubic, rgb2yuv: KernelT = Bicubic
        ) -> vs.VideoNode:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.SCALEDBLUR], clip: vs.VideoNode, /, planes: PlanesT = None,
            scale: int = 2, radius: int = 1, mode: ConvMode = ConvMode.SQUARE,
            downscaler: ScalerT = Bilinear, upscaler: ScalerT | None = None
        ) -> vs.VideoNode:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.GAUSSBLUR], clip: vs.VideoNode, /, planes: PlanesT = None,
            *, sigma: float | None = 1.0, sharp: float | None = None, mode: ConvMode = ConvMode.SQUARE
        ) -> vs.VideoNode:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.GAUSSBLUR1], clip: vs.VideoNode, /, planes: PlanesT = None,
            *, radius: int = 1, strength: int = 90, sigma: float | None = 1.75,
            sharp: float | None = None, mode: ConvMode = ConvMode.SQUARE
        ) -> vs.VideoNode:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.GAUSSBLUR2], clip: vs.VideoNode, /, planes: PlanesT = None,
            *, radius: int = 1, strength: int = 50, sigma: float | None = 1.75,
            sharp: float | None = None, mode: ConvMode = ConvMode.SQUARE
        ) -> vs.VideoNode:
            ...

        @overload
        def __call__(self, clip: vs.VideoNode, /, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
            ...

        def __call__(  # type: ignore
            self, clip: vs.VideoNode, /, planes: PlanesT = None, **kwargs: Any
        ) -> vs.VideoNode:
            ...


def prefilter_to_full_range(pref: vs.VideoNode, range_conversion: float, planes: PlanesT = None) -> vs.VideoNode:
    """@@PLACEHOLDER@@"""

    planes = normalize_planes(pref, planes)

    work_clip, *chroma = split(pref) if planes == [0] else (pref, )

    assert (fmt := work_clip.format) and pref.format

    bits = get_depth(pref)
    is_integer = fmt.sample_type == vs.INTEGER

    # Luma expansion TV->PC (up to 16% more values for motion estimation)
    if range_conversion >= 1.0:
        neutral = get_neutral_value(work_clip, True)
        max_val = get_peak_value(work_clip)

        c = sin(0.0625)
        k = (range_conversion - 1) * c

        if is_integer:
            t = f'x {scale_8bit(pref, 16)} - {scale_8bit(pref, 219)} / {ExprOp.clamp(0, 1)}'
        else:
            t = ExprOp.clamp(0, 1, 'x').to_str()

        head = f'{k} {1 + c} {(1 + c) * c}'

        if aka_expr_available:
            head = f'{t} T! {head}'
            t = 'T@'

        luma_expr = f'{head} {t} {c} + / - * {t} 1 {k} - * +'

        if is_integer:
            luma_expr += f' {max_val} *'

        pref_full = norm_expr(work_clip, (luma_expr, f'x {neutral} - 128 * 112 / {neutral} +'), planes)
    elif range_conversion > 0.0:
        pref_full = work_clip.retinex.MSRCP(None, range_conversion, None, False, True)
    else:
        pref_full = depth(
            work_clip, bits, range_out=ColorRange.FULL, range_in=ColorRange.LIMITED, dither_type=DitherType.NONE
        )

    if chroma:
        return join(pref_full, *chroma, family=pref.format.color_family)

    return pref_full


if TYPE_CHECKING:
    PelTypeBase = CustomEnum
else:
    class PelTypeBase(CustomEnum):
        ...

    class CUSTOM(Scaler):
        def __init__(self, scaler: str | type[Scaler] | Scaler, **kwargs: Any) -> None:
            self.scaler = Scaler.ensure_obj(scaler)
            self.kwargs = kwargs

        @disallow_variable_format
        @disallow_variable_resolution
        def __call__(
            self, clip: vs.VideoNode, pel: int, subpixel: int = 3,
            default: ScalerT | None = None, **kwargs: Any
        ) -> vs.VideoNode:
            return PelType.__call__(self.scaler, clip, pel, subpixel, default, **(self.kwargs | kwargs))

        def scale(
            self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
        ) -> vs.VideoNode:
            self.scaler.scale(clip, width, height, shift, **kwargs)

    BICUBIC = CUSTOM(Bicubic)
    WIENER = CUSTOM(BicubicZopti)

    PelTypeBase.CUSTOM = CUSTOM
    PelTypeBase.BICUBIC = BICUBIC
    PelTypeBase.WIENER = WIENER


class PelType(int, PelTypeBase):
    AUTO = -1
    """@@PLACEHOLDER@@"""

    NONE = 0
    """@@PLACEHOLDER@@"""

    NNEDI3 = 4
    """@@PLACEHOLDER@@"""

    if TYPE_CHECKING:
        from .prefilters import PelType

        class CUSTOM(Scaler, PelType):  # type: ignore
            """@@PLACEHOLDER@@"""

            def __init__(self, scaler: str | Type[Scaler] | Scaler, **kwargs: Any) -> None:
                """@@PLACEHOLDER@@"""

            def scale(  # type: ignore
                self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
            ) -> vs.VideoNode:
                ...

        BICUBIC: CUSTOM
        """@@PLACEHOLDER@@"""

        WIENER: CUSTOM
        """@@PLACEHOLDER@@"""

        def __new__(cls, value: int) -> PelType:
            ...

        def __init__(self, value: int) -> None:
            ...

    @disallow_variable_format
    @disallow_variable_resolution
    def __call__(
        pel_type: Scaler | PelType, clip: vs.VideoNode, pel: int, subpixel: int = 3,
        default: ScalerT | PelType | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        """@@PLACEHOLDER@@"""

        assert clip.format

        if pel_type is PelType.NONE or pel <= 1:
            return clip

        if pel_type is PelType.AUTO:
            if subpixel == 4:
                pel_type = PelType.NNEDI3
            elif default:
                pel_type = default if isinstance(default, PelType) else Scaler.ensure_obj(default)
            else:
                pel_type = PelType(1 << 3 - ceil(clip.height / 1000))

        if pel_type == PelType.NNEDI3:
            nnedicl, nnedi, znedi = (hasattr(core, ns) for ns in ('nnedi3cl', 'nnedi3', 'znedi3'))
            do_nnedi = (nnedicl or nnedi) and not znedi

            if not any((nnedi, znedi, nnedicl)):
                raise CustomRuntimeError('Missing any nnedi3 implementation!', PelType.NNEDI3)

            kwargs |= {'nsize': 0, 'nns': clamp(((pel - 1) // 2) + 1, 0, 4), 'qual': clamp(pel - 1, 1, 3)} | kwargs

            pel_type = Nnedi3(**kwargs, opencl=nnedicl) if do_nnedi else Znedi3(**kwargs)

        assert isinstance(pel_type, Scaler)

        return pel_type.scale(clip, clip.width * pel, clip.height * pel, **kwargs)
