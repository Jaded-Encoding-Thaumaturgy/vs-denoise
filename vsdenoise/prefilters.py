"""
This module implements prefilters for denoisers
"""

from __future__ import annotations

from math import ceil, sin
from typing import TYPE_CHECKING, Any, Type

from vsaa import Nnedi3, Znedi3
from vsexprtools import ExprOp, norm_expr, aka_expr_available
from vskernels import Bicubic, BicubicZopti, Bilinear, Scaler, ScalerT
from vsrgtools import gauss_blur, min_blur, replace_low_frequencies, blur
from vstools import (
    ColorRange, CustomRuntimeError, DitherType, PlanesT, core, depth, disallow_variable_format,
    disallow_variable_resolution, get_depth, get_neutral_value, get_peak_value, get_y, join, normalize_planes,
    scale_8bit, scale_value, split, vs, CustomEnum, CustomIntEnum, clamp
)

from .bm3d import BM3D as BM3DM
from .bm3d import BM3DCPU, AbstractBM3D, BM3DCuda, BM3DCudaRTC, Profile
from .knlm import ChannelMode, knl_means_cl

__all__ = [
    'Prefilter', 'prefilter_to_full_range',
    'PelType'
]


class Prefilter(CustomIntEnum):
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

    BM3D_CPU = 7
    """@@PLACEHOLDER@@"""

    BM3D_CUDA = 8
    """@@PLACEHOLDER@@"""

    BM3D_CUDA_RTC = 9
    """@@PLACEHOLDER@@"""

    DGDENOISE = 10
    """@@PLACEHOLDER@@"""

    HALFBLUR = 11
    """@@PLACEHOLDER@@"""

    GAUSSBLUR1 = 12
    """@@PLACEHOLDER@@"""

    GAUSSBLUR2 = 13
    """@@PLACEHOLDER@@"""

    @disallow_variable_format
    @disallow_variable_resolution
    def __call__(self, clip: vs.VideoNode, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
        """@@PLACEHOLDER@@"""

        pref_type = Prefilter.MINBLUR3 if self == Prefilter.AUTO else self

        bits = get_depth(clip)
        peak = get_peak_value(clip)
        planes = normalize_planes(clip, planes)

        if pref_type == Prefilter.NONE:
            return clip

        if pref_type.value in {0, 1, 2}:
            return min_blur(clip, pref_type.value, planes)

        if pref_type == Prefilter.MINBLURFLUX:
            return min_blur(clip, 2, planes).flux.SmoothST(2, 2, planes)

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
            knl = knl_means_cl(clip, 7.0, 1, 2, 2, ChannelMode.from_planes(planes), **kwargs)

            return replace_low_frequencies(knl, clip, 600 * (clip.width / 1920), False, planes)

        if pref_type in {Prefilter.BM3D, Prefilter.BM3D_CPU, Prefilter.BM3D_CUDA, Prefilter.BM3D_CUDA_RTC}:
            bm3d_arch: Type[AbstractBM3D]

            if pref_type == Prefilter.BM3D:
                bm3d_arch, sigma, profile = BM3DM, 10, Profile.FAST
            elif pref_type == Prefilter.BM3D_CPU:
                bm3d_arch, sigma, profile = BM3DCPU, 10, Profile.LOW_COMPLEXITY
            elif pref_type == Prefilter.BM3D_CUDA:
                bm3d_arch, sigma, profile = BM3DCuda, 8, Profile.NORMAL
            elif pref_type == Prefilter.BM3D_CUDA_RTC:
                bm3d_arch, sigma, profile = BM3DCudaRTC, 8, Profile.NORMAL
            else:
                raise ValueError

            sigmas = [sigma if 0 in planes else 0, sigma if (1 in planes or 2 in planes) else 0]

            bm3d_args = dict[str, Any](sigma=sigmas, radius=1, profile=profile) | kwargs

            return bm3d_arch(clip, **bm3d_args).clip

        if pref_type == Prefilter.DGDENOISE:
            # dgd = core.dgdecodenv.DGDenoise(pref, 0.10)

            # pref = replace_low_frequencies(dgd, pref, w / 2)
            return gauss_blur(clip, 1, planes=planes, **kwargs)

        if pref_type == Prefilter.HALFBLUR:
            half_clip = Bilinear.scale(clip, clip.width // 2, clip.height // 2)

            boxblur = blur(half_clip, planes=planes)

            return Bilinear.scale(boxblur, clip.width, clip.height)

        if pref_type in {Prefilter.GAUSSBLUR1, Prefilter.GAUSSBLUR2}:
            boxblur = blur(clip, planes=planes)

            gaussblur = gauss_blur(boxblur, 1.75, planes=planes, **kwargs)

            if pref_type == Prefilter.GAUSSBLUR2:
                i2, i7 = (scale_8bit(clip, x) for x in (2, 7))

                merge_expr = f'x {i7} + y < x {i2} + x {i7} - y > x {i2} - x 51 * y 49 * + 100 / ? ?'
            else:
                merge_expr = 'x 0.9 * y 0.1 * +'

            return norm_expr([gaussblur, clip], merge_expr, planes)

        return clip


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
        def __init__(self, scaler: str | Type[Scaler] | Scaler, **kwargs: Any) -> None:
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
