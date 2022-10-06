"""
This module implements prefilters for denoisers
"""

from __future__ import annotations

from enum import IntEnum
from math import ceil
from typing import Any, Type

from vsaa import Nnedi3, Znedi3
from vsexprtools import ExprOp, norm_expr
from vskernels import Bicubic, BicubicZopti, Bilinear
from vsrgtools import gauss_blur, min_blur, replace_low_frequencies
from vsrgtools.util import wmean_matrix
from vstools import (
    ColorRange, CustomRuntimeError, DitherType, PlanesT, core, depth, disallow_variable_format,
    disallow_variable_resolution, get_depth, get_neutral_value, get_peak_value, get_y, join, normalize_planes,
    scale_8bit, scale_value, split, vs
)

from .bm3d import BM3D as BM3DM
from .bm3d import BM3DCPU, AbstractBM3D, BM3DCuda, BM3DCudaRTC, Profile
from .knlm import ChannelMode, knl_means_cl

__all__ = ['Prefilter', 'prefilter_to_full_range', 'PelType']


class Prefilter(IntEnum):
    AUTO = -2
    NONE = -1
    MINBLUR1 = 0
    MINBLUR2 = 1
    MINBLUR3 = 2
    MINBLURFLUX = 3
    DFTTEST = 4
    KNLMEANSCL = 5
    BM3D = 6
    BM3D_CPU = 7
    BM3D_CUDA = 8
    BM3D_CUDA_RTC = 9
    DGDENOISE = 10
    HALFBLUR = 11
    GAUSSBLUR1 = 12
    GAUSSBLUR2 = 13

    @disallow_variable_format
    @disallow_variable_resolution
    def __call__(self, clip: vs.VideoNode, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
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

            pref_mask = get_y(clip).std.Expr(
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

            boxblur = half_clip.std.Convolution(wmean_matrix, planes=planes, **kwargs)

            return Bilinear.scale(boxblur, clip.width, clip.height)

        if pref_type in {Prefilter.GAUSSBLUR1, Prefilter.GAUSSBLUR2}:
            boxblur = clip.std.Convolution(wmean_matrix, planes=planes, **kwargs)

            gaussblur = gauss_blur(boxblur, 1.75, planes=planes, **kwargs)

            if pref_type == Prefilter.GAUSSBLUR2:
                i2, i7 = (scale_8bit(clip, x) for x in (2, 7))

                merge_expr = f'x {i7} + y < x {i2} + x {i7} - y > x {i2} - x 51 * y 49 * + 100 / ? ?'
            else:
                merge_expr = 'x 0.9 * y 0.1 * +'

            return norm_expr([gaussblur, clip], merge_expr, planes)

        return clip


def prefilter_to_full_range(pref: vs.VideoNode, range_conversion: float, planes: PlanesT = None) -> vs.VideoNode:
    planes = normalize_planes(pref, planes)
    work_clip, *chroma = split(pref) if planes == [0] else (pref, )
    assert (fmt := work_clip.format) and pref.format

    bits = get_depth(pref)
    is_gray = fmt.color_family == vs.GRAY
    is_integer = fmt.sample_type == vs.INTEGER

    # Luma expansion TV->PC (up to 16% more values for motion estimation)
    if range_conversion >= 1.0:
        neutral = get_neutral_value(work_clip, True)
        max_val = get_peak_value(work_clip)
        min_tv_val = scale_8bit(pref, 16)
        max_tv_val = scale_8bit(pref, 219)

        c = 0.0625

        k = (range_conversion - 1) * c
        t = f'x {min_tv_val} - {max_tv_val} / {ExprOp.clamp(0, 1)}' if is_integer else ExprOp.clamp(0, 1, 'x')

        pref_full = work_clip.std.Expr([
            f"{k} {1 + c} {(1 + c) * c} {t} {c} + / - * {t} 1 {k} - * + {f'{max_val} *' if is_integer else ''}",
            f'x {neutral} - 128 * 112 / {neutral} +'
        ][:1 + (not is_gray and is_integer)])
    elif range_conversion > 0.0:
        pref_full = work_clip.retinex.MSRCP(None, range_conversion, None, False, True)
    else:
        pref_full = depth(
            work_clip, bits, range_out=ColorRange.FULL, range_in=ColorRange.LIMITED, dither_type=DitherType.NONE
        )

    if chroma:
        return join([pref_full, *chroma], pref.format.color_family)

    return pref_full


class PelType(IntEnum):
    AUTO = -1
    NONE = 0
    BICUBIC = 1
    WIENER = 2
    NNEDI3 = 4

    @disallow_variable_format
    @disallow_variable_resolution
    def __call__(self, clip: vs.VideoNode, pel: int, **kwargs: Any) -> vs.VideoNode:
        assert clip.format

        pel_type = self

        if pel_type == PelType.AUTO:
            pel_type = PelType(1 << 3 - ceil(clip.height / 1000))

        if pel_type == PelType.NONE or pel <= 1:
            return clip

        factor = 2 ** pel
        width, height = clip.width * factor, clip.height * factor

        if pel_type == PelType.NNEDI3:
            nnedicl, nnedi, znedi = (hasattr(core, ns) for ns in ('nnedi3cl', 'nnedi3', 'znedi3'))
            do_nnedi = (nnedicl or nnedi) and not znedi

            if not any((nnedi, znedi, nnedicl)):
                raise CustomRuntimeError('Missing any nnedi3 implementation!', PelType.NNEDI3)

            upscaler = Nnedi3(0, 1, 1, **kwargs, opencl=nnedicl) if do_nnedi else Znedi3(0, 1, 1, **kwargs)

            return upscaler.scale(clip, width, height)

        kernel: type[Bicubic] = BicubicZopti if pel_type == PelType.WIENER else Bicubic  # type: ignore[assignment]

        return kernel.scale(clip, width, height, **kwargs)
