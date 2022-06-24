"""
This module implements prefilters for denoisers
"""

from __future__ import annotations

from enum import IntEnum
from math import e, log, pi, sin, sqrt

import vapoursynth as vs
from vsrgtools import minblur
from vsutil import Dither
from vsutil import Range as CRange
from vsutil import depth, get_depth, get_y, scale_value
from typing import Type

from .bm3d import BM3D as BM3DM
from .bm3d import BM3DCPU, BM3DCuda, BM3DCudaRTC, Profile, AbstractBM3D
from .knlm import knl_means_cl
from .utils import get_neutral_value, get_peak_value

core = vs.core


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




def prefilter_clip(clip: vs.VideoNode, pref_type: Prefilter) -> vs.VideoNode:
    pref_type = Prefilter.MINBLUR3 if pref_type == Prefilter.AUTO else pref_type

    if pref_type == Prefilter.NONE:
        return clip
    elif pref_type.value in {0, 1, 2}:
        return minblur(clip, pref_type.value)
    elif pref_type == Prefilter.MINBLURFLUX:
        return minblur(clip, 2).flux.SmoothST(2, 2)
    elif pref_type == Prefilter.DFTTEST:
        bits = get_depth(clip)
        peak = get_peak_value(clip)

        y = get_y(clip)
        i = scale_value(16, 8, bits, range=CRange.FULL)
        j = scale_value(75, 8, bits, range=CRange.FULL)

        dfft = clip.dfttest.DFTTest(
            tbsize=1, slocation=[0.0, 4.0, 0.2, 9.0, 1.0, 15.0],
            sbsize=12, sosize=6, swin=2
        )

        prefmask = y.std.Expr(f'x {i} < {peak} x {j} > 0 {peak} x {i} - {peak} {j} {i} - / * - ? ?')

        return dfft.std.MaskedMerge(clip, prefmask)
    elif pref_type == Prefilter.KNLMEANSCL:
        knl = knl_means_cl(clip, 7.0, 1, 2, 2)

        return replace_low_frequencies(knl, clip, 600 * (clip.width / 1920))
    elif pref_type in {Prefilter.BM3D, Prefilter.BM3D_CPU, Prefilter.BM3D_CUDA, Prefilter.BM3D_CUDA_RTC}:
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

        return bm3d_arch(clip, sigma=sigma, radius=1, profile=profile).clip
    elif pref_type == Prefilter.DGDENOISE:
        # dgd = core.dgdecodenv.DGDenoise(pref, 0.10)

        # pref = replace_low_frequencies(dgd, pref, w / 2)
        return clip.bilateral.Gaussian(1)

    return clip


def prefilter_to_full_range(
    pref: vs.VideoNode, prefilter: Prefilter, range_conversion: float
) -> vs.VideoNode:
    pref = prefilter_clip(pref, prefilter)
    fmt = pref.format
    assert fmt

    # Luma expansion TV->PC (up to 16% more values for motion estimation)
    if range_conversion > 1.0:
        is_gray = fmt.color_family == vs.GRAY
        is_integer = fmt.sample_type == vs.INTEGER

        bits = get_depth(pref)
        neutral = get_neutral_value(pref)
        max_val = get_peak_value(pref)
        min_tv_val = scale_value(16, 8, bits)
        max_tv_val = scale_value(219, 8, bits)

        c = 0.0625

        k = (range_conversion - 1) * c
        t = f'x {min_tv_val} - {max_tv_val} / 0 max 1 min' if is_integer else 'x 0 max 1 min'

        pref = pref.std.Expr([
            f"{k} {1 + c} {(1 + c) * c} {t} {c} + / - * {t} 1 {k} - * + {f'{max_val} *' if is_integer else ''}",
            f'x {neutral} - 128 * 112 / {neutral} +'
        ][:1 + (not is_gray and is_integer)])
    elif range_conversion > 0.0:
        pref = pref.retinex.MSRCP(None, range_conversion, None, False, True)
    else:
        pref = depth(
            pref, fmt.bits_per_sample,
            range=CRange.FULL, range_in=CRange.LIMITED,
            dither_type=Dither.NONE
        )

    return pref


def replace_low_frequencies(
    flt: vs.VideoNode, ref: vs.VideoNode, LFR: float, DCTFlicker: bool = False, chroma: bool = True
) -> vs.VideoNode:
    assert flt.format
    LFR = max(LFR or (300 * flt.width / 1920), 50)

    freq_sample = max(flt.width, flt.height) * 2    # Frequency sample rate is resolution * 2 (for Nyquist)
    k = sqrt(log(2) / 2) * LFR                      # Constant for -3dB
    LFR = freq_sample / (k * 2 * pi)                # Frequency Cutoff for Gaussian Sigma
    sec0 = sin(e) + .1

    sec = scale_value(sec0, 8, flt.format.bits_per_sample, range=CRange.FULL)

    expr = "x y - z + "

    if DCTFlicker:
        expr += f"y z - d! y z = swap dup d@ 0 = 0 d@ 0 < -1 1 ? ? {sec} * + ?"

    final = core.akarin.Expr([flt, flt.bilateral.Gaussian(LFR), ref.bilateral.Gaussian(LFR)], expr)

    return final if chroma else core.std.ShufflePlanes([final, flt], [0, 1, 2], vs.YUV)
