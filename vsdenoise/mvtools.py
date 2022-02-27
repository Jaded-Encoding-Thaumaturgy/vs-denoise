"""
This module implements wrappers for mvtool
"""

from __future__ import annotations

from enum import Enum, IntEnum
from math import exp, sqrt, pi, log, sin, e
from typing import Dict, Any, NamedTuple, Sequence, Type

from vardefunc.aa import Nnedi3SS
from lvsfunc.kernels import BicubicDogWay
from havsfunc import MinBlur, DitherLumaRebuild
from vsutil import (
    get_y, get_subsampling, get_depth, scale_value, get_peak_value, fallback, Range as CRange,
    disallow_variable_format, disallow_variable_resolution
)

from .knlm import knl_means_cl, ChannelMode
from .bm3d import AbstractBM3D, BM3D, BM3DCuda, BM3DCPU, BM3DCudaRTC, Profile

import vapoursynth as vs

core = vs.core


class VectorsMode(IntEnum):
    IGNORE = 0
    READ = 1
    WRITE = 2
    WRITEONLY = 3


class Pel(IntEnum):
    FULL = 1
    HALF = 2
    QUARTER = 4


class SourceType(IntEnum):
    BFF = 0
    TFF = 1
    PROGRESSIVE = 1


class PelType(IntEnum):
    AUTO = 0
    BICUBIC = 1
    WIENER = 2
    NNEDI3 = 3


class Prefilter(IntEnum):
    MINBLUR1 = 0
    MINBLUR2 = 1
    MINBLUR3 = 2
    MINBLURFLUX = 3
    DFTTEST = 4
    KNLMEANSCL = 5
    BM3D = 6
    DGDENOISE = 7
    AUTO = 8
    NONE = 9


class SMDegrain:
    """Simple MVTools Degrain with motion analysis"""
    ...
    wclip: vs.VideoNode
    thSAD: _SceneAnalyzeThreshold
    thSADC: _SceneAnalyzeThreshold
    thSCD: _SceneChangeThreshold
    refine: int
    vmode: VectorsMode

    analyze_args: Dict[str, Any]
    recalculate_args: Dict[str, Any]
    degrain_args: Dict[str, Any]

    _clip: vs.VideoNode
    _format: vs.VideoFormat
    _matrix: int

    UHDhalf: bool

    Amp: float = 1 / 32
    DCT: int = 0

    bm3d: Type[BM3D | BM3DCPU | BM3DCuda | BM3DCudaRTC] = BM3D

    __vectors: Dict[str, vs.VideoNode] = {}

    class _SceneAnalyzeThreshold(NamedTuple):
        recalculate: float
        degrain_hard: float
        degrain_soft: float

    class _SceneChangeThreshold(NamedTuple):
        recalculate: float
        degrain_hard: float
        degrain_soft: float

    def _analyze(self, /) -> None:
        ...

    def _recalculate(self, /) -> None:
        ...

    def _degrain(self, /) -> None:
        ...

    @staticmethod
    def ReplaceLowFrequency(
        flt: vs.VideoNode, ref: vs.VideoNode, LFR: float, DCTFlicker: bool = False, chroma: bool = True
    ) -> vs.VideoNode:
        LFR = max(LFR or (300 * flt.width / 1920), 50)

        freq_sample = max(flt.width, flt.height) * 2  # Frequency sample rate is resolution * 2 (for Nyquist)
        k = sqrt(log(2) / 2) * LFR                      # Constant for -3dB
        LFR = freq_sample / (k * 2 * pi)              # Frequency Cutoff for Gaussian Sigma
        sec0 = sin(e) + .1

        sec = scale_value(sec0, 8, flt.format.bits_per_sample, range=CRange.FULL)

        expr = "x y - z + "

        if DCTFlicker:
            expr += f"y z - d! y z = swap dup d@ 0 = 0 d@ 0 < -1 1 ? ? {sec} * + ?"

        final = core.akarin.Expr([flt, flt.bilateral.Gaussian(LFR), ref.bilateral.Gaussian(LFR)], expr)

        return final if chroma else core.std.ShufflePlanes([final, flt], [0, 1, 2], vs.YUV)

