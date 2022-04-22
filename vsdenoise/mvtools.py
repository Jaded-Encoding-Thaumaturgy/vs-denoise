"""
This module implements wrappers for mvtool
"""

from __future__ import annotations

from enum import IntEnum, auto
from math import exp, sqrt, pi, log, sin, e
from typing import Dict, Any, NamedTuple, Sequence, Type, Tuple, List

from havsfunc import MinBlur
from vsutil import (
    get_y, get_depth, scale_value, fallback, Range as CRange,  # get_peak_value,
    disallow_variable_format, disallow_variable_resolution
)

from .knlm import knl_means_cl, ChannelMode
from .bm3d import BM3D, AbstractBM3D, _AbstractBM3DCuda, Profile

import vapoursynth as vs

core = vs.core


# here until vsutil gets a new release
def get_peak_value(clip: vs.VideoNode, chroma: bool = False) -> float:
    assert clip.format
    return (0.5 if chroma else 1.) if clip.format.sample_type == vs.FLOAT else (1 << get_depth(clip)) - 1.


class Pel(IntEnum):
    FULL = 1
    HALF = 2
    QUARTER = 4


class SourceType(IntEnum):
    BFF = auto()
    TFF = auto()
    PROGRESSIVE = auto()


class PelType(IntEnum):
    AUTO = auto()
    BICUBIC = auto()
    WIENER = auto()
    NNEDI3 = auto()


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


class SMDegrainMode(IntEnum):
    Degrain = auto()
    Median = auto()
    FluxSmooth = auto()
    ML3DEx = auto()
    Hybrid = auto()
    TL3D = auto()
    STWM = auto()
    IQMT = auto()
    Gauss = auto()


class SMDegrain:
    """Simple MVTools Degrain with motion analysis"""
    analyze_args: Dict[str, Any]
    recalculate_args: Dict[str, Any]
    degrain_args: Dict[str, Any]

    bm3d_arch: Type[AbstractBM3D] = BM3D
    device_id: int = 0

    __vectors: Dict[str, vs.VideoNode] = {}

    clip: vs.VideoNode

    scaleCSAD: int
    isHD: bool
    isUHD: bool
    tr: int
    refine: int
    mode: SMDegrainMode
    source_type: SourceType
    prefilter: Prefilter | vs.VideoNode
    range_in: CRange
    pel: int
    subpixel: int
    chroma: bool
    is_gray: bool
    planes: List[int]
    mvplane: int
    refinemotion: bool
    truemotion: bool
    temporalSoften: bool
    rangeConversion: float
    lowFrequencyRestore: float
    DCTFlicker: float
    hpad: int
    hpadU: int
    vpad: int
    vpadU: int
    mfilter: vs.VideoNode | None
    rfilter: int

    class _SceneAnalyzeThreshold(NamedTuple):
        luma: float
        chroma: float

    class _SceneChangeThreshold(NamedTuple):
        first: int
        second: int

    @disallow_variable_format
    @disallow_variable_resolution
    def __init__(
        self, clip: vs.VideoNode,
        tr: int = 2, refine: int = 3, mode: SMDegrainMode = SMDegrainMode.Degrain,
        source_type: SourceType = SourceType.PROGRESSIVE,
        prefilter: Prefilter | vs.VideoNode = Prefilter.AUTO,
        range_in: CRange = CRange.LIMITED,
        pel: int | None = None, subpixel: int = 3,
        planes: int | Sequence[int] | None = None,
        refinemotion: bool = False, truemotion: bool | None = None,
        temporalSoften: bool = False, rangeConversion: float = 5.0,
        MFilter: vs.VideoNode | None = None, lowFrequencyRestore: float | bool = False,
        DCTFlicker: bool = False, fixFades: bool = False,
        hpad: int | None = None, vpad: int | None = None,
        rfilter: int = 3, UHDhalf: bool = True
    ) -> None:
        assert clip.format

        if clip.format.color_family not in {vs.GRAY, vs.YUV}:
            raise ValueError("SMDegrain: Only GRAY or YUV format clips supported")
        self.clip = clip

        if not isinstance(UHDhalf, bool):
            raise ValueError("SMDegrain: 'UHDhalf' has to be a boolean!")

        self.isHD = clip.width >= 1100 or clip.height >= 600
        if self.clip.width >= 2600 or self.clip.height >= 1500:
            self.isUHD, self.UHDhalf = True, UHDhalf
        else:
            self.isUHD = self.UHDhalf = False

        if not isinstance(tr, int):
            raise ValueError("SMDegrain: 'tr' has to be an int!")
        self.tr = tr

        if not isinstance(refine, int):
            raise ValueError("SMDegrain: 'refine' has to be an int!")
        self.refine = max(0, fallback(refine, 3))

        if mode is None or not isinstance(mode, int):
            raise ValueError("SMDegrain: 'mode' has to be from SMDegrainMode (enum)!")
        self.mode = mode

        if source_type is None or source_type not in SourceType:
            raise ValueError("SMDegrain: 'source_type' has to be from SourceType (enum)!")
        self.source_type = source_type

        if prefilter is None or prefilter not in Prefilter and not isinstance(prefilter, vs.VideoNode):
            raise ValueError("SMDegrain: 'prefilter' has to be from Prefilter (enum) or a VideoNode!")
        if isinstance(prefilter, vs.VideoNode):
            self._check_ref_clip(prefilter)
        self.prefilter = prefilter

        if range_in is None or range_in not in CRange:
            raise ValueError("SMDegrain: 'range_in' has to be 0 (limited) or 1 (full)!")
        self.range_in = range_in

        if not isinstance(pel, int) and pel is not None:
            raise ValueError("SMDegrain: 'pel' has to be an int or None!")
        self.pel = fallback(pel, 1 + int(self.isHD))

        if not isinstance(subpixel, int):
            raise ValueError("SMDegrain: 'subpixel' has to be an int!")
        self.subpixel = subpixel

        if planes is not None and isinstance(planes, int):
            planes = [planes]

        if clip.format.color_family == vs.GRAY:
            planes = [0]
            self.chroma = False
        elif planes is None:
            planes = [0, 1, 2]
            self.chroma = True

        self.is_gray = planes == [0]

        self.planes, self.mvplane = self._get_planes(planes)

        if not isinstance(refinemotion, bool):
            raise ValueError("SMDegrain: 'refinemotion' has to be a boolean!")
        self.refinemotion = refinemotion

        if not isinstance(temporalSoften, bool):
            raise ValueError("SMDegrain: 'temporalSoften' has to be a boolean!")
        self.temporalSoften = temporalSoften

        if not isinstance(truemotion, bool) and truemotion is not None:
            raise ValueError("SMDegrain: 'truemotion' has to be a boolean or None!")
        self.truemotion = fallback(truemotion, self.temporalSoften or not self.isHD)

        if not isinstance(fixFades, bool):
            raise ValueError("SMDegrain: 'fixFades' has to be a boolean!")

        if not isinstance(rangeConversion, float):
            raise ValueError("SMDegrain: 'rangeConversion' has to be a float!")
        self.rangeConversion = rangeConversion

        if not isinstance(lowFrequencyRestore, bool) and not isinstance(lowFrequencyRestore, float):
            raise ValueError("SMDegrain: 'lowFrequencyRestore' has to be a float or boolean!")

        if isinstance(lowFrequencyRestore, bool):
            if lowFrequencyRestore:
                self.lowFrequencyRestore = 3.46 * (clip.width / 1920.)
            else:
                self.lowFrequencyRestore = 0
        else:
            self.lowFrequencyRestore = (
                max(clip.width, clip.height) * 2
            ) / ((sqrt(log(2) / 2) * max(lowFrequencyRestore, 50)) * 2 * pi)

        if not isinstance(DCTFlicker, bool):
            raise ValueError("SMDegrain: 'DCTFlicker' has to be a boolean!")
        self.DCTFlicker = DCTFlicker

        if not isinstance(hpad, int) and pel is not None:
            raise ValueError("SMDegrain: 'hpad' has to be an int or None!")
        self.hpad = fallback(hpad, 0 if self.isHD else 8)
        self.hpadU = self.hpad // 2 if self.isUHD else self.hpad

        if not isinstance(vpad, int) and pel is not None:
            raise ValueError("SMDegrain: 'vpad' has to be an int or None!")
        self.vpad = fallback(vpad, 0 if self.isHD else 8)
        self.vpadU = self.vpad // 2 if self.isUHD else self.vpad

        if not isinstance(rfilter, int):
            raise ValueError("SMDegrain: 'rfilter' has to be an int!")
        self.rfilter = rfilter

        self.mfilter = self._check_ref_clip(MFilter)

        self.scaleCSAD = 2
        self.scaleCSAD -= 1 if clip.format.subsampling_w == 2 and clip.format.subsampling_h == 0 else 0
        self.scaleCSAD -= 1 if not self.isHD else 0

        self.DCT = 5 if fixFades else 0

    def analyze(
        self, ref: vs.VideoNode | None = None,
        overlap: int | None = None, blksize: int | None = None,
        search: int | None = None, pelsearch: int | None = None,
        searchparam: int | None = None
    ) -> None:
        self._check_ref_clip(ref)

        if not isinstance(blksize, int) and blksize is not None:
            raise ValueError("SMDegrain.analyse: 'blksize' has to be an int or None!")

        if not isinstance(overlap, int) and overlap is not None:
            raise ValueError("SMDegrain.analyse: 'overlap' has to be an int or None!")

        if not isinstance(search, int) and search is not None:
            raise ValueError("SMDegrain.analyse: 'search' has to be an int or None!")

        if not isinstance(pelsearch, int) and pelsearch is not None:
            raise ValueError("SMDegrain.analyse: 'pelsearch' has to be an int or None!")

        if not isinstance(searchparam, int) and searchparam is not None:
            raise ValueError("SMDegrain.analyse: 'searchparam' has to be an int or None!")

        searchparam = fallback(
            searchparam, (2 if self.isUHD else 5) if self.refinemotion and self.truemotion else (1 if self.isUHD else 2)
        )

        searchparamr = max(0, round(exp(0.69 * searchparam - 1.79) - 0.67))

        pelsearch = fallback(pelsearch, max(0, searchparam * 2 - 2))

        blocksize = fallback(blksize, max(2 ** (self.refine + 2), 16 if self.isHD else 8))

        halfblocksize = blocksize // 2
        halfoverlap = max(2, halfblocksize)

        overlap = fallback(overlap, halfblocksize)

        search = fallback(search, 4 if self.refinemotion else 2)

    def degrain(
        self,
        thSAD: int = 300, thSADC: int | None = None,
        thSCD1: int | None = None, thSCD2: int = 130,
        contrasharpening: bool | float | vs.VideoNode | None = None,
        limit: int | None = None, limitC: float | None = None, limitS: bool = True,
    ) -> None:
        if not isinstance(thSAD, int):
            raise ValueError("SMDegrain: 'thSAD' has to be an int!")

        if not isinstance(thSADC, int) and thSADC is not None:
            raise ValueError("SMDegrain: 'thSADC' has to be an int or None!")

        if not isinstance(thSCD1, int) and thSCD1 is not None:
            raise ValueError("SMDegrain: 'thSCD1' has to be an int or None!")

        if not isinstance(thSCD2, int):
            raise ValueError("SMDegrain: 'thSCD2' has to be an int!")

        if type(contrasharpening) not in {bool, float, type(None), vs.VideoNode}:
            raise ValueError("SMDegrain: 'contrasharpening' has to be a boolean or None!")
        elif isinstance(contrasharpening, vs.VideoNode) and contrasharpening.format != self.clip.format:
            raise ValueError("SMDegrain: 'All ref clips formats must be the same as the source clip'")

        if not isinstance(limit, int) and limit is not None:
            raise ValueError("SMDegrain: 'limit' has to be an int or None!")

        if not isinstance(limitC, float) and limitC is not None:
            raise ValueError("SMDegrain: 'limitC' has to be a float or None!")

        if not isinstance(limitS, bool):
            raise ValueError("SMDegrain: 'limitS' has to be a boolean!")

        thrSAD = self._SceneAnalyzeThreshold(
            round(exp(-101. / (thSAD * 0.83)) * 360),
            fallback(thSADC, round(thSAD * 0.18875 * exp(self.scaleCSAD * 0.693)))

        )

        thrSCD = self._SceneChangeThreshold(
            fallback(thSCD1, round(pow(16 * 2.5, 2))),
            fallback(thSCD2, 130)
        )

        print(thrSAD, thrSCD)

    def _get_subpel_clip(self, pel_type: PelType) -> vs.VideoNode:
        ...

    def _get_prefiltered_clip(self, clip: vs.VideoNode, pref_type: Prefilter) -> vs.VideoNode:
        if pref_type == Prefilter.AUTO:
            pref_type = Prefilter.MINBLUR3

        if self.UHDhalf:
            return clip.resize.Bicubic(clip.width // 2, clip.height // 2, filter_param_a=-0.6, filter_param_b=0.4)

        if pref_type == Prefilter.NONE:
            return clip

        if pref_type.value in {0, 1, 2}:
            return MinBlur(clip, pref_type.value, self.planes)

        if pref_type == Prefilter.MINBLURFLUX:
            return MinBlur(clip, 2).flux.SmoothST(2, 2, self.planes)

        if pref_type == Prefilter.DFTTEST:
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

        if pref_type == Prefilter.KNLMEANSCL:
            knl = knl_means_cl(
                clip, 7.0, 1, 2, 2, ChannelMode.ALL_PLANES if self.chroma else ChannelMode.LUMA,
                device_id=self.device_id
            )

            return self._ReplaceLowFrequency(knl, clip, 600 * (clip.width / 1920), chroma=self.chroma)

        if pref_type == Prefilter.BM3D:
            return self.bm3d_arch(
                clip, sigma=10 if isinstance(self.bm3d_arch, _AbstractBM3DCuda) else 8,
                radius=1, profile=Profile.LOW_COMPLEXITY
            ).clip

        if pref_type == Prefilter.DGDENOISE:
            # dgd = core.dgdecodenv.DGDenoise(pref, 0.10)

            # pref = self._ReplaceLowFrequency(dgd, pref, w / 2, chroma=self.chroma)
            return clip.bilateral.Gaussian(1)

        return clip

    @staticmethod
    def _ReplaceLowFrequency(
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

    def _get_planes(self, planes: Sequence[int]) -> Tuple[List[int], int]:
        if not (isinstance(planes, Sequence) and isinstance(planes[0], int)):
            raise ValueError("'planes' has to be a sequence of ints!")

        if planes == [0, 1, 2]:
            mvplane = 4
        elif len(planes) == 1 and planes[0] in {0, 1, 2}:
            mvplane = planes[0]
        elif planes == [1, 2]:
            mvplane = 3
        else:
            raise ValueError("Invalid planes specified!")

        return list(planes), mvplane

    def _check_ref_clip(self, ref: vs.VideoNode | None) -> vs.VideoNode | None:
        if not isinstance(ref, vs.VideoNode) and ref is not None:
            raise ValueError('Ref clip has to be a VideoNode or None!')

        if ref is None:
            return None

        assert self.clip.format
        assert ref.format

        if ref.format.id != self.clip.format.id:
            raise ValueError("Ref clip format must match the source clip's!")

        if ref.width != self.clip.width or ref.height != self.clip.height:
            raise ValueError("Ref clip sizes must match the source clip's!")

        return ref
