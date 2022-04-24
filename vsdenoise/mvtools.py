"""
This module implements wrappers for mvtool
"""

from __future__ import annotations

from enum import Enum, IntEnum, auto
from math import exp, sqrt, pi, log, sin, e
from typing import Dict, Any, NamedTuple, Sequence, Type, Tuple, List, Callable, cast

from havsfunc import MinBlur, DitherLumaRebuild
from vsutil import (
    Dither, get_y, get_depth, scale_value, fallback, Range as CRange,  # get_peak_value,
    disallow_variable_format, disallow_variable_resolution, depth
)

from .knlm import knl_means_cl, ChannelMode
from .bm3d import BM3D, AbstractBM3D, _AbstractBM3DCuda, Profile

import vapoursynth as vs

core = vs.core
blackman_args: Dict[str, Any] = dict(filter_param_a=-0.6, filter_param_b=0.4)


# here until vsutil gets a new release
def get_peak_value(clip: vs.VideoNode, chroma: bool = False) -> float:
    assert clip.format
    return (0.5 if chroma else 1.) if clip.format.sample_type == vs.FLOAT else (1 << get_depth(clip)) - 1.


class SourceType(IntEnum):
    BFF = 0
    TFF = 1
    PROGRESSIVE = 2

    @property
    def is_inter(self) -> bool:
        return self != SourceType.PROGRESSIVE


class PelType(IntEnum):
    AUTO = auto()
    NONE = auto()
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

    vectors: Dict[str, Any]

    clip: vs.VideoNode

    isHD: bool
    isUHD: bool
    tr: int
    refine: int
    mode: SMDegrainMode
    source_type: SourceType
    prefilter: Prefilter | vs.VideoNode
    pel_type: Tuple[PelType, PelType]
    range_in: CRange
    pel: int
    subpixel: int
    chroma: bool
    is_gray: bool
    planes: List[int]
    mvplane: int
    refinemotion: bool
    truemotion: bool
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
        tr: int = 2, refine: int = 3,
        mode: SMDegrainMode = SMDegrainMode.Degrain,
        source_type: SourceType = SourceType.PROGRESSIVE,
        prefilter: Prefilter | vs.VideoNode = Prefilter.AUTO,
        pel_type: Tuple[PelType, PelType] = (PelType.AUTO, PelType.AUTO),
        range_in: CRange = CRange.LIMITED,
        pel: int | None = None, subpixel: int = 3,
        planes: int | Sequence[int] | None = None,
        refinemotion: bool = False, truemotion: bool | None = None, rangeConversion: float = 5.0,
        MFilter: vs.VideoNode | None = None, lowFrequencyRestore: float | bool = False,
        DCTFlicker: bool = False, fixFades: bool = False,
        hpad: int | None = None, vpad: int | None = None,
        rfilter: int = 3, vectors: Dict[str, Any] = {}
    ) -> None:
        assert clip.format

        if clip.format.color_family not in {vs.GRAY, vs.YUV}:
            raise ValueError("SMDegrain: Only GRAY or YUV format clips supported")
        self.clip = clip

        self.isHD = clip.width >= 1100 or clip.height >= 600
        self.isUHD = self.clip.width >= 2600 or self.clip.height >= 1500

        if not isinstance(tr, int):
            raise ValueError("SMDegrain: 'tr' has to be an int!")
        self.tr = tr

        if not isinstance(refine, int):
            raise ValueError("SMDegrain: 'refine' has to be an int!")
        if refine > 6:
            raise ValueError("refine > 6 is not supported")
        self.refine = refine

        if mode is None or not isinstance(mode, int):
            raise ValueError("SMDegrain: 'mode' has to be from SMDegrainMode (enum)!")
        self.mode = mode

        if source_type is None or source_type not in SourceType:
            raise ValueError("SMDegrain: 'source_type' has to be from SourceType (enum)!")
        self.source_type = source_type

        if prefilter is None or prefilter not in Prefilter and not isinstance(prefilter, vs.VideoNode):
            raise ValueError("SMDegrain: 'prefilter' has to be from Prefilter (enum) or a VideoNode!")
        self.prefilter = prefilter

        if pel_type is None or not isinstance(pel_type, tuple) or any(p is None or p not in PelType for p in pel_type):
            raise ValueError("SMDegrain: 'source_type' has to be a tuple of PelType (enum)!")
        self.pel_type = pel_type

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

        if not isinstance(truemotion, bool) and truemotion is not None:
            raise ValueError("SMDegrain: 'truemotion' has to be a boolean or None!")
        self.truemotion = fallback(truemotion, not self.isHD)

        if not isinstance(fixFades, bool):
            raise ValueError("SMDegrain: 'fixFades' has to be a boolean!")

        if not isinstance(rangeConversion, float):
            raise ValueError("SMDegrain: 'rangeConversion' has to be a float!")
        self.rangeConversion = rangeConversion

        if not isinstance(lowFrequencyRestore, bool) and not isinstance(lowFrequencyRestore, float):
            raise ValueError("SMDegrain: 'lowFrequencyRestore' has to be a float or boolean!")

        self.vectors = vectors

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

        self.DCT = 5 if fixFades else 0


        if isinstance(prefilter, vs.VideoNode):
            self._check_ref_clip(prefilter)

        self.mfilter = self._check_ref_clip(MFilter)

    def analyze(
        self, ref: vs.VideoNode | None = None,
        overlap: int | None = None, blksize: int | None = None,
        search: int | None = None, pelsearch: int | None = None,
        searchparam: int | None = None
    ) -> None:
        ref = fallback(ref, self.workclip)

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
        limit = fallback(limit, 2 if self.isUHD else 255)

        if not isinstance(limitC, float) and limitC is not None:
            raise ValueError("SMDegrain: 'limitC' has to be a float or None!")
        limitC = fallback(limitC, limit)

        if not isinstance(limitS, bool):
            raise ValueError("SMDegrain: 'limitS' has to be a boolean!")

        thrSAD = self._SceneAnalyzeThreshold(
            round(exp(-101. / (thSAD * 0.83)) * 360),
            fallback(thSADC, round(thSAD * 0.18875 * exp(2 * 0.693)))

        )

        thrSCD = self._SceneChangeThreshold(
            fallback(thSCD1, round(0.35 * thSAD + 260)),
            fallback(thSCD2, 130)
        )

        print(thrSAD, thrSCD)

    def _get_subpel(self, clip: vs.VideoNode, pel_type: PelType) -> vs.VideoNode | None:
        bicubic_args: Dict[str, Any] = dict(width=clip.width * self.pel, height=(clip.height * self.pel))

        if pel_type == PelType.BICUBIC or pel_type == PelType.WIENER:
            if pel_type == PelType.WIENER:
                bicubic_args |= blackman_args
            return clip.resize.Bicubic(**bicubic_args)
        elif pel_type == PelType.NNEDI3:
            nnargs = dict(nsize=0, nns=1, qual=1, pscrn=2)

            nmsp = 'znedi3' if hasattr(core, 'znedi3') else 'nnedi3'

            nnedi3_cpu = getattr(
                getattr(clip.std.Transpose(), nmsp).nnedi3(0, True, **nnargs).std.Transpose(), nmsp
            ).nnedi3(0, True, **nnargs)

            if hasattr(core, 'nnedi3cl'):
                upscale = core.std.Interleave([
                    nnedi3_cpu[::2], clip[1::2].nnedi3cl.NNEDI3CL(0, True, True, **nnargs)  # type: ignore
                ])
            else:
                upscale = nnedi3_cpu

            return upscale.resize.Bicubic(src_top=.5, src_left=.5)

        return None

    def _get_subpel_clip(
        self, pref: vs.VideoNode, ref: vs.VideoNode
    ) -> Tuple[vs.VideoNode | None, vs.VideoNode | None]:
        pel_types = list(self.pel_type)

        for i, val in enumerate(pel_types):
            if val != PelType.AUTO:
                continue

            if i == 0:
                if self.prefilter == Prefilter.NONE:
                    pel_types[i] = PelType.NONE
                else:
                    if self.subpixel == 4:
                        pel_types[i] = PelType.NNEDI3
                    else:
                        pel_types[i] = PelType.BICUBIC
            else:
                pel_types[i] = PelType.NNEDI3 if self.subpixel == 4 else PelType.WIENER

        pel_type, pel2_type = pel_types

        return self._get_subpel(pref, pel_type), self._get_subpel(ref, pel2_type)

    def _get_pref(self, clip: vs.VideoNode, pref_type: Prefilter | vs.VideoNode) -> vs.VideoNode:
        if pref_type == Prefilter.NONE or isinstance(pref_type, vs.VideoNode):
            return clip
        elif pref_type.value in {0, 1, 2}:
            return MinBlur(clip, pref_type.value, self.planes)
        elif pref_type == Prefilter.MINBLURFLUX:
            return MinBlur(clip, 2).flux.SmoothST(2, 2, self.planes)
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
            knl = knl_means_cl(
                clip, 7.0, 1, 2, 2, ChannelMode.ALL_PLANES if self.chroma else ChannelMode.LUMA,
                device_id=self.device_id
            )

            return self._ReplaceLowFrequency(knl, clip, 600 * (clip.width / 1920), chroma=self.chroma)
        elif pref_type == Prefilter.BM3D:
            return self.bm3d_arch(
                clip, sigma=10 if isinstance(self.bm3d_arch, _AbstractBM3DCuda) else 8,
                radius=1, profile=Profile.LOW_COMPLEXITY
            ).clip
        elif pref_type == Prefilter.DGDENOISE:
            # dgd = core.dgdecodenv.DGDenoise(pref, 0.10)

            # pref = self._ReplaceLowFrequency(dgd, pref, w / 2, chroma=self.chroma)
            return clip.bilateral.Gaussian(1)

        return clip

    def _get_prefiltered_clip(self, pref: vs.VideoNode) -> vs.VideoNode:
        if isinstance(self.prefilter, vs.VideoNode):
            return self.prefilter

        pref = self._get_pref(pref, Prefilter.MINBLUR3 if self.prefilter == Prefilter.AUTO else self.prefilter)

        # Luma expansion TV->PC (up to 16% more values for motion estimation)
        if self.range_in == CRange.LIMITED:
            if self.rangeConversion > 1.0:
                pref = DitherLumaRebuild(pref, self.rangeConversion)
            elif self.rangeConversion > 0.0:
                pref = pref.retinex.MSRCP(None, self.rangeConversion, None, False, True)
            else:
                pref = depth(pref, 8, range=CRange.FULL, range_in=CRange.LIMITED, dither_type=Dither.NONE)

        # Low Frequency expansion (higher SAD -> more protection)
        # if self.lowFrequencyRestore > 0.0:
        #     pref = pref8.ex_unsharp(thSAD / 1800., Fc=w / 8., th=0.0)

        return pref

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

        assert self.workclip.format
        assert ref.format

        if ref.format.id != self.workclip.format.id:
            raise ValueError("Ref clip format must match the source clip's!")

        if ref.width != self.workclip.width or ref.height != self.workclip.height:
            raise ValueError("Ref clip sizes must match the source clip's!")

        return ref
