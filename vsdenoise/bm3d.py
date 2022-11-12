"""
This module implements wrappers for BM3D
"""

from __future__ import annotations

__all__ = [
    'Profile', 'AbstractBM3D',
    'BM3D', 'BM3DCuda', 'BM3DCudaRTC', 'BM3DCPU'
]

from abc import ABC, abstractmethod
from typing import Any, ClassVar, NamedTuple, final

from vskernels import Bicubic, Kernel, KernelT, Point
from vstools import (
    ColorRange, CustomStrEnum, CustomValueError, DitherType, Matrix, SingleOrArr, check_variable, core, get_y, iterate,
    join, normalize_seq, vs
)

from .types import _PluginBm3dcpuCoreUnbound, _PluginBm3dcuda_rtcCoreUnbound, _PluginBm3dcudaCoreUnbound


@final
class Profile(CustomStrEnum):
    """
    BM3D profiles that set default parameters for each of them.\n
    See the original documentation for more information:\n
    https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D#profile-default
    """

    FAST = 'fast'
    """Profile aimed at maximizing speed."""

    LOW_COMPLEXITY = 'lc'
    """Profile for content with low-complexity noise."""

    NORMAL = 'np'
    """Neutral profile."""

    HIGH = 'high'
    """Profile aimed at high-precision denoising."""

    VERY_NOISY = 'vn'
    """Profile for very noisy content."""


class AbstractBM3D(ABC):
    """Abstract BM3D based denoiser interface."""

    wclip: vs.VideoNode

    sigma: _Sigma
    radius: _Radius
    profile: Profile

    ref: vs.VideoNode | None

    refine: int

    yuv2rgb: Kernel
    rgb2yuv: Kernel

    is_gray: bool

    basic_args: dict[str, Any]
    """Custom kwargs passed to bm3d for the :py:attr:`basic` clip."""

    final_args: dict[str, Any]
    """Custom kwargs passed to bm3d for the :py:attr:`final` clip."""

    _clip: vs.VideoNode
    _format: vs.VideoFormat
    _matrix: Matrix

    class _Sigma(NamedTuple):
        y: float
        u: float
        v: float

    class _Radius(NamedTuple):
        basic: int
        final: int

    def __init__(
        self, clip: vs.VideoNode, /,
        sigma: SingleOrArr[float], radius: SingleOrArr[int] | None = None,
        profile: Profile = Profile.FAST,
        ref: vs.VideoNode | None = None,
        refine: int = 1,
        yuv2rgb: KernelT = Bicubic,
        rgb2yuv: KernelT = Bicubic
    ) -> None:
        """
        :param clip:        Source clip.
        :param sigma:       Strength of denoising, valid range is [0, +inf].
        :param radius:      Temporal radius, valid range is [1, 16].
        :param profile:     Preset profile. See :py:attr:`vsdenoise.bm3d.Profile`.
        :param ref:         Reference clip used in block-matching, replacing the basic estimation.
                            If not specified, the input clip is used instead.
        :param refine:      Times to refine the estimation.
                             * 0 means basic estimate only.
                             * 1 means basic estimate with one final estimate.
                             * n means basic estimate refined with final estimate applied n times.
        :param yuv2rgb:     Kernel used for converting the clip from YUV to RGB.
        :param rgb2yuv:     Kernel used for converting back the clip from RGB to YUV.
        """
        assert check_variable(clip, self.__class__)

        self._format = clip.format
        self._clip = clip
        self._check_clips(clip, ref)
        self._matrix = Matrix.from_video(clip, True)

        self.wclip = clip

        self.sigma = self._Sigma(*normalize_seq(sigma, 3))
        self.radius = self._Radius(*normalize_seq(radius or 0, 2))

        self.profile = profile
        self.ref = ref
        self.refine = refine

        self.yuv2rgb = Kernel.ensure_obj(yuv2rgb)
        self.rgb2yuv = Kernel.ensure_obj(rgb2yuv)

        self.is_gray = clip.format.color_family == vs.GRAY

        self.basic_args = {}
        self.final_args = {}

        if self.is_gray:
            self.sigma = self.sigma._replace(u=0, v=0)
        elif sum(self.sigma[1:]) == 0:
            self.is_gray = True

    def yuv2opp(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Convert a YUV clip to the OPP colorspace.

        :param clip:    YUV clip to be processed.

        :return:        OPP clip.
        """
        return self.rgb2opp(self.yuv2rgb.resample(clip, vs.RGBS))

    def rgb2opp(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Convert an RGB clip to the OPP colorspace.

        :param clip:    RGB clip to be processed.

        :return:        OPP clip.
        """
        return clip.bm3d.RGB2OPP(sample=1)

    def opp2rgb(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Convert an OPP clip to the RGB colorspace.

        :param clip:    OPP clip to be processed.

        :return:        RGB clip.
        """
        return clip.bm3d.OPP2RGB(sample=1)

    def to_fullgray(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Extract Y plane from GRAY/YUV clip and if not float32, upsample to it.

        :param clip:    GRAY or YUV clip to be processed.

        :return:        GRAYS clip.
        """
        return get_y(clip).resize.Point(format=vs.GRAYS)

    @abstractmethod
    def basic(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Retrieve the "basic" clip, typically used as a `ref` clip for :py:attr:`final`.

        :param clip:    OPP or GRAY colorspace clip to be processed.

        :return:        Denoised clip to be used as `ref`.
        """

    @abstractmethod
    def final(self, clip: vs.VideoNode, ref: vs.VideoNode | None = None) -> vs.VideoNode:
        """
        Retrieve the "final" clip.

        :param clip:    OPP or GRAY colorspace clip to be processed.
        :param ref:     Reference clip used for weight calculations.

        :return:        Final, refined, denoised clip.
        """

    @property
    def clip(self) -> vs.VideoNode:
        """
        Final denoised clip.

        ``denoised_clip = BM3D(...).clip`` is the intended use in encoding scripts.

        :return:        Output clip.
        """
        self._preprocessing()

        # Make basic estimation
        if self.ref is None:
            refv = self.basic(self.wclip)
        else:
            if self.is_gray:
                refv = self.to_fullgray(self.ref)
            else:
                refv = self.yuv2opp(self.ref)

        # Make final estimation
        self.wclip = iterate(self.wclip, self.final, self.refine, ref=refv)

        self._post_processing()

        return self.wclip

    def _preprocessing(self) -> None:
        # Initialise the input clip
        if self.is_gray:
            self.wclip = self.to_fullgray(self.wclip)
        else:
            self.wclip = self.yuv2opp(self.wclip)

    def _post_processing(self) -> None:
        # Resize
        dither = DitherType.ERROR_DIFFUSION if DitherType.should_dither(self._format, self.wclip) else DitherType.NONE

        if self.is_gray:
            self.wclip = Point.resample(
                self.wclip, self._format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id,
                dither_type=dither
            )
            if self._format.color_family == vs.YUV:
                self.wclip = join(self.wclip, self._clip)
        else:
            dither = DitherType.from_param(self.rgb2yuv.kwargs.get('dither_type', None)) or dither
            self.wclip = self.rgb2yuv.resample(
                self.opp2rgb(self.wclip), format=self._format, matrix=self._matrix, dither_type=dither
            )

        if self.sigma.y == 0:
            self.wclip = join(self._clip, self.wclip)

    def _check_clips(self, *clips: vs.VideoNode | None) -> None:
        for clip in clips:
            if clip:
                ColorRange.from_video(clip, True)
                Matrix.from_video(clip, True)


class BM3D(AbstractBM3D):
    """BM3D implementation by mawen1250."""

    pre: vs.VideoNode | None
    """Reference clip for :py:attr:`basic`."""

    fp32: bool = True
    """Whether to process in int16 or float32."""

    def __init__(
        self, clip: vs.VideoNode, /,
        sigma: SingleOrArr[float], radius: SingleOrArr[int] | None = None,
        profile: Profile = Profile.FAST,
        pre: vs.VideoNode | None = None, ref: vs.VideoNode | None = None,
        refine: int = 1,
        yuv2rgb: KernelT = Bicubic,
        rgb2yuv: KernelT = Bicubic
    ) -> None:
        """
        :param clip:                Source clip.
        :param sigma:               Strength of denoising, valid range is [0, +inf].
        :param radius:              Temporal radius, valid range is [1, 16].
        :param profile:             Preset profiles.
        :param pre:                 Pre-filtered clip for basic estimate.
                                    Should be a clip better suited for block-matching than the input clip.
        :param ref:                 Reference clip used in block-matching, replacing the basic estimation.
                                    If not specified, the input clip is used instead.
        :param refine:              Times to refine the estimation.
                                     * 0 means basic estimate only.
                                     * 1 means basic estimate with one final estimate.
                                     * n means basic estimate refined with final estimate for n times.
        :param yuv2rgb:             Kernel used for converting the clip from YUV to RGB.
        :param rgb2yuv:             Kernel used for converting the clip back from RGB to YUV.
        """
        super().__init__(clip, sigma, radius, profile, ref, refine, yuv2rgb, rgb2yuv)
        self._check_clips(pre)
        self.pre = pre

    def rgb2opp(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.bm3d.RGB2OPP(self.fp32)

    def opp2rgb(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.bm3d.OPP2RGB(self.fp32)

    def to_fullgray(self, clip: vs.VideoNode) -> vs.VideoNode:
        return get_y(clip).resize.Point(format=vs.GRAYS if self.fp32 else vs.GRAY16)

    def basic(self, clip: vs.VideoNode) -> vs.VideoNode:
        kwargs = dict[str, Any](ref=self.pre, profile=self.profile, sigma=self.sigma, matrix=100) | self.basic_args

        if self.radius.basic:
            clip = core.bm3d.VBasic(
                clip, radius=self.radius.basic, **kwargs
            ).bm3d.VAggregate(self.radius.basic, self.fp32)
        else:
            clip = core.bm3d.Basic(clip, **kwargs)

        return clip

    def final(self, clip: vs.VideoNode, ref: vs.VideoNode | None = None) -> vs.VideoNode:
        kwargs = dict[str, Any](profile=self.profile, sigma=self.sigma, matrix=100) | self.final_args

        if ref is None:
            ref = self.basic(self.wclip)

        if self.radius.final:
            clip = core.bm3d.VFinal(
                clip, ref=ref, radius=self.radius.final, **kwargs
            ).bm3d.VAggregate(self.radius.final, self.fp32)
        else:
            clip = core.bm3d.Final(clip, ref=ref, **kwargs)

        return clip

    def _preprocessing(self) -> None:
        # Initialise pre
        if self.pre is not None:
            if self.is_gray:
                self.pre = self.to_fullgray(self.pre)
            else:
                self.pre = self.yuv2opp(self.pre)

        super()._preprocessing()


class _AbstractBM3DCuda(AbstractBM3D, ABC):
    """BM3D implementation by WolframRhodium."""

    @property
    @abstractmethod
    def plugin(self) -> _PluginBm3dcudaCoreUnbound | _PluginBm3dcuda_rtcCoreUnbound | _PluginBm3dcpuCoreUnbound:
        ...

    def __init__(
        self, clip: vs.VideoNode, /,
        sigma: SingleOrArr[float], radius: SingleOrArr[int] | None = None,
        profile: Profile = Profile.FAST,
        ref: vs.VideoNode | None = None,
        refine: int = 1,
        yuv2rgb: KernelT = Bicubic,
        rgb2yuv: KernelT = Bicubic
    ) -> None:
        super().__init__(clip, sigma, radius, profile, ref, refine, yuv2rgb, rgb2yuv)
        if self.profile == Profile.VERY_NOISY:
            raise CustomValueError('Profile "VERY_NOISY" is not supported!', self.__class__)

    CUDA_BASIC_PROFILES: ClassVar[dict[str, dict[str, Any]]] = {
        Profile.FAST: dict(block_step=8, bm_range=9),
        Profile.LOW_COMPLEXITY: dict(block_step=6, bm_range=9),
        Profile.NORMAL: dict(block_step=4, bm_range=16),
        Profile.HIGH: dict(block_step=3, bm_range=16),
    }
    CUDA_FINAL_PROFILES: ClassVar[dict[str, dict[str, Any]]] = {
        Profile.FAST: dict(block_step=7, bm_range=9),
        Profile.LOW_COMPLEXITY: dict(block_step=5, bm_range=9),
        Profile.NORMAL: dict(block_step=3, bm_range=16),
        Profile.HIGH: dict(block_step=2, bm_range=16),
    }
    CUDA_VBASIC_PROFILES: ClassVar[dict[str, dict[str, Any]]] = {
        Profile.FAST: dict(block_step=8, bm_range=7, ps_num=2, ps_range=4),
        Profile.LOW_COMPLEXITY: dict(block_step=6, bm_range=9, ps_num=2, ps_range=4),
        Profile.NORMAL: dict(block_step=4, bm_range=12, ps_num=2, ps_range=5),
        Profile.HIGH: dict(block_step=3, bm_range=16, ps_num=2, ps_range=7),
    }
    CUDA_VFINAL_PROFILES: ClassVar[dict[str, dict[str, Any]]] = {
        Profile.FAST: dict(block_step=7, bm_range=7, ps_num=2, ps_range=5),
        Profile.LOW_COMPLEXITY: dict(block_step=5, bm_range=9, ps_num=2, ps_range=5),
        Profile.NORMAL: dict(block_step=3, bm_range=12, ps_num=2, ps_range=6),
        Profile.HIGH: dict(block_step=2, bm_range=16, ps_num=2, ps_range=8),
    }

    def basic(self, clip: vs.VideoNode) -> vs.VideoNode:
        if self.radius.basic:
            clip = self.plugin.BM3D(
                clip, sigma=self.sigma, radius=self.radius.basic,
                **self.CUDA_VBASIC_PROFILES[self.profile] | self.basic_args
            ).bm3d.VAggregate(self.radius.basic, 1)
        else:
            clip = self.plugin.BM3D(
                clip, sigma=self.sigma, radius=0,
                **self.CUDA_BASIC_PROFILES[self.profile] | self.basic_args
            )
        return clip

    def final(self, clip: vs.VideoNode, ref: vs.VideoNode | None = None) -> vs.VideoNode:
        if self.radius.final:
            clip = self.plugin.BM3D(
                clip, ref, self.sigma, radius=self.radius.final,
                **self.CUDA_VFINAL_PROFILES[self.profile] | self.final_args
            ).bm3d.VAggregate(self.radius.final, 1)
        else:
            clip = self.plugin.BM3D(
                clip, ref, self.sigma, radius=0,
                **self.CUDA_FINAL_PROFILES[self.profile] | self.final_args
            )
        return clip


class BM3DCuda(_AbstractBM3DCuda):
    @property
    def plugin(self) -> _PluginBm3dcudaCoreUnbound:
        return core.bm3dcuda


class BM3DCudaRTC(_AbstractBM3DCuda):
    @property
    def plugin(self) -> _PluginBm3dcuda_rtcCoreUnbound:
        return core.bm3dcuda_rtc


class BM3DCPU(_AbstractBM3DCuda):
    @property
    def plugin(self) -> _PluginBm3dcpuCoreUnbound:
        return core.bm3dcpu
