"""
This module implements wrappers for BM3D
"""

from __future__ import annotations

__all__ = [
    'Profile', 'AbstractBM3D',
    'BM3D', 'BM3DCuda', 'BM3DCudaRTC', 'BM3DCPU'
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple, final

from vskernels import Bicubic, Kernel, KernelT, Point
from vstools import (
    ColorRange, ColorRangeT, CustomStrEnum, CustomValueError, DitherType, FuncExceptT, KwargsT, Matrix, MatrixT,
    SingleOrArr, check_variable, core, get_video_format, get_y, iterate, join, normalize_seq, vs
)

from .types import _Plugin_bm3dcpu_Core_Bound, _Plugin_bm3dcuda_Core_Bound, _Plugin_bm3dcuda_rtc_Core_Bound


class ProfileBase:
    @dataclass
    class Config:
        """Profile config for arguments passed to the bm3d implementation basic/final calls."""

        profile: Profile
        """Which preset to use as a base."""

        kwargs: KwargsT
        """
        Base kwargs used for basic/final calls.\n
        These can be overridden by ``overrides`` or Profile defaults.
        """

        basic_kwargs: KwargsT
        """
        Kwargs used for basic calls.\n
        These can be overridden by ``overrides``/``overrides_basic`` or Profile defaults.
        """

        final_kwargs: KwargsT
        """
        Kwargs used for final calls.\n
        These can be overridden by ``overrides``/``overrides_final`` or Profile defaults.
        """

        overrides: KwargsT
        """Overrides for profile defaults and kwargs."""

        overrides_basic: KwargsT
        """Overrides for profile defaults and kwargs for basic calls."""

        overrides_final: KwargsT
        """Overrides for profile defaults and kwargs for final calls."""

        def as_dict(
            self, cuda: bool = False, basic: bool = False, aggregate: bool = False,
            args: KwargsT | None = None, **kwargs: Any
        ) -> KwargsT:
            """
            Get kwargs from this Config.

            :param cuda:        Whether the implementation is cuda or not.
            :param basic:       Whether the call is basic or final.
            :param aggregate:   Whether it's an aggregate (refining) call or not.
            :param args:        Custom args that will take priority over everything else.
            :param kwargs:      Additional kwargs to add.

            :return:            Dictionary of keyword arguments for the call.
            """

            kwargs |= self.kwargs

            if basic:
                kwargs |= self.basic_kwargs
            else:
                kwargs |= self.final_kwargs

            if self.profile is Profile.CUSTOM:
                values = KwargsT()
            elif not cuda:
                values = KwargsT(profile=self.profile.value)
            elif self.profile is Profile.VERY_NOISY:
                raise CustomValueError('Profile "VERY_NOISY" is not supported!', reason='BM3DCuda')
            else:
                if aggregate:
                    if basic:
                        PROFILES = {
                            Profile.FAST: KwargsT(block_step=8, bm_range=7, ps_num=2, ps_range=4),
                            Profile.LOW_COMPLEXITY: KwargsT(block_step=6, bm_range=9, ps_num=2, ps_range=4),
                            Profile.NORMAL: KwargsT(block_step=4, bm_range=12, ps_num=2, ps_range=5),
                            Profile.HIGH: KwargsT(block_step=3, bm_range=16, ps_num=2, ps_range=7),
                        }
                    else:
                        PROFILES = {
                            Profile.FAST: KwargsT(block_step=7, bm_range=7, ps_num=2, ps_range=5),
                            Profile.LOW_COMPLEXITY: KwargsT(block_step=5, bm_range=9, ps_num=2, ps_range=5),
                            Profile.NORMAL: KwargsT(block_step=3, bm_range=12, ps_num=2, ps_range=6),
                            Profile.HIGH: KwargsT(block_step=2, bm_range=16, ps_num=2, ps_range=8),
                        }
                else:
                    if basic:
                        PROFILES = {
                            Profile.FAST: KwargsT(block_step=8, bm_range=9),
                            Profile.LOW_COMPLEXITY: KwargsT(block_step=6, bm_range=9),
                            Profile.NORMAL: KwargsT(block_step=4, bm_range=16),
                            Profile.HIGH: KwargsT(block_step=3, bm_range=16),
                        }
                    else:
                        PROFILES = {
                            Profile.FAST: KwargsT(block_step=7, bm_range=9),
                            Profile.LOW_COMPLEXITY: KwargsT(block_step=5, bm_range=9),
                            Profile.NORMAL: KwargsT(block_step=3, bm_range=16),
                            Profile.HIGH: KwargsT(block_step=2, bm_range=16),
                        }

                values = PROFILES[self.profile]

            values |= kwargs | self.overrides

            if basic:
                values |= self.overrides_basic
            else:
                values |= self.overrides_final

            if args:
                values |= args

            if cuda:
                cuda_keys = set[str](core.bm3dcuda.BM3D.__signature__.parameters.keys())  # type: ignore

                values = {
                    key: value for key, value in values.items()
                    if key in cuda_keys
                }

            return values


@final
class Profile(ProfileBase, CustomStrEnum):
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

    CUSTOM = 'custom'
    """Member for your own profile with no defaults."""

    def __call__(
        self,
        block_step: int | None = None, bm_range: int | None = None,
        block_size: int | None = None, group_size: int | None = None,
        bm_step: int | None = None, th_mse: float | None = None, hard_thr: float | None = None,
        ps_num: int | None = None, ps_range: int | None = None, ps_step: int | None = None,
        basic_kwargs: KwargsT | None = None, final_kwargs: KwargsT | None = None, **kwargs: Any
    ) -> Profile.Config:
        return ProfileBase.Config(
            self, kwargs, basic_kwargs or {}, final_kwargs or {},
            {
                key: value for key, value in KwargsT(
                    block_step=block_step, bm_range=bm_range, block_size=block_size,
                    group_size=group_size, bm_step=bm_step, th_mse=th_mse
                ).items() if value is not None
            },
            {
                key: value for key, value in KwargsT(hard_thr=hard_thr).items() if value is not None
            },
            {
                key: value for key, value in KwargsT(
                    ps_num=ps_num, ps_range=ps_range, ps_step=ps_step
                ).items() if value is not None
            }
        )


class AbstractBM3D(ABC):
    """Abstract BM3D based denoiser interface."""

    wclip: vs.VideoNode

    sigma: _Sigma
    radius: _Radius
    profile: Profile.Config

    ref: vs.VideoNode | None

    refine: int

    yuv2rgb: Kernel
    rgb2yuv: Kernel

    is_gray: bool

    basic_args: KwargsT
    """Custom kwargs passed to bm3d for the :py:attr:`basic` clip."""

    final_args: KwargsT
    """Custom kwargs passed to bm3d for the :py:attr:`final` clip."""

    _clip: vs.VideoNode
    _format: vs.VideoFormat
    _matrix: Matrix | None

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
        profile: Profile | Profile.Config = Profile.FAST,
        ref: vs.VideoNode | None = None, refine: int = 1,
        matrix: MatrixT | None = None, range_in: ColorRangeT | None = None,
        yuv2rgb: KernelT = Bicubic, rgb2yuv: KernelT = Bicubic
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
        self._clip = self._check_clip(clip, matrix, range_in, self.__class__)

        self._matrix = Matrix.from_video(self._clip, True) if self._format.color_family is vs.YUV else None

        self.wclip = self._clip

        self.sigma = self._Sigma(*normalize_seq(sigma, 3))
        self.radius = self._Radius(*normalize_seq(radius or 0, 2))

        self.profile = profile if isinstance(profile, ProfileBase.Config) else profile()
        self.ref = ref and self._check_clip(ref, matrix, range_in, self.__class__)
        self.refine = refine

        self.yuv2rgb = Kernel.ensure_obj(yuv2rgb)
        self.rgb2yuv = Kernel.ensure_obj(rgb2yuv)

        self.is_gray = self._format.color_family == vs.GRAY

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

    def _check_clip(
        self, clip: vs.VideoNode, matrix: MatrixT | None, range_in: ColorRangeT | None, func: FuncExceptT
    ) -> vs.VideoNode:
        fmt = get_video_format(clip)

        if fmt.sample_type != vs.FLOAT or fmt.bits_per_sample != 32:
            clip = ColorRange.ensure_presence(clip, range_in, func)

        if fmt.color_family is vs.YUV:
            clip = Matrix.ensure_presence(clip, matrix, func)

        return clip


class BM3D(AbstractBM3D):
    """BM3D implementation by mawen1250."""

    pre: vs.VideoNode | None
    """Reference clip for :py:attr:`basic`."""

    fp32: bool = True
    """Whether to process in int16 or float32."""

    def __init__(
        self, clip: vs.VideoNode, /,
        sigma: SingleOrArr[float], radius: SingleOrArr[int] | None = None,
        profile: Profile | Profile.Config = Profile.FAST,
        pre: vs.VideoNode | None = None,
        ref: vs.VideoNode | None = None, refine: int = 1,
        matrix: MatrixT | None = None, range_in: ColorRangeT | None = None,
        yuv2rgb: KernelT = Bicubic, rgb2yuv: KernelT = Bicubic
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
        super().__init__(clip, sigma, radius, profile, ref, refine, matrix, range_in, yuv2rgb, rgb2yuv)
        self.pre = pre and self._check_clip(pre, matrix, range_in, self.__class__)

    def rgb2opp(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.bm3d.RGB2OPP(self.fp32)

    def opp2rgb(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.bm3d.OPP2RGB(self.fp32)

    def to_fullgray(self, clip: vs.VideoNode) -> vs.VideoNode:
        return get_y(clip).resize.Point(format=vs.GRAYS if self.fp32 else vs.GRAY16)

    def basic(self, clip: vs.VideoNode) -> vs.VideoNode:
        kwargs = KwargsT(ref=self.pre, sigma=self.sigma, matrix=100, args=self.basic_args)

        if self.radius.basic:
            clip = clip.bm3d.VBasic(**self.profile.as_dict(**kwargs, radius=self.radius.basic))
            clip = clip.bm3d.VAggregate(self.radius.basic, self.fp32)
        else:
            clip = clip.bm3d.Basic(**self.profile.as_dict(**kwargs))

        return clip

    def final(self, clip: vs.VideoNode, ref: vs.VideoNode | None = None) -> vs.VideoNode:
        if ref is None:
            ref = self.basic(self.wclip)

        kwargs = KwargsT(ref=ref, sigma=self.sigma, matrix=100, args=self.final_args)

        if self.radius.final:
            clip = clip.bm3d.VFinal(**self.profile.as_dict(**kwargs, radius=self.radius.final))
            clip = clip.bm3d.VAggregate(self.radius.final, self.fp32)
        else:
            clip = clip.bm3d.Final(**self.profile.as_dict(**kwargs))

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
    def plugin(self) -> _Plugin_bm3dcuda_Core_Bound | _Plugin_bm3dcuda_rtc_Core_Bound | _Plugin_bm3dcpu_Core_Bound:
        ...

    def __init__(
        self, clip: vs.VideoNode, /,
        sigma: SingleOrArr[float], radius: SingleOrArr[int] | None = None,
        profile: Profile | Profile.Config = Profile.FAST,
        ref: vs.VideoNode | None = None, refine: int = 1,
        matrix: MatrixT | None = None, range_in: ColorRangeT | None = None,
        yuv2rgb: KernelT = Bicubic, rgb2yuv: KernelT = Bicubic
    ) -> None:
        super().__init__(clip, sigma, radius, profile, ref, refine, matrix, range_in, yuv2rgb, rgb2yuv)

    def basic(self, clip: vs.VideoNode) -> vs.VideoNode:
        clip = self.plugin.BM3D(clip, **self.profile.as_dict(
            True, True, False, self.basic_args, sigma=self.sigma, radius=self.radius.basic
        ))

        if self.radius.basic:
            clip = clip.bm3d.VAggregate(self.radius.basic, 1)

        return clip

    def final(self, clip: vs.VideoNode, ref: vs.VideoNode | None = None) -> vs.VideoNode:
        clip = self.plugin.BM3D(clip, ref, **self.profile.as_dict(
            True, False, True, self.final_args, sigma=self.sigma, radius=self.radius.final
        ))

        if self.radius.final:
            clip = clip.bm3d.VAggregate(self.radius.final, 1)

        return clip


class BM3DCuda(_AbstractBM3DCuda):
    @property
    def plugin(self) -> _Plugin_bm3dcuda_Core_Bound:
        return core.bm3dcuda


class BM3DCudaRTC(_AbstractBM3DCuda):
    @property
    def plugin(self) -> _Plugin_bm3dcuda_rtc_Core_Bound:
        return core.bm3dcuda_rtc


class BM3DCPU(_AbstractBM3DCuda):
    @property
    def plugin(self) -> _Plugin_bm3dcpu_Core_Bound:
        return core.bm3dcpu
