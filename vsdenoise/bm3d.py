"""
This module implements wrappers for BM3D
"""

from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple, final

from vskernels import Catrom, Kernel, KernelT, Point
from vstools import (
    ColorRange, ColorRangeT, CustomStrEnum, CustomValueError, DitherType, FuncExceptT, KwargsT, Matrix, MatrixT, Self,
    SingleOrArr, check_variable, check_variable_format, core, get_video_format, join, normalize_seq, vs, vs_object
)

from .types import _Plugin_bm3dcpu_Core_Bound, _Plugin_bm3dcuda_Core_Bound, _Plugin_bm3dcuda_rtc_Core_Bound

__all__ = [
    'Profile',
    'BM3D', 'BM3DCuda', 'BM3DCudaRTC', 'BM3DCPU'
]


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

                values = PROFILES[self.profile]  # type: ignore[assignment]

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


class AbstractBM3D(vs_object):
    """Abstract BM3D-based denoiser interface."""

    @dataclass
    class ResampleConfig:
        yuv2rgb: KernelT = Catrom
        """Kernel used for YUV => RGB resampling."""

        rgb2yuv: KernelT | None = None
        """Kernel used for RGB => YUV resampling. Fallbacks to yuv2rgb if None."""

        fp32: bool = True
        """Whether to process in int16 or float32."""

        def __post_init__(self) -> None:
            self._yuv2rgb = Kernel.ensure_obj(self.yuv2rgb)
            self._rgb2yuv = self._yuv2rgb.ensure_obj(self.rgb2yuv)

        def clip2opp(self, clip: vs.VideoNode, is_gray: bool = False) -> vs.VideoNode:
            """
            Convert a clip to the OPP colorspace.

            :param clip:    Clip to be processed.

            :return:        OPP clip.
            """
            assert check_variable_format(clip, self.clip2opp)

            return self.to_fullgray(clip) if is_gray else (
                self.rgb2opp(clip) if clip.format.color_family is vs.RGB else self.yuv2opp(clip)
            )

        def yuv2opp(self, clip: vs.VideoNode) -> vs.VideoNode:
            """
            Convert a YUV clip to the OPP colorspace.

            :param clip:    YUV clip to be processed.

            :return:        OPP clip.
            """
            return self.rgb2opp(self._yuv2rgb.resample(clip, vs.RGBS))

        def rgb2opp(self, clip: vs.VideoNode) -> vs.VideoNode:
            """
            Convert an RGB clip to the OPP colorspace.

            :param clip:    RGB clip to be processed.

            :return:        OPP clip.
            """
            return clip.bm3d.RGB2OPP(self.fp32)

        def opp2rgb(self, clip: vs.VideoNode) -> vs.VideoNode:
            """
            Convert an OPP clip to the RGB colorspace.

            :param clip:    OPP clip to be processed.

            :return:        RGB clip.
            """
            return clip.bm3d.OPP2RGB(self.fp32)

        def opp2yuv(
            self, clip: vs.VideoNode, format: vs.VideoFormat, matrix: Matrix | None, dither: DitherType | None = None
        ) -> vs.VideoNode:
            """
            Convert an OPP clip to the YUV colorspace.

            :param clip:    OPP clip to be processed.

            :return:        YUV clip.
            """
            dither = DitherType.from_param(self._rgb2yuv.kwargs.get('dither_type', None)) or dither

            return self._rgb2yuv.resample(
                self.opp2rgb(clip), format=format, matrix=matrix, dither_type=dither
            )

        def to_fullgray(self, clip: vs.VideoNode) -> vs.VideoNode:
            """
            Extract Y plane from GRAY/YUV clip and if not float32, upsample to it.

            :param clip:    GRAY or YUV clip to be processed.

            :return:        GRAYS clip.
            """
            return Point.resample(clip, vs.GRAYS if self.fp32 else vs.GRAY16)

    wclip: vs.VideoNode

    sigma: _Sigma
    radius: _Radius
    profile: Profile.Config

    ref: vs.VideoNode | None

    refine: int

    resampler = ResampleConfig()

    basic_args: KwargsT
    """Custom kwargs passed to bm3d for the :py:attr:`basic` clip."""

    final_args: KwargsT
    """Custom kwargs passed to bm3d for the :py:attr:`final` clip."""

    class _Sigma(NamedTuple):
        y: float
        u: float
        v: float

    class _Radius(NamedTuple):
        basic: int
        final: int

    def __init__(
        self, clip: vs.VideoNode, sigma: SingleOrArr[float], radius: SingleOrArr[int] | None = None,
        profile: Profile | Profile.Config = Profile.FAST, ref: vs.VideoNode | None = None, refine: int = 1,
        matrix: MatrixT | None = None, range_in: ColorRangeT | None = None
    ) -> None:
        """
        :param clip:            Source clip.
        :param sigma:           Strength of denoising, valid range is [0, +inf].
        :param radius:          Temporal radius, valid range is [1, 16].
        :param profile:         Preset profile. See :py:attr:`vsdenoise.bm3d.Profile`.
                                Default: Profile.FAST.
        :param ref:             Reference clip used in block-matching, replacing the basic estimation.
                                If not specified, the input clip is used instead.
                                Default: None.
        :param refine:          The number of times to refine the estimation.

                                 * 0 means basic estimate only.
                                 * 1 means basic estimate with one final estimate.
                                 * n means basic estimate refined with final estimate applied n times.

                                Default: 1.
        :param matrix:          Enum for the matrix of the input clip.
                                See :py:attr:`vstools.enums.Matrix` for more info.
                                If not specified, gets the matrix from the "_Matrix" prop of the clip
                                unless it's an RGB clip, in which case it stays as `None`.
        :param range_in:        Enum for the color range of the input clip.
                                See :py:attr:`vstools.enums.ColorRange` for more info.
                                If not specified, gets the color from the "_ColorRange" prop of the clip.
                                This check is not performed if the input clip is float.
        """
        assert check_variable(clip, self.__class__)

        self._format = clip.format
        self._dither = DitherType.ERROR_DIFFUSION if DitherType.should_dither(self._format, clip) else DitherType.NONE
        self._clip = self._check_clip(clip, matrix, range_in, self.__class__)

        self._matrix = Matrix.from_video(self._clip, True) if self._format.color_family is vs.YUV else None

        self.sigma = self._Sigma(*normalize_seq(sigma, 3))
        self.radius = self._Radius(*normalize_seq(radius or 0, 2))

        self.profile = profile if isinstance(profile, ProfileBase.Config) else profile()
        self.ref = ref and self._check_clip(ref, matrix, range_in, self.__class__)
        self.refine = refine

        self._is_gray = self._format.color_family == vs.GRAY

        self.basic_args = {}
        self.final_args = {}

        if self._is_gray:
            self.sigma = self.sigma._replace(u=0, v=0)
        elif sum(self.sigma[1:]) == 0:
            self._is_gray = True

        self.__post_init__()

    @abstractmethod
    def basic(self, clip: vs.VideoNode, opp: bool = False) -> vs.VideoNode:
        """
        Retrieve the "basic" clip, typically used as a `ref` clip for :py:attr:`final`.

        :param clip:    OPP or GRAY colorspace clip to be processed.

        :return:        Denoised clip to be used as `ref`.
        """

    @abstractmethod
    def final(
        self, clip: vs.VideoNode | None = None, ref: vs.VideoNode | None = None, refine: int | None = None
    ) -> vs.VideoNode:
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

        :return:        Output clip.
        """
        from inspect import currentframe, getframeinfo

        fi = getframeinfo(currentframe())  # type: ignore

        print(warnings.formatwarning(
            "The `clip` property is deprecated! Please use the .final() method!",
            DeprecationWarning, fi.filename, fi.lineno - 8
        ))
        return self.final()

    def _post_processing(self, clip: vs.VideoNode) -> vs.VideoNode:
        if self._is_gray:
            clip = Point.resample(
                clip, self._format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id,
                dither_type=self._dither
            )
            if self._format.color_family == vs.YUV:
                clip = join(clip, self._clip)
        else:
            clip = self.resampler.opp2yuv(clip, self._format, self._matrix, self._dither)

        if self.sigma.y == 0:
            clip = join(self._clip, clip)

        return clip

    def _check_clip(
        self, clip: vs.VideoNode, matrix: MatrixT | None, range_in: ColorRangeT | None, func: FuncExceptT
    ) -> vs.VideoNode:
        fmt = get_video_format(clip)

        if fmt.sample_type != vs.FLOAT or fmt.bits_per_sample != 32:
            clip = ColorRange.ensure_presence(clip, range_in, func)

        if fmt.color_family is vs.YUV:
            clip = Matrix.ensure_presence(clip, matrix, func)

        return clip

    def _get_clip(self, base: vs.VideoNode, pre: vs.VideoNode, ref: vs.VideoNode | None = None) -> vs.VideoNode:
        return pre if ref is None or ref is base or ref is pre else self.resampler.clip2opp(ref)

    def __post_init__(self) -> None:
        self._pre_clip = self.resampler.clip2opp(self._clip, self._is_gray)
        self._pre_ref = self.ref and self.resampler.clip2opp(self.ref, self._is_gray)

        return super().__post_init__()

    def __vs_del__(self, core_id: int) -> None:
        del self._clip, self._format, self.ref
        self.basic_args.clear()
        self.final_args.clear()


class BM3D(AbstractBM3D):
    """BM3D implementation by mawen1250."""

    def __init__(
        self, clip: vs.VideoNode, sigma: SingleOrArr[float], radius: SingleOrArr[int] | None = None,
        profile: Profile | Profile.Config = Profile.FAST, pre: vs.VideoNode | None = None,
        ref: vs.VideoNode | None = None, refine: int = 1, matrix: MatrixT | None = None,
        range_in: ColorRangeT | None = None
    ) -> None:
        self.pre = pre and self._check_clip(pre, matrix, range_in, self.__class__)

        super().__init__(clip, sigma, radius, profile, ref, refine, matrix, range_in)

    def __post_init__(self) -> None:
        self._pre_pre = self.pre and self.resampler.clip2opp(self.pre)

        return super().__post_init__()

    def basic(self, clip: vs.VideoNode | None = None, opp: bool = False) -> vs.VideoNode:
        clip = self._get_clip(self._clip, self._pre_clip, clip)

        kwargs = KwargsT(ref=self.pre, sigma=self.sigma, matrix=100, args=self.basic_args)

        if self.radius.basic:
            clip = clip.bm3d.VBasic(**self.profile.as_dict(**kwargs, radius=self.radius.basic))  # type: ignore

            clip = clip.bm3d.VAggregate(self.radius.basic, self.resampler.fp32)
        else:
            clip = clip.bm3d.Basic(**self.profile.as_dict(**kwargs))  # type: ignore

        return clip if opp else self._post_processing(clip)

    def final(
        self, clip: vs.VideoNode | None = None, ref: vs.VideoNode | None = None, refine: int | None = None
    ) -> vs.VideoNode:
        clip = self._get_clip(self._clip, self._pre_clip, clip)

        if self.ref and self._pre_ref:
            ref = self._get_clip(self.ref, self._pre_ref, ref)
        else:
            ref = self.basic(clip, True)

        kwargs = KwargsT(ref=ref, sigma=self.sigma, matrix=100, args=self.final_args)

        for _ in range(refine or self.refine):
            if self.radius.final:
                clip = clip.bm3d.VFinal(**self.profile.as_dict(**kwargs, radius=self.radius.final))  # type: ignore
                clip = clip.bm3d.VAggregate(self.radius.final, self.resampler.fp32)
            else:
                clip = clip.bm3d.Final(**self.profile.as_dict(**kwargs))  # type: ignore

        return self._post_processing(clip)


class AbstractBM3DCudaMeta(ABCMeta):
    def __new__(
        __mcls: type[Self], __name: str, __bases: tuple[type, ...], __namespace: dict[str, Any], **kwargs: Any
    ) -> Self:
        cls = super().__new__(__mcls, __name, __bases, __namespace)  # type: ignore
        cls.plugin = kwargs.get('plugin')
        return cls  # type: ignore


class AbstractBM3DCuda(AbstractBM3D, metaclass=AbstractBM3DCudaMeta):
    """BM3D implementation by WolframRhodium."""

    plugin: _Plugin_bm3dcuda_Core_Bound | _Plugin_bm3dcuda_rtc_Core_Bound | _Plugin_bm3dcpu_Core_Bound

    def basic(self, clip: vs.VideoNode | None = None, opp: bool = False) -> vs.VideoNode:
        clip = self._get_clip(self._clip, self._pre_clip, clip)

        clip = self.plugin.BM3D(clip, **self.profile.as_dict(
            True, True, False, self.basic_args, sigma=self.sigma, radius=self.radius.basic
        ))

        if self.radius.basic:
            clip = clip.bm3d.VAggregate(self.radius.basic, 1)

        return clip if opp else self._post_processing(clip)

    def final(
        self, clip: vs.VideoNode | None = None, ref: vs.VideoNode | None = None, refine: int | None = None
    ) -> vs.VideoNode:
        clip = self._get_clip(self._clip, self._pre_clip, clip)

        if self.ref and self._pre_ref:
            ref = self._get_clip(self.ref, self._pre_ref, ref)
        else:
            ref = self.basic(clip, True)

        for _ in range(refine or self.refine):
            clip = self.plugin.BM3D(clip, ref, **self.profile.as_dict(
                True, False, True, self.final_args, sigma=self.sigma, radius=self.radius.final
            ))

            if self.radius.final:
                clip = clip.bm3d.VAggregate(self.radius.final, 1)

        return self._post_processing(clip)


class BM3DCuda(AbstractBM3DCuda, plugin=core.lazy.bm3dcuda):
    ...


class BM3DCudaRTC(AbstractBM3DCuda, plugin=core.lazy.bm3dcuda_rtc):
    ...


class BM3DCPU(AbstractBM3DCuda, plugin=core.lazy.bm3dcpu):
    ...
