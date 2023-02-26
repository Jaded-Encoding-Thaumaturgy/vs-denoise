"""
This module implements wrappers for BM3D
"""

from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple, final, overload

from vskernels import Point
from vstools import (
    ColorRange, ColorRangeT, CustomIndexError, CustomIntEnum, CustomStrEnum, CustomValueError, DitherType, FuncExceptT,
    KwargsT, Matrix, MatrixT, ResampleUtil, Self, SingleOrArr, check_variable, core, depth, get_video_format,
    inject_self, join, normalize_seq, vs, vs_object
)

from .types import _Plugin_bm3dcpu_Core_Bound, _Plugin_bm3dcuda_Core_Bound, _Plugin_bm3dcuda_rtc_Core_Bound

__all__ = [
    'Profile',
    'BM3D', 'BM3DCuda', 'BM3DCudaRTC', 'BM3DCPU',
    'BM3DColorspace'
]


class ResampleBM3D(ResampleUtil):
    @inject_self
    def rgb2opp(self, clip: vs.VideoNode, func: FuncExceptT | None = None) -> vs.VideoNode:  # type: ignore
        return clip.bm3d.RGB2OPP(self.fp32)

    @inject_self
    def opp2rgb(self, clip: vs.VideoNode, func: FuncExceptT | None = None) -> vs.VideoNode:  # type: ignore
        return clip.bm3d.OPP2RGB(self.fp32)


@dataclass
class BM3DColorspaceConfig:
    csp_type: BM3DColorspace
    clip: vs.VideoNode
    matrix: Matrix | None
    format: vs.VideoFormat
    is_gray: bool
    chroma_only: bool

    resampler: ResampleUtil

    @overload
    def check_clip(
        self, clip: vs.VideoNode, matrix: MatrixT | None, range_in: ColorRangeT | None, func: FuncExceptT
    ) -> vs.VideoNode:
        ...

    @overload
    def check_clip(
        self, clip: None, matrix: MatrixT | None, range_in: ColorRangeT | None, func: FuncExceptT
    ) -> None:
        ...

    def check_clip(
        self, clip: vs.VideoNode | None, matrix: MatrixT | None, range_in: ColorRangeT | None, func: FuncExceptT
    ) -> vs.VideoNode | None:
        if clip is None:
            return None

        fmt = get_video_format(clip)

        if fmt.sample_type != vs.FLOAT or fmt.bits_per_sample != 32 and self.resampler.fp32:
            clip = ColorRange.ensure_presence(clip, range_in or ColorRange.from_video(clip), func)

        if fmt.color_family == vs.YUV and (self.csp_type.is_rgb or self.csp_type.is_opp):
            clip = Matrix.ensure_presence(clip, matrix or Matrix.from_video(clip), func)

        return clip

    def get_clip(self, base: vs.VideoNode, pre: vs.VideoNode, ref: vs.VideoNode | None = None) -> vs.VideoNode:
        return pre if ref is None or ref is base or ref is pre else self.prepare_clip(ref)

    @overload
    def prepare_clip(self, clip: vs.VideoNode) -> vs.VideoNode:
        ...

    @overload
    def prepare_clip(self, clip: None) -> None:
        ...

    def prepare_clip(self, clip: vs.VideoNode | None) -> vs.VideoNode | None:
        if clip is None:
            return None

        assert clip.format

        if self.csp_type.is_rgb:
            if clip.format.color_family != vs.RGB:
                clip = clip.resize.Bicubic(format=vs.RGBS)
        elif self.csp_type.is_yuv:
            if clip.format.color_family != vs.YUV:  # type: ignore
                clip = clip.resize.Bicubic(
                    format=self.format.replace(
                        color_family=vs.YUV, sample_type=vs.FLOAT, bits_per_sample=32, subsampling_h=0, subsampling_w=0
                    ).id, matrix=self.matrix
                )
        elif self.csp_type.is_opp:
            clip = self.resampler.clip2opp(clip, self.is_gray)

        return depth(
            clip, self.format.bits_per_sample if self.resampler.fp32 is None else (32 if self.resampler.fp32 else 16)
        )

    def post_processing(self, clip: vs.VideoNode) -> vs.VideoNode:
        dither = DitherType.ERROR_DIFFUSION if DitherType.should_dither(self.format, clip) else DitherType.NONE

        if self.is_gray:
            clip = Point.resample(
                clip, self.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id,
                dither_type=dither
            )

            if self.format.color_family == vs.YUV:
                clip = join(clip, self.clip)
        elif self.csp_type.is_opp:
            if self.format.color_family == vs.YUV:
                clip = self.resampler.opp2yuv(clip, self.format, self.matrix, dither)
            else:
                clip = self.resampler.opp2rgb(clip)

        if self.chroma_only:
            clip = join(self.clip, clip)

        return clip


class BM3DColorspace(CustomIntEnum):
    AUTO = -1
    OPP_OLD = 0
    OPP = 1
    YUV = 2
    RGB = 3

    @property
    def is_opp(self) -> bool:
        return 'OPP' in self.name

    @property
    def is_rgb(self) -> bool:
        return 'RGB' in self.name

    @property
    def is_yuv(self) -> bool:
        return 'YUV' in self.name

    def __call__(
        self, clip: vs.VideoNode, format: vs.VideoFormat, is_gray: bool, chroma_only: bool
    ) -> BM3DColorspaceConfig:
        if self is self.AUTO:
            self = self.OPP_OLD if hasattr(core, 'bm3d') else self.OPP  # type: ignore

        resampler = (ResampleBM3D if self is self.OPP_OLD else ResampleUtil)(True)

        return BM3DColorspaceConfig(
            self, clip, Matrix.from_video(clip) if format.color_family == vs.YUV else None,
            format, is_gray, chroma_only, resampler
        )


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

    wclip: vs.VideoNode

    sigma: _Sigma
    radius: _Radius
    profile: Profile.Config

    ref: vs.VideoNode | None

    refine: int

    cspconfig: BM3DColorspaceConfig

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
        matrix: MatrixT | None = None, range_in: ColorRangeT | None = None,
        colorspace: BM3DColorspace = BM3DColorspace.AUTO
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

        self.sigma = self._Sigma(*normalize_seq(sigma, 3))
        self.radius = self._Radius(*normalize_seq(radius or 0, 2))

        _is_gray = clip.format.color_family == vs.GRAY

        if _is_gray:
            self.sigma = self.sigma._replace(u=0, v=0)
        elif sum(self.sigma[1:]) == 0:
            _is_gray = True

        self.cspconfig = colorspace(clip, clip.format, _is_gray, self.sigma.y == 0)
        self.cspconfig.clip = self.cspconfig.check_clip(clip, matrix, range_in, self.__class__)

        self.profile = profile if isinstance(profile, ProfileBase.Config) else profile()
        self.ref = self.cspconfig.check_clip(ref, matrix, range_in, self.__class__)
        if refine < 1:
            raise CustomIndexError('"refine" must be >= 1!', self.__class__)

        self.refine = refine

        self.basic_args = {}
        self.final_args = {}

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

    @classmethod
    def denoise(
        cls, clip: vs.VideoNode, sigma: SingleOrArr[float], radius: SingleOrArr[int] | None = None,
        refine: int = 1, profile: Profile | Profile.Config = Profile.FAST, ref: vs.VideoNode | None = None,
        matrix: MatrixT | None = None, range_in: ColorRangeT | None = None,
        colorspace: BM3DColorspace = BM3DColorspace.OPP
    ) -> vs.VideoNode:
        return cls(clip, sigma, radius, profile, ref, refine, matrix, range_in, colorspace).final()

    def __post_init__(self) -> None:
        self._pre_clip = self.cspconfig.prepare_clip(self.cspconfig.clip)
        self._pre_ref = self.cspconfig.prepare_clip(self.ref)

        return super().__post_init__()

    def __vs_del__(self, core_id: int) -> None:
        del self.cspconfig, self.ref
        self.basic_args.clear()
        self.final_args.clear()


class BM3D(AbstractBM3D):
    """BM3D implementation by mawen1250."""

    def __init__(
        self, clip: vs.VideoNode, sigma: SingleOrArr[float], radius: SingleOrArr[int] | None = None,
        profile: Profile | Profile.Config = Profile.FAST, pre: vs.VideoNode | None = None,
        ref: vs.VideoNode | None = None, refine: int = 1, matrix: MatrixT | None = None,
        range_in: ColorRangeT | None = None, colorspace: BM3DColorspace = BM3DColorspace.AUTO
    ) -> None:
        self.pre = pre and self.cspconfig.check_clip(pre, matrix, range_in, self.__class__)

        super().__init__(clip, sigma, radius, profile, ref, refine, matrix, range_in, colorspace)

    def __post_init__(self) -> None:
        self._pre_pre = self.cspconfig.prepare_clip(self.pre)

        return super().__post_init__()

    def basic(self, clip: vs.VideoNode | None = None, opp: bool = False) -> vs.VideoNode:
        clip = self.cspconfig.get_clip(self.cspconfig.clip, self._pre_clip, clip)

        kwargs = KwargsT(ref=self.pre, sigma=self.sigma, matrix=100, args=self.basic_args)

        if self.radius.basic:
            clip = clip.bm3d.VBasic(**self.profile.as_dict(**kwargs, radius=self.radius.basic))  # type: ignore

            clip = clip.bm3d.VAggregate(self.radius.basic, self.cspconfig.resampler.fp32)
        else:
            clip = clip.bm3d.Basic(**self.profile.as_dict(**kwargs))  # type: ignore

        return clip if opp else self.cspconfig.post_processing(clip)

    def final(
        self, clip: vs.VideoNode | None = None, ref: vs.VideoNode | None = None, refine: int | None = None
    ) -> vs.VideoNode:
        clip = self.cspconfig.get_clip(self.cspconfig.clip, self._pre_clip, clip)

        if self.ref and self._pre_ref:
            ref = self.cspconfig.get_clip(self.ref, self._pre_ref, ref)
        else:
            ref = self.basic(clip, True)

        kwargs = KwargsT(ref=ref, sigma=self.sigma, matrix=100, args=self.final_args)

        for _ in range(refine or self.refine):
            if self.radius.final:
                clip = clip.bm3d.VFinal(**self.profile.as_dict(**kwargs, radius=self.radius.final))  # type: ignore
                clip = clip.bm3d.VAggregate(self.radius.final, self.cspconfig.resampler.fp32)
            else:
                clip = clip.bm3d.Final(**self.profile.as_dict(**kwargs))  # type: ignore

        return self.cspconfig.post_processing(clip)


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
        clip = self.cspconfig.get_clip(self.cspconfig.clip, self._pre_clip, clip)

        kwargs = self.profile.as_dict(
            True, True, False, self.basic_args, sigma=self.sigma, radius=self.radius.basic
        )

        if hasattr(self.plugin, 'BM3Dv2'):
            clip = self.plugin.BM3Dv2(clip, **kwargs)
        else:
            clip = self.plugin.BM3D(clip, **kwargs)

            if self.radius.basic:
                clip = clip.bm3d.VAggregate(self.radius.basic, self.cspconfig.resampler.fp32)

        return clip if opp else self.cspconfig.post_processing(clip)

    def final(
        self, clip: vs.VideoNode | None = None, ref: vs.VideoNode | None = None, refine: int | None = None
    ) -> vs.VideoNode:
        clip = self.cspconfig.get_clip(self.cspconfig.clip, self._pre_clip, clip)

        if self.ref and self._pre_ref:
            ref = self.cspconfig.get_clip(self.ref, self._pre_ref, ref)
        else:
            ref = self.basic(clip, True)

        kwargs = self.profile.as_dict(
            True, False, True, self.final_args, sigma=self.sigma, radius=self.radius.final
        )

        if hasattr(self.plugin, 'BM3Dv2'):
            for _ in range(refine or self.refine):
                clip = self.plugin.BM3Dv2(clip, ref, **kwargs)
        else:
            for _ in range(refine or self.refine):
                clip = self.plugin.BM3D(clip, ref, **kwargs)

                if self.radius.final:
                    clip = clip.bm3d.VAggregate(self.radius.final, self.cspconfig.resampler.fp32)

        return self.cspconfig.post_processing(clip)


class BM3DCuda(AbstractBM3DCuda, plugin=core.lazy.bm3dcuda):
    ...


class BM3DCudaRTC(AbstractBM3DCuda, plugin=core.lazy.bm3dcuda_rtc):
    ...


class BM3DCPU(AbstractBM3DCuda, plugin=core.lazy.bm3dcpu):
    ...
