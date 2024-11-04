"""
This module implements wrappers for BM3D
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, final, overload

from vstools import (
    MISSING, ColorRange, ColorRangeT, Colorspace, ConstantFormatVideoNode, CustomIndexError, CustomRuntimeError,
    CustomStrEnum, CustomValueError, FieldBased, FuncExceptT, FunctionUtil, KwargsT, Matrix, MatrixT, MissingT, PlanesT,
    Self, SingleOrArr, check_variable, core, depth, get_video_format, get_y, is_gpu_available, join, normalize_seq, vs,
    vs_object, UnsupportedFieldBasedError
)

from .types import _Plugin_bm3dcpu_Core_Bound, _Plugin_bm3dcuda_Core_Bound, _Plugin_bm3dcuda_rtc_Core_Bound

__all__ = [
    'Profile',

    'BM3D',

    'BM3DMawen',

    'BM3DCuda', 'BM3DCudaRTC', 'BM3DCPU'
]


@dataclass
class BM3DColorspaceConfig:
    csp: Colorspace
    clip: ConstantFormatVideoNode
    matrix: Matrix | None
    chroma_only: bool

    fp32: bool

    @overload
    def check_clip(
        self, clip: vs.VideoNode, matrix: MatrixT | None, range_in: ColorRangeT | None, func: FuncExceptT
    ) -> ConstantFormatVideoNode:
        ...

    @overload
    def check_clip(
        self, clip: None, matrix: MatrixT | None, range_in: ColorRangeT | None, func: FuncExceptT
    ) -> None:
        ...

    def check_clip(
        self, clip: vs.VideoNode | None, matrix: MatrixT | None, range_in: ColorRangeT | None, func: FuncExceptT
    ) -> ConstantFormatVideoNode | None:
        if clip is None:
            return None

        fmt = get_video_format(clip)

        if fmt.sample_type != vs.FLOAT or fmt.bits_per_sample != 32:
            clip = ColorRange.ensure_presence(clip, range_in or ColorRange.from_video(clip), func)

        if fmt.color_family == vs.YUV and (self.csp.is_rgb or self.csp.is_opp):
            clip = Matrix.ensure_presence(clip, matrix, func)

        assert check_variable(clip, func)

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

        assert check_variable(clip, self.prepare_clip)

        return self.csp.from_clip(clip, self.fp32, self.prepare_clip)

    def post_processing(self, clip: vs.VideoNode) -> vs.VideoNode:
        assert clip.format

        if self.clip.format.color_family is vs.YUV:
            if clip.format.color_family is vs.GRAY:
                clip = join(depth(clip, self.clip), self.clip)
            else:
                clip = self.csp.to_yuv(clip, self.fp32, self.post_processing, self.clip, matrix=self.matrix)
        elif self.clip.format.color_family is vs.RGB:
            clip = self.csp.to_rgb(clip, self.fp32, self.post_processing, self.clip)

        if self.chroma_only:
            clip = join(self.clip, clip)

        return depth(clip, self.clip)


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
            self, cuda: vs.Plugin | Literal[False], basic: bool = False, aggregate: bool = False,
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
                func = cuda.BM3Dv2 if hasattr(cuda, 'BM3Dv2') else cuda.BM3D
                cuda_keys = set[str](func.__signature__.parameters.keys())

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
    tr: _TemporalRadius
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

    class _TemporalRadius(NamedTuple):
        basic: int
        final: int

    def __init__(
        self, clip: vs.VideoNode, sigma: SingleOrArr[float] = 0.5, tr: SingleOrArr[int] | None = None,
        profile: Profile | Profile.Config = Profile.FAST, ref: vs.VideoNode | None = None, refine: int = 1,
        matrix: MatrixT | None = None, range_in: ColorRangeT | None = None,
        colorspace: Colorspace | None = None, fp32: bool = True, *, radius: SingleOrArr[int] | MissingT = MISSING
    ) -> None:
        """
        :param clip:            Source clip.
        :param sigma:           Strength of denoising, valid range is [0, +inf].
        :param tr:              Temporal radius, valid range is [1, 16].
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
        self.tr = self._TemporalRadius(*normalize_seq(tr or 0, 2))

        _is_gray = clip.format.color_family == vs.GRAY

        if _is_gray:
            self.sigma = self.sigma._replace(u=0, v=0)
        elif sum(self.sigma[1:]) == 0:
            _is_gray = True

        if _is_gray:
            colorspace = Colorspace.GRAY
        elif colorspace is None:
            colorspace = Colorspace.OPP_BM3D

        matrix = Matrix.from_param(matrix)

        if (fb := FieldBased.from_video(clip, False, self.__class__)).is_inter:
            raise UnsupportedFieldBasedError('Interlaced input is not supported!', self.__class__, fb)

        self.cspconfig = BM3DColorspaceConfig(colorspace, clip, matrix, self.sigma.y == 0, fp32)

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

    @classmethod
    def denoise(
        cls, clip: vs.VideoNode, sigma: SingleOrArr[float] = 0.5, tr: SingleOrArr[int] | None = None,
        refine: int = 1, profile: Profile | Profile.Config = Profile.FAST, ref: vs.VideoNode | None = None,
        matrix: MatrixT | None = None, range_in: ColorRangeT | None = None,
        colorspace: Colorspace | None = None, fp32: bool = True, planes: PlanesT = None, **kwargs: Any
    ) -> vs.VideoNode:
        func = FunctionUtil(clip, cls.denoise, planes)

        sigma = func.norm_seq(sigma)

        ref = get_y(ref) if func.luma_only and ref else ref

        bm3d = cls(func.work_clip, sigma, tr, profile, ref, refine, matrix, range_in, colorspace, fp32, **kwargs)

        if refine:
            denoise = bm3d.final()
        else:
            denoise = bm3d.basic(
                bm3d.cspconfig.get_clip(bm3d.cspconfig.clip, bm3d._pre_clip, None)
            )

        return func.return_clip(denoise)

    def __post_init__(self) -> None:
        super().__post_init__()

        self._pre_clip = self.cspconfig.prepare_clip(self.cspconfig.clip)
        self._pre_ref = self.cspconfig.prepare_clip(self.ref)

    def __vs_del__(self, core_id: int) -> None:
        del self.cspconfig, self.ref

        self.basic_args.clear()
        self.final_args.clear()


class BM3DMawen(AbstractBM3D):
    """BM3D implementation by mawen1250."""

    def __init__(
        self, clip: vs.VideoNode, sigma: SingleOrArr[float] = 0.5, tr: SingleOrArr[int] | None = None,
        profile: Profile | Profile.Config = Profile.FAST, pre: vs.VideoNode | None = None,
        ref: vs.VideoNode | None = None, refine: int = 1, matrix: MatrixT | None = None,
        range_in: ColorRangeT | None = None, colorspace: Colorspace | None = None, fp32: bool = True,
        *, radius: SingleOrArr[int] | MissingT = MISSING
    ) -> None:
        super().__init__(clip, sigma, tr, profile, ref, refine, matrix, range_in, colorspace, fp32, radius=radius)

        self.pre = pre and self.cspconfig.check_clip(pre, matrix, range_in, self.__class__)

    def __post_init__(self) -> None:
        super().__post_init__()

        self._pre_pre = self.cspconfig.prepare_clip(self.pre)

    def basic(self, clip: vs.VideoNode | None = None, opp: bool = False) -> vs.VideoNode:
        clip = self.cspconfig.get_clip(self.cspconfig.clip, self._pre_clip, clip)

        kwargs = KwargsT(ref=self.pre, sigma=self.sigma, matrix=100, args=self.basic_args)

        if self.tr.basic:
            clip = clip.bm3d.VBasic(**self.profile.as_dict(**kwargs, radius=self.tr.basic))  # type: ignore

            clip = clip.bm3d.VAggregate(self.tr.basic, self.cspconfig.fp32)
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
            if self.tr.final:
                clip = clip.bm3d.VFinal(**self.profile.as_dict(**kwargs, radius=self.tr.final))  # type: ignore
                clip = clip.bm3d.VAggregate(self.tr.final, self.cspconfig.fp32)
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
            self.plugin, True, False, self.basic_args, sigma=self.sigma, radius=self.tr.basic
        )

        if hasattr(self.plugin, 'BM3Dv2'):
            clip = self.plugin.BM3Dv2(clip, **kwargs)
        else:
            clip = self.plugin.BM3D(clip, **kwargs)

            if self.tr.basic:
                clip = clip.bm3d.VAggregate(self.tr.basic, self.cspconfig.fp32)

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
            self.plugin, False, True, self.final_args, sigma=self.sigma, radius=self.tr.final
        )

        if hasattr(self.plugin, 'BM3Dv2'):
            for _ in range(refine or self.refine):
                clip = self.plugin.BM3Dv2(clip, ref, **kwargs)
        else:
            for _ in range(refine or self.refine):
                clip = self.plugin.BM3D(clip, ref, **kwargs)

                if self.tr.final:
                    clip = clip.bm3d.VAggregate(self.tr.final, self.cspconfig.fp32)

        return self.cspconfig.post_processing(clip)


class BM3DCuda(AbstractBM3DCuda, plugin=core.lazy.bm3dcuda):
    ...


class BM3DCudaRTC(AbstractBM3DCuda, plugin=core.lazy.bm3dcuda_rtc):
    ...


class BM3DCPU(AbstractBM3DCuda, plugin=core.lazy.bm3dcpu):
    ...


class BM3D(AbstractBM3D):
    def __new__(cls, *args: Any, **kwargs: Any) -> AbstractBM3D:  # type: ignore
        new_cls: type[AbstractBM3D] | None = None
        gpu_available = is_gpu_available()

        if gpu_available and hasattr(core, 'bm3dcuda_rtc'):
            new_cls = BM3DCudaRTC
        elif gpu_available and hasattr(core, 'bm3dcuda'):
            new_cls = BM3DCuda
        elif hasattr(core, 'bm3dcpu'):
            new_cls = BM3DCPU
        elif hasattr(core, 'bm3d'):
            new_cls = BM3DMawen

        if new_cls is None:
            raise CustomRuntimeError('You have no bm3d plugin installed!')

        return new_cls(*args, **kwargs)
