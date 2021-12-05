from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import (Any, ClassVar, Dict, Mapping, NamedTuple, Optional,
                    Sequence, TypedDict, Union)

import vapoursynth as vs
# TODO: Move lvsfunc.kernels to vsutil
from lvsfunc.kernels import Catrom, Kernel
from vsutil import Dither as DitherType
from vsutil import get_depth, get_y, iterate

from .types import (MATRIX, DataArray, PluginBm3dcpuCoreUnbound,
                    PluginBm3dcuda_rtcCoreUnbound, PluginBm3dcudaCoreUnbound)

core = vs.core


class Profile(str, Enum):
    FAST = 'fast'
    LOW_COMPLEXITY = 'lc'
    NORMAL = 'np'
    HIGH = 'high'
    VERY_NOISY = 'vn'

    F = FAST
    LC = LOW_COMPLEXITY
    NP = NORMAL
    H = HIGH
    VN = VERY_NOISY


class AbstractBM3D(ABC):
    wclip: vs.VideoNode
    sigma: _Sigma
    radius: _Radius
    profile: Optional[Profile]
    ref: Optional[vs.VideoNode]
    refine: int
    yuv2rgb_kernel: Kernel
    rgb2yuv_kernel: Kernel

    is_gray: bool

    basic_args: Mapping[str, Any]
    final_args: Mapping[str, Any]

    _refv: vs.VideoNode
    _clip: vs.VideoNode
    _format: vs.VideoFormat

    class _Sigma(NamedTuple):
        y: float
        u: float
        v: float

    class _Radius(NamedTuple):
        basic: int
        final: int

    def __init__(
        self, clip: vs.VideoNode, /,
        sigma: float | Sequence[float], radius: int | Sequence[int] | None = None,
        profile: Profile = Profile.FAST,
        ref: Optional[vs.VideoNode] = None,
        refine: int = 1,
        matrix: vs.MatrixCoefficients | MATRIX = vs.MATRIX_BT709,
        yuv2rgb_kernel: Kernel = Catrom(),
        rgb2yuv_kernel: Kernel = Catrom()
    ) -> None:
        if clip.format is None:
            raise ValueError(f"{self.__class__.__name__}: Variable format clips not supported")

        self._format = clip.format
        self._clip = clip
        self._check_clips(clip, ref)

        self.wclip = clip
        if not isinstance(sigma, Sequence):
            self.sigma = self._Sigma(sigma, sigma, sigma)
        else:
            self.sigma = self._Sigma(*(list(sigma) + [sigma[-1]] * (3 - len(sigma)))[:3])
        if radius is None:
            self.radius = self._Radius(0, 0)
        elif not isinstance(radius, Sequence):
            self.radius = self._Radius(radius, radius)
        else:
            self.radius = self._Radius(*(list(radius) + [radius[-1]] * (2 - len(radius)))[:2])
        self.profile = profile
        self.ref = ref
        self.refine = refine
        self.yuv2rgb_kernel = yuv2rgb_kernel
        self.yuv2rgb_kernel.kwargs.update(format=vs.RGBS, matrix_in=matrix)
        self.rgb2yuv_kernel = rgb2yuv_kernel
        self.rgb2yuv_kernel.kwargs.update(format=self._format.id, matrix=matrix)

        self.is_gray = clip.format.color_family == vs.GRAY

        self.basic_args = {}
        self.final_args = {}

        if self.is_gray:
            self.sigma = self.sigma._replace(u=0, v=0)

        if sum(self.sigma[1:]) == 0:
            self.is_gray = True

    def yuv2opp(self, clip: vs.VideoNode) -> vs.VideoNode:
        return self.rgb2opp(self.yuv2rgb_kernel.scale(clip, clip.width, clip.height))

    def rgb2opp(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.bm3d.RGB2OPP(sample=1)

    def opp2rgb(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.bm3d.OPP2RGB(sample=1)

    def to_fullgray(self, clip: vs.VideoNode) -> vs.VideoNode:
        return get_y(clip).resize.Point(format=vs.GRAYS)

    @abstractmethod
    def basic(self, clip: vs.VideoNode) -> vs.VideoNode:
        ...

    @abstractmethod
    def final(self, clip: vs.VideoNode) -> vs.VideoNode:
        ...

    @property
    def clip(self) -> vs.VideoNode:
        self._preprocessing()

        # Make basic estimation
        if self.ref is None:
            self._refv = self.basic(self.wclip)
        else:
            if self.is_gray:
                self._refv = self.to_fullgray(self.ref)
            else:
                self._refv = self.yuv2opp(self.ref)

        # Make final estimation
        self.wclip = iterate(self.wclip, self.final, self.refine)

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
        if self.is_gray:
            self.wclip = core.resize.Point(
                self.wclip, format=self._format.replace(
                    color_family=vs.GRAY, subsampling_w=0, subsampling_h=0
                ).id,
                dither_type=self._get_dither_type()
            )
            if self._format.color_family == vs.YUV:
                self.wclip = core.std.ShufflePlanes([self.wclip, self._clip], [0, 1, 2], vs.YUV)
        else:
            if 'dither_type' not in self.rgb2yuv_kernel.kwargs:
                self.rgb2yuv_kernel.kwargs.update(dither_type=self._get_dither_type())
            self.wclip = self.rgb2yuv_kernel.scale(self.opp2rgb(self.wclip), self.wclip.width, self.wclip.height)

        if self.sigma.y == 0:
            self.wclip = core.std.ShufflePlanes([self._clip, self.wclip], [0, 1, 2], vs.YUV)

    def _get_dither_type(self) -> str:
        return DitherType.ERROR_DIFFUSION  \
            if self._format.bits_per_sample < get_depth(self.wclip) else DitherType.NONE

    def _check_clips(self, *clips: Optional[vs.VideoNode]) -> None:
        for c in [c for c in clips if c]:
            assert c.format
            if c.format.color_family != vs.RGB and any(
                p not in c.get_frame(0).props.keys()
                for p in ['_ColorRange', '_Matrix']
            ):
                raise ValueError(f'{self.__class__.__name__}: "_ColorRange" or "_Matrix" prop missing')


class BM3D(AbstractBM3D):
    class BasicArgs(TypedDict, total=False):
        block_size: Optional[int]
        block_step: Optional[int]
        group_size: Optional[int]
        bm_range: Optional[int]
        bm_step: Optional[int]
        th_mse: Optional[float]
        hard_thr: Optional[float]
        matrix: Optional[int]

    class FinalArgs(TypedDict, total=False):
        block_size: Optional[int]
        block_step: Optional[int]
        group_size: Optional[int]
        bm_range: Optional[int]
        bm_step: Optional[int]
        th_mse: Optional[float]
        matrix: Optional[int]

    class VBasicArgs(TypedDict, total=False):
        block_size: Optional[int]
        block_step: Optional[int]
        group_size: Optional[int]
        bm_range: Optional[int]
        bm_step: Optional[int]
        ps_num: Optional[int]
        ps_range: Optional[int]
        ps_step: Optional[int]
        th_mse: Optional[float]
        hard_thr: Optional[float]
        matrix: Optional[int]

    class VFinalArgs(TypedDict, total=False):
        block_size: Optional[int]
        block_step: Optional[int]
        group_size: Optional[int]
        bm_range: Optional[int]
        bm_step: Optional[int]
        ps_num: Optional[int]
        ps_range: Optional[int]
        ps_step: Optional[int]
        th_mse: Optional[float]
        matrix: Optional[int]

    basic_args: BasicArgs | VBasicArgs
    final_args: FinalArgs | VFinalArgs

    pre: Optional[vs.VideoNode]
    fp32: bool = True

    def __init__(
        self, clip: vs.VideoNode, /,
        sigma: float | Sequence[float], radius: int | Sequence[int] | None = None,
        profile: Profile = Profile.FAST,
        pre: Optional[vs.VideoNode] = None, ref: Optional[vs.VideoNode] = None,
        refine: int = 1,
        matrix: vs.MatrixCoefficients | MATRIX = vs.MATRIX_BT709,
        yuv2rgb_kernel: Kernel = Catrom(),
        rgb2yuv_kernel: Kernel = Catrom()
    ) -> None:
        super().__init__(clip, sigma, radius, profile, ref, refine, matrix, yuv2rgb_kernel, rgb2yuv_kernel)
        self._check_clips(pre)
        self.pre = pre

    def rgb2opp(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.bm3d.RGB2OPP(self.fp32)

    def opp2rgb(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.bm3d.OPP2RGB(self.fp32)

    def to_fullgray(self, clip: vs.VideoNode) -> vs.VideoNode:
        return get_y(clip).resize.Point(format=vs.GRAYS if self.fp32 else vs.GRAY16)

    def basic(self, clip: vs.VideoNode) -> vs.VideoNode:
        kwargs: Dict[str, Any] = dict(ref=self.pre, profile=self.profile, sigma=self.sigma, matrix=100)
        if self.radius and self.radius.basic:
            clip = core.bm3d.VBasic(clip, radius=self.radius.basic, **kwargs | self.basic_args)
            clip = core.bm3d.VAggregate(clip, self.radius.basic, self.fp32)
        else:
            clip = core.bm3d.Basic(clip, **kwargs | self.basic_args)
        return clip

    def final(self, clip: vs.VideoNode) -> vs.VideoNode:
        kwargs: Dict[str, Any] = dict(profile=self.profile, sigma=self.sigma, matrix=100)
        if self.radius.final:
            clip = core.bm3d.VFinal(clip, ref=self._refv, radius=self.radius.final, **kwargs | self.final_args)
            clip = core.bm3d.VAggregate(clip, self.radius.final, self.fp32)
        else:
            clip = core.bm3d.Final(clip, ref=self._refv, **kwargs | self.final_args)
        return clip

    def _preprocessing(self) -> None:
        # Initialise pre
        if self.pre is not None:
            if self.is_gray:
                self.pre = self.to_fullgray(self.pre)
            else:
                self.pre = self.yuv2opp(self.pre)

        super()._preprocessing()


class _AbstractBM3DCuda(AbstractBM3D):
    plugin: ClassVar[
        Union[
            PluginBm3dcudaCoreUnbound,
            PluginBm3dcuda_rtcCoreUnbound,
            PluginBm3dcpuCoreUnbound
        ]
    ]

    CUDA_BASIC_PROFILES: ClassVar[Dict[str | None, Dict[str, Any]]] = {
        Profile.FAST: dict(block_step=8, bm_range=9),
        Profile.LC: dict(block_step=6, bm_range=9),
        Profile.NP: dict(block_step=4, bm_range=16),
        Profile.HIGH: dict(block_step=3, bm_range=16),
        # 'vn': dict(block_step=4, bm_range=16),
        None: {}
    }
    CUDA_FINAL_PROFILES: ClassVar[Dict[str | None, Dict[str, Any]]] = {
        Profile.FAST: dict(block_step=7, bm_range=9),
        Profile.LC: dict(block_step=5, bm_range=9),
        Profile.NP: dict(block_step=3, bm_range=16),
        Profile.HIGH: dict(block_step=2, bm_range=16),
        # 'vn': dict(block_step=4, bm_range=16),
        None: {}
    }
    CUDA_VBASIC_PROFILES: ClassVar[Dict[str | None, Dict[str, Any]]] = {
        Profile.FAST: dict(block_step=8, bm_range=7, ps_num=2, ps_range=4),
        Profile.LC: dict(block_step=6, bm_range=9, ps_num=2, ps_range=4),
        Profile.NP: dict(block_step=4, bm_range=12, ps_num=2, ps_range=5),
        Profile.HIGH: dict(block_step=3, bm_range=16, ps_num=2, ps_range=7),
        # 'vn': dict(block_step=4, bm_range=16),
        None: {}
    }
    CUDA_VFINAL_PROFILES: ClassVar[Dict[str | None, Dict[str, Any]]] = {
        Profile.FAST: dict(block_step=7, bm_range=7, ps_num=2, ps_range=5),
        Profile.LC: dict(block_step=5, bm_range=9, ps_num=2, ps_range=5),
        Profile.NP: dict(block_step=3, bm_range=12, ps_num=2, ps_range=6),
        Profile.HIGH: dict(block_step=2, bm_range=16, ps_num=2, ps_range=8),
        # 'vn': dict(block_step=4, bm_range=16),
        None: {}
    }

    def basic(self, clip: vs.VideoNode) -> vs.VideoNode:
        kwargs: Dict[str, Any] = dict(sigma=self.sigma)
        if self.radius.basic > 0:
            clip = self.plugin.BM3D(
                clip, radius=self.radius.basic,
                **kwargs | self.CUDA_VBASIC_PROFILES[self.profile] | self.basic_args
            ).bm3d.VAggregate(self.radius.basic, 1)
        else:
            clip = self.plugin.BM3D(
                clip, radius=0,
                **kwargs | self.CUDA_BASIC_PROFILES[self.profile] | self.basic_args
            )
        return clip

    def final(self, clip: vs.VideoNode) -> vs.VideoNode:
        kwargs: Dict[str, Any] = dict(sigma=self.sigma)
        if self.radius.final:
            clip = self.plugin.BM3D(
                clip, self._refv, radius=self.radius.final,
                **kwargs | self.CUDA_VFINAL_PROFILES[self.profile] | self.final_args
            ).bm3d.VAggregate(self.radius.final, 1)
        else:
            clip = self.plugin.BM3D(
                clip, ref=self._refv, radius=0,
                **kwargs | self.CUDA_FINAL_PROFILES[self.profile] | self.final_args
            )
        return clip

    @property
    def clip(self) -> vs.VideoNode:
        if self.profile == Profile.VERY_NOISY:
            raise ValueError(f'{self.__class__.__name__}: Profile "vn" is not supported!')
        return super().clip


class BM3DCuda(_AbstractBM3DCuda):
    plugin = core.bm3dcuda

    class BasicArgs(TypedDict, total=False):
        block_step: Optional[int]
        bm_range: Optional[int]
        chroma: bool
        device_id: int
        fast: bool
        extractor_exp: int

    class FinalArgs(TypedDict, total=False):
        block_step: Optional[int]
        bm_range: Optional[int]
        chroma: bool
        device_id: int
        fast: bool
        extractor_exp: int

    class VBasicArgs(TypedDict, total=False):
        block_step: Optional[int]
        bm_range: Optional[int]
        ps_num: Optional[int]
        ps_range: Optional[int]
        chroma: bool
        device_id: int
        fast: bool
        extractor_exp: int

    class VFinalArgs(TypedDict, total=False):
        block_step: Optional[int]
        bm_range: Optional[int]
        ps_num: Optional[int]
        ps_range: Optional[int]
        chroma: bool
        device_id: int
        fast: bool
        extractor_exp: int

    basic_args: BasicArgs | VBasicArgs
    final_args: FinalArgs | VFinalArgs


class BM3DCudaRTC(_AbstractBM3DCuda):
    plugin = core.bm3dcuda_rtc

    class BasicArgs(TypedDict, total=False):
        block_step: Optional[int]
        bm_range: Optional[int]
        chroma: bool
        device_id: int
        fast: bool
        extractor_exp: int
        bm_error_s: DataArray
        transform_2d_s: DataArray
        transform_1d_s: DataArray

    class FinalArgs(TypedDict, total=False):
        block_step: Optional[int]
        bm_range: Optional[int]
        chroma: bool
        device_id: int
        fast: bool
        extractor_exp: int
        bm_error_s: DataArray
        transform_2d_s: DataArray
        transform_1d_s: DataArray

    class VBasicArgs(TypedDict, total=False):
        block_step: Optional[int]
        bm_range: Optional[int]
        ps_num: Optional[int]
        ps_range: Optional[int]
        chroma: bool
        device_id: int
        fast: bool
        extractor_exp: int
        bm_error_s: DataArray
        transform_2d_s: DataArray
        transform_1d_s: DataArray

    class VFinalArgs(TypedDict, total=False):
        block_step: Optional[int]
        bm_range: Optional[int]
        ps_num: Optional[int]
        ps_range: Optional[int]
        chroma: bool
        device_id: int
        fast: bool
        extractor_exp: int
        bm_error_s: DataArray
        transform_2d_s: DataArray
        transform_1d_s: DataArray

    basic_args: BasicArgs | VBasicArgs
    final_args: FinalArgs | VFinalArgs


class BM3DCPU(_AbstractBM3DCuda):
    plugin = core.bm3dcpu

    class BasicArgs(TypedDict, total=False):
        block_step: Optional[int]
        bm_range: Optional[int]
        chroma: bool

    class FinalArgs(TypedDict, total=False):
        block_step: Optional[int]
        bm_range: Optional[int]
        chroma: bool

    class VBasicArgs(TypedDict, total=False):
        block_step: Optional[int]
        bm_range: Optional[int]
        ps_num: Optional[int]
        ps_range: Optional[int]
        chroma: bool

    class VFinalArgs(TypedDict, total=False):
        block_step: Optional[int]
        bm_range: Optional[int]
        ps_num: Optional[int]
        ps_range: Optional[int]
        chroma: bool

    basic_args: BasicArgs | VBasicArgs
    final_args: FinalArgs | VFinalArgs
