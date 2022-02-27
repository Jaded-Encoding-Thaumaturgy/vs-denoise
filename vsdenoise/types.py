from typing import Literal, Optional, Protocol, Sequence, Union

from vapoursynth import VideoNode

MATRIX = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

Data = Union[str, bytes, bytearray]
DataArray = Union[Data, Sequence[Data]]


class _PluginBm3dcudaCoreUnbound(Protocol):
    def BM3D(
        self,
        clip: VideoNode, ref: Optional[VideoNode] = None,
        sigma: Union[float, Sequence[float], None] = None,
        block_step: Union[int, Sequence[int], None] = None, bm_range: Union[int, Sequence[int], None] = None,
        radius: Optional[int] = None,
        ps_num: Union[int, Sequence[int], None] = None, ps_range: Union[int, Sequence[int], None] = None,
        chroma: Optional[int] = None,
        device_id: Optional[int] = None,
        fast: Optional[int] = None, extractor_exp: Optional[int] = None
    ) -> VideoNode:
        ...


class _PluginBm3dcuda_rtcCoreUnbound(Protocol):
    def BM3D(
        self,
        clip: VideoNode, ref: Optional[VideoNode] = None,
        sigma: Union[float, Sequence[float], None] = None,
        block_step: Union[int, Sequence[int], None] = None, bm_range: Union[int, Sequence[int], None] = None,
        radius: Optional[int] = None,
        ps_num: Union[int, Sequence[int], None] = None, ps_range: Union[int, Sequence[int], None] = None,
        chroma: Optional[int] = None,
        device_id: Optional[int] = None,
        fast: Optional[int] = None, extractor_exp: Optional[int] = None,
        bm_error_s: Optional[DataArray] = None,
        transform_2d_s: Optional[DataArray] = None,
        transform_1d_s: Optional[DataArray] = None
    ) -> VideoNode:
        ...


class _PluginBm3dcpuCoreUnbound(Protocol):
    def BM3D(
        self,
        clip: VideoNode, ref: Optional[VideoNode] = None,
        sigma: Union[float, Sequence[float], None] = None,
        block_step: Union[int, Sequence[int], None] = None, bm_range: Union[int, Sequence[int], None] = None,
        radius: Optional[int] = None,
        ps_num: Optional[int] = None, ps_range: Optional[int] = None,
        chroma: Optional[int] = None
    ) -> VideoNode:
        ...


class ZResizer(Protocol):
    def __call__(self, clip: VideoNode, *, format: Optional[int] = ..., matrix: Optional[int] = ...,
                 dither_type: Optional[Data] = ...) -> VideoNode:
        ...
