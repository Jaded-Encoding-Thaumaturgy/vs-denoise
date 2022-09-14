from enum import IntEnum
from typing import Optional, Protocol, Sequence, Union, Any

from vapoursynth import VideoNode

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


class LambdaVSFunction(Protocol):
    def __call__(self, clip: VideoNode, *args: Any, **kwargs: Any) -> VideoNode:
        ...


class SourceType(IntEnum):
    BFF = 0
    TFF = 1
    PROGRESSIVE = 2

    @property
    def is_inter(self) -> bool:
        return self != SourceType.PROGRESSIVE

    def __eq__(self, o: Any) -> bool:
        if not isinstance(o, SourceType):
            raise NotImplementedError

        return self.value == o.value

    def __ne__(self, o: Any) -> bool:
        return not (self == o)
