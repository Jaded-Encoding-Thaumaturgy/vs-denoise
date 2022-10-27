from __future__ import annotations

from typing import Optional, Protocol, Sequence, Union

from vapoursynth import VideoNode
from vstools import SimpleByteDataArray


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
        bm_error_s: Optional[SimpleByteDataArray] = None,
        transform_2d_s: Optional[SimpleByteDataArray] = None,
        transform_1d_s: Optional[SimpleByteDataArray] = None
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
