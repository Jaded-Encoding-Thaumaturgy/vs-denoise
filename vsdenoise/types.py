from __future__ import annotations

from typing import Protocol

from vapoursynth import VideoNode
from vstools import SimpleByteDataArray, SingleOrSeq, SingleOrSeqOpt


class _Plugin_bm3dcuda_Core_Bound(Protocol):
    def BM3D(
        self, clip: VideoNode, ref: VideoNode | None = None, sigma: SingleOrSeqOpt[float] = None,
        block_step: SingleOrSeqOpt[int] = None, bm_range: SingleOrSeqOpt[int] = None,
        radius: int | None = None, ps_num: SingleOrSeqOpt[int] = None,
        ps_range: SingleOrSeqOpt[int] = None, chroma: int | None = None, device_id: int | None = None,
        fast: int | None = None, extractor_exp: int | None = None, zero_init: int | None = None
    ) -> VideoNode:
        ...

    def BM3Dv2(
        self, clip: VideoNode, ref: VideoNode | None = None, sigma: SingleOrSeqOpt[float] = None,
        block_step: SingleOrSeqOpt[int] = None, bm_range: SingleOrSeqOpt[int] = None,
        radius: int | None = None, ps_num: SingleOrSeqOpt[int] = None,
        ps_range: SingleOrSeqOpt[int] = None, chroma: int | None = None, device_id: int | None = None,
        fast: int | None = None, extractor_exp: int | None = None, zero_init: int | None = None
    ) -> VideoNode:
        ...

    def VAggregate(self, clip: VideoNode, src: VideoNode, planes: SingleOrSeq[int]) -> VideoNode:
        ...


class _Plugin_bm3dcuda_rtc_Core_Bound(Protocol):
    def BM3D(
        self, clip: VideoNode, ref: VideoNode | None = None, sigma: SingleOrSeqOpt[float] = None,
        block_step: SingleOrSeqOpt[int] = None, bm_range: SingleOrSeqOpt[int] = None,
        radius: int | None = None, ps_num: SingleOrSeqOpt[int] = None,
        ps_range: SingleOrSeqOpt[int] = None, chroma: int | None = None, device_id: int | None = None,
        fast: int | None = None, extractor_exp: int | None = None,
        bm_error_s: SimpleByteDataArray | None = None,
        transform_2d_s: SimpleByteDataArray | None = None,
        transform_1d_s: SimpleByteDataArray | None = None, zero_init: int | None = None
    ) -> VideoNode:
        ...

    def BM3Dv2(
        self, clip: VideoNode, ref: VideoNode | None = None, sigma: SingleOrSeqOpt[float] = None,
        block_step: SingleOrSeqOpt[int] = None, bm_range: SingleOrSeqOpt[int] = None,
        radius: int | None = None, ps_num: SingleOrSeqOpt[int] = None,
        ps_range: SingleOrSeqOpt[int] = None, chroma: int | None = None, device_id: int | None = None,
        fast: int | None = None, extractor_exp: int | None = None,
        bm_error_s: SimpleByteDataArray | None = None,
        transform_2d_s: SimpleByteDataArray | None = None,
        transform_1d_s: SimpleByteDataArray | None = None, zero_init: int | None = None
    ) -> VideoNode:
        ...

    def VAggregate(self, clip: VideoNode, src: VideoNode, planes: SingleOrSeq[int]) -> VideoNode:
        ...


class _Plugin_bm3dcpu_Core_Bound(Protocol):
    def BM3D(
        self, clip: VideoNode, ref: VideoNode | None = None, sigma: SingleOrSeqOpt[float] = None,
        block_step: SingleOrSeqOpt[int] = None, bm_range: SingleOrSeqOpt[int] = None,
        radius: int | None = None, ps_num: int | None = None, ps_range: int | None = None,
        chroma: int | None = None, zero_init: int | None = None
    ) -> VideoNode:
        ...

    def BM3Dv2(
        self, clip: VideoNode, ref: VideoNode | None = None, sigma: SingleOrSeqOpt[float] = None,
        block_step: SingleOrSeqOpt[int] = None, bm_range: SingleOrSeqOpt[int] = None,
        radius: int | None = None, ps_num: int | None = None, ps_range: int | None = None,
        chroma: int | None = None, zero_init: int | None = None
    ) -> VideoNode:
        ...

    def VAggregate(self, clip: VideoNode, src: VideoNode, planes: SingleOrSeq[int]) -> VideoNode:
        ...
