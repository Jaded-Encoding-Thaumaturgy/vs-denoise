from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

from vsexprtools import ExprOp, ExprToken, norm_expr
from vskernels import Bilinear
from vsmasktools import EdgeDetectT, FDoGTCanny, range_mask
from vsrgtools import RemoveGrainMode, bilateral, box_blur, gauss_blur, removegrain
from vsrgtools.util import norm_rmode_planes
from vstools import (
    CustomIndexError, CustomIntEnum, FuncExceptT, InvalidColorFamilyError, KwargsT, PlanesT, check_ref_clip,
    check_variable, fallback, flatten_vnodes, get_y, normalize_planes, scale_8bit, scale_value, vs
)

from .fft import DFTTest, fft3d
from .knlm import nl_means

__all__ = [
    'decrease_size',

    'PostProcess'
]


def decrease_size(
    clip: vs.VideoNode, sigmaS: float = 10.0, sigmaR: float = 0.009,
    min_in: int = 180, max_in: int = 230, gamma: float = 1.0,
    mask: vs.VideoNode | tuple[float, float] | tuple[float, float, EdgeDetectT] = (0.0496, 0.125, FDoGTCanny),
    prefilter: bool | tuple[int, int] | float = True, planes: PlanesT = None, show_mask: bool = False, **kwargs: Any
) -> vs.VideoNode:
    assert check_variable(clip, decrease_size)

    if min_in > max_in:
        raise CustomIndexError('The blur min must be lower than max!', decrease_size, dict(min=min_in, max=max_in))

    InvalidColorFamilyError.check(clip, vs.YUV, decrease_size)

    planes = normalize_planes(clip, planes)

    pre = get_y(clip)

    if isinstance(mask, vs.VideoNode):
        InvalidColorFamilyError.check(mask, vs.GRAY, decrease_size)  # type: ignore
        check_ref_clip(pre, mask)  # type: ignore
    else:
        pm_min, pm_max, *emask = mask  # type: ignore

        if pm_min > pm_max:
            raise CustomIndexError('The mask min must be lower than max!', decrease_size, dict(min=pm_min, max=pm_max))

        pm_min, pm_max = scale_value(pm_min, 32, clip), scale_value(pm_max, 32, clip)

        yuv444 = Bilinear.resample(
            range_mask(clip, rad=3, radc=2), clip.format.replace(subsampling_h=0, subsampling_w=0)
        )

        mask = FDoGTCanny.ensure_obj(emask[0] if emask else None).edgemask(pre)
        mask = mask.std.Maximum().std.Minimum()

        mask_planes = flatten_vnodes(yuv444, mask, split_planes=True)

        mask = norm_expr(
            mask_planes, f'x y max z max {pm_min} < 0 {ExprToken.RangeMax} ? a max {pm_max} < 0 {ExprToken.RangeMax} ?'
        )

        mask = box_blur(mask, 1, 2)

    if prefilter is True:
        prefilter = (2, 4)

    if prefilter:
        if isinstance(prefilter, tuple):
            pre = box_blur(pre, *prefilter)
        else:
            pre = gauss_blur(pre, prefilter)

    minf, maxf = scale_8bit(pre, min_in), scale_8bit(pre, max_in)

    mask = norm_expr(
        [pre, mask],  # type: ignore
        f'x {ExprOp.clamp(minf, maxf)} {minf} - {maxf} {minf} - / {1 / gamma} '
        f'pow {ExprOp.clamp(0, 1)} {ExprToken.RangeMax} * y -', planes
    )

    if show_mask:
        return mask

    denoise = bilateral(clip, sigmaS, sigmaR, **kwargs)

    return clip.std.MaskedMerge(denoise, mask, planes)


@dataclass
class PostProcessConfig:
    mode: PostProcess
    kwargs: KwargsT

    _sigma: float | None = None
    _tr: int | None = None
    _block_size: int | None = None
    merge_strength: int = 0

    @property
    def sigma(self) -> float:
        sigma = fallback(self._sigma, 1.0)

        if self.mode is PostProcess.DFTTEST:
            return sigma * 4

        if self.mode is PostProcess.NL_MEANS:
            return sigma / 2

        return sigma

    @property
    def tr(self) -> int:
        if self.mode <= 0:
            return 0

        tr = fallback(self._tr, 1)

        if self.mode is PostProcess.DFTTEST:
            return min(tr, 3)

        if self.mode in {PostProcess.FFT3D_MED, PostProcess.FFT3D_HIGH}:
            return min(tr, 2)

        return tr

    @property
    def block_size(self) -> int:
        if self.mode is PostProcess.DFTTEST:
            from .fft import BackendInfo

            backend_info = BackendInfo.from_param(self.kwargs.pop('plugin', DFTTest.Backend.AUTO))

            if backend_info.resolved_backend.is_dfttest2:
                return 16

        return fallback(self._block_size, [0, 48, 32, 12, 0][self.mode.value])

    def __call__(self, clip: vs.VideoNode, planes: PlanesT = None, func: FuncExceptT | None = None) -> vs.VideoNode:
        func = func or self.__class__

        if self.mode is PostProcess.REPAIR:
            return removegrain(clip, norm_rmode_planes(clip, RemoveGrainMode.MINMAX_AROUND1, planes))

        if self.mode in {PostProcess.FFT3D_MED, PostProcess.FFT3D_HIGH}:
            return fft3d(clip, func, bw=self.block_size, bh=self.block_size, bt=self.tr * 2 + 1, **self.kwargs)

        if self.mode is PostProcess.DFTTEST:
            return DFTTest.denoise(
                clip, self.sigma, tr=self.tr, block_size=self.block_size,
                planes=planes, **(KwargsT(overlap=int(self.block_size * 9 / 12)) | self.kwargs)  # type: ignore
            )

        if self.mode is PostProcess.NL_MEANS:
            return nl_means(
                clip, self.sigma, self.tr, planes=planes, **(KwargsT(sr=2) | self.kwargs)  # type: ignore
            )

        return clip


class PostProcess(CustomIntEnum):
    REPAIR = 0
    FFT3D_HIGH = 1
    FFT3D_MED = 2
    DFTTEST = 3
    NL_MEANS = 4

    if TYPE_CHECKING:
        from .postprocess import PostProcess

        @overload
        def __call__(  # type: ignore
            self: Literal[PostProcess.REPAIR], *, merge_strength: int = 0
        ) -> PostProcessConfig:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[PostProcess.NL_MEANS], *, sigma: float = 1.0, tr: int | None = None,
            merge_strength: int = 0, **kwargs: Any
        ) -> PostProcessConfig:
            ...

        @overload
        def __call__(
            self, *, sigma: float = 1.0, tr: int | None = None, block_size: int | None = None,
            merge_strength: int = 0, **kwargs: Any
        ) -> PostProcessConfig:
            ...

        def __call__(
            self, *, sigma: float = 1.0, tr: int | None = None, block_size: int | None = None,
            merge_strength: int = 0, **kwargs: Any
        ) -> PostProcessConfig:
            ...
    else:
        def __call__(
            self, *, sigma: float = 1.0, tr: int | None = None, block_size: int | None = None,
            merge_strength: int = 0, **kwargs: Any
        ) -> PostProcessConfig:
            return PostProcessConfig(self, kwargs, sigma, tr, block_size, merge_strength)
