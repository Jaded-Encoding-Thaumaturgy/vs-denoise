from __future__ import annotations

from typing import Any

from vsexprtools import ExprOp, ExprToken, norm_expr
from vskernels import Bilinear
from vsmasktools import EdgeDetectT, FDoGTCanny, range_mask
from vsrgtools import bilateral, box_blur, gauss_blur
from vstools import (
    CustomIndexError, InvalidColorFamilyError, PlanesT, check_ref_clip, check_variable, flatten_vnodes, get_y,
    normalize_planes, scale_8bit, scale_value, vs
)

__all__ = [
    'decrease_size'
]


def decrease_size(
    clip: vs.VideoNode, sigmaS: float = 10.0, sigmaR: float = 0.009,
    min_in: int = 180, max_in: int = 230, gamma: float = 1.0,
    mask: vs.VideoNode | tuple[float, float] | tuple[float, float, EdgeDetectT] = (0.0496, 0.125, FDoGTCanny),
    prefilter: bool | tuple[int, int] | float = True, planes: PlanesT = None, show_mask: bool = False, **kwargs: Any
) -> vs.VideoNode:
    """
    Forcibly reduce the required bitrate to encode a clip by blurring away noise and grain
    in areas they won't be visible in.

    Grain and noise in really bright areas can be incredibly hard to spot for even experienced encoders,
    and will eat up a lot of extra bitrate. As this grain is invisible, there's little reason
    to go out of your way to better preserve it, and aq-modes like aq-mode 3 already incentivize
    the encoder to spend more bits in darker areas anyway.

    A gradient mask is used internally to prevent "hard edges" from forming on the boundaries of the mask.
    Additionally, an edgemask is used to prevent clearly-defined detail from being blurred away.

    :param clip:        Clip to process.
    :param sigmaS:      Sigma of Gaussian function to calculate spatial weight.
                        See the `vsrgtools.bilateral` documentation for more information.
                        Default: 10.0.
    :param sigmaR:      Sigma of Gaussian function to calculate range weight.
                        See the `vsrgtools.bilateral` documentation for more information.
                        Default: 0.009.
    :param min_in:      Starting pixel value for the gradient mask. Must be a value between 0–255.
                        Low values are not recommended, as this will start to blur a lot more detail.
                        Default: 180.
    :param max_in:      Ending pixel value for the gradient mask. Must be a value between 0–255.
                        This value must be greater than `min_in`. Any pixel values above this will be fully masked.
                        Default: 230.
    :param mask:        @PLACEHOLDER@
    :param prefilter:   Prefilter the clip prior to masked blurring.

                        If you pass a float, a gauss blur will be used with the value determining its sigma.
                        If you pass a tuple of floats, a box blur will be used.
                        The first value is the radii, and the second is the number of passes.
                        If you pass `True`, it defaults to `box_blur(2, 4)`.
                        Set `False` to disable.

                        Default: True.
    :param planes:      Planes to process. If None, all planes. Default: None.
    :param show_mask:   Return the gradient mask clip. Default: False.
    :param kwargs:      Additional keyword arguments to pass to bilateral.

    :return:            Clip with the brightest areas, as defined by the gradient mask, heavily blurred.

    :raises IndexError:                 `min_in` is greater than `max_in`.
    :raises InvalidColorFamilyError:    Input clip is not a YUV clip.
    :raises InvalidColorFamilyError:    A VideoNode is passed to `mask` and the clip is not a GRAY clip.
    """
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
