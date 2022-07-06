"""
This module contains a CCD implementation
"""

from __future__ import annotations

from enum import IntEnum
from math import sin, sqrt
from vsrgtools.util import norm_expr_planes, PlanesT, normalise_planes
# from typing import Any

import vapoursynth as vs
from vsutil import EXPR_VARS, split, join, get_peak_value  # , plane

core = vs.core

__all__ = ['ccd', 'CCDMode']


class CCDMode(IntEnum):
    CHROMA_ONLY = 0
    BICUBIC_CHROMA = 1
    BICUBIC_LUMA = 2
    NNEDI_BICUBIC = 3
    # NNEDI_SSIM = 4


def ccd(
    src: vs.VideoNode, thr: float = 4, tr: int = 0, ref: vs.VideoNode | None = None,
    mode: CCDMode | None = None, scale: float | None = None, matrix: int | None = None,
    i444: bool = False, planes: PlanesT = None  # , **ssim_kwargs: Any
) -> vs.VideoNode:
    assert src.format

    if ref is not None:
        assert ref.format

        if src.format.id != ref.format.id or src.num_frames != ref.num_frames:
            raise ValueError('ccd: src and ref must have the same format and length!')

    if src.format.color_family == vs.GRAY:
        raise ValueError('ccd: GRAY format is not supported!')

    if tr < 0 or tr > 3:
        raise ValueError('ccd: Temporal radius must be between 0 and 3 (inclusive)!')
    elif tr > src.num_frames // 2:
        raise ValueError('ccd: Temporal radius must be less than half of the clip length!')

    is_yuv = src.format.color_family == vs.YUV
    is_subsampled = src.format.subsampling_h and src.format.subsampling_w

    if mode is not None:
        if not is_subsampled:
            raise ValueError('ccd: Mode is available only for subsampled video!')
        elif mode not in CCDMode:
            raise ValueError('ccd: Passed an invalid mode, use CCDMode (0~4).')
    else:
        mode = CCDMode.CHROMA_ONLY

    src_width, src_height = src.width, src.height
    src444_format = src.format.replace(subsampling_w=0, subsampling_h=0)

    if planes is None and mode in {CCDMode.CHROMA_ONLY, CCDMode.BICUBIC_CHROMA}:
        planes = [1, 2]

    planes = normalise_planes(src, planes)

    def expr(src: vs.VideoNode, rgb: vs.VideoNode) -> vs.VideoNode:
        nonlocal scale

        tr_nclips = tr * 2 + 1

        rgb_clips = [
            core.std.ShufflePlanes([rgb, rgb, rgb], [i, i, i], vs.RGB) for i in range(3)
        ]

        peak = get_peak_value(src, False)

        thrs = thr ** 2 / (255 * 3 * 255)

        expr_clips = [src, *rgb_clips]

        for i in range(1, tr + 1):
            for clip in rgb_clips:
                back_clip = clip[i:] + clip[-1] * i
                forw_clip = clip[0] * i + clip[:-i]

                expr_clips.extend([back_clip, forw_clip])

        if not scale or scale <= 0:
            scale = 1.0
        elif scale == 1:
            scale = src_height / 240

        x_d, y_d = round(scale * 4), round(scale * 12)

        expr_points = {
            'A': (-y_d, -y_d), 'B': (-x_d, -y_d), 'C': (+x_d, -y_d), 'D': (+y_d, -y_d),
            'E': (-y_d, -x_d), 'F': (-x_d, -x_d), 'G': (+x_d, -x_d), 'H': (+y_d, -x_d),
            'I': (-y_d, +x_d), 'J': (-x_d, +x_d), 'K': (+x_d, +x_d), 'L': (+y_d, +x_d),
            'M': (-y_d, +y_d), 'N': (-x_d, +y_d), 'O': (+x_d, +y_d), 'P': (+y_d, +y_d)
        }

        expression = list[str]()

        plusses_plane = '+ ' * (tr_nclips - 1)

        def _get_weight_expr(x: int, y: int, c: str, weight: float | None = None) -> str:
            scale_str = peak != 1 and f'{peak} / ' or ''
            weigth_str = weight is not None and f'{weight_b} *' or ''

            return f'{c}[{x},{y}] {c} - {scale_str} 2 pow {weigth_str}'

        for char, (x, y) in expr_points.items():
            rgb_expr = []

            for i, c in enumerate(EXPR_VARS[1:4], 1):
                rgb_expr.append(_get_weight_expr(x, y, c))

                if tr:
                    for j in range(0, tr):
                        offset = i + 3 + j * 6
                        bc, fc = EXPR_VARS[offset], EXPR_VARS[offset + 1]
                        weight_f, weight_b = sqrt((4 - j) / 8), sin((5 - j) / 8)
                        rgb_expr.append(_get_weight_expr(x, y, bc, weight_b))
                        rgb_expr.append(_get_weight_expr(x, y, fc, weight_f))

                    rgb_expr.append(f'{plusses_plane} {tr_nclips} /')

            expression.append(f"{' '.join(rgb_expr)} + + {char}!")

        for char in expr_points:
            expression.append(f'{char}@ {thrs} < 1 0 ?')

        expression.append('+ + + + + + + + + + + + + + + 1 + Q!')

        for char, (x, y) in expr_points.items():
            expression.append(f'{char}@ {thrs} < x[{x},{y}] 0 ?')

        expression.append('+ + + + + + + + + + + + + + + x + Q@ /')

        return core.akarin.Expr(
            expr_clips, norm_expr_planes(src, ' '.join(expression), planes), src444_format.id, True, False
        )

    if not is_yuv:
        return expr(ref or src, src)

    # if matrix is None:
    #     matrix = get_matrix(src)

    if mode == CCDMode.BICUBIC_LUMA:
        yuvw, yuvh = src_width, src_height
    else:
        divw, divh = 1 << src.format.subsampling_w, 1 << src.format.subsampling_h
        yuvw, yuvh = src_width // divw, src_height // divh

    yuv = yuvref = None

    if not is_subsampled:
        src_left = 0.0

        yuv = src
        yuvref = ref
    elif mode in {CCDMode.CHROMA_ONLY, CCDMode.BICUBIC_CHROMA, CCDMode.BICUBIC_LUMA}:
        src_left = -0.25

        yuv = src.resize.Bicubic(yuvw, yuvh, src444_format.id)
        yuvref = ref and ref.resize.Bicubic(yuvw, yuvh, src444_format.id)
    else:
        src_left = -0.5

        ref_clips = [split(src), ref and split(ref)]

        yuv, yuvref = [
            join(planes[:1] + [
                p.nnedi3.nnedi3(1, 1, 0, 0, 3, 2).std.Transpose()
                .nnedi3.nnedi3(1, 1, 0, 0, 3, 2).std.Transpose()
                .resize.Bicubic(src_top=-0.5) for p in planes[1:]
            ]) if planes else None for planes in ref_clips
        ]

    assert yuv and yuv.format

    rgb = yuv.resize.Point(
        format=yuv.format.replace(color_family=vs.RGB).id, matrix_in=matrix
    )

    denoised = expr(yuvref or yuv, rgb)

    down_format = src444_format

    if not i444:
        if mode == CCDMode.NNEDI_BICUBIC:
            down_format = src.format
        # elif mode == CCDMode.NNEDI_SSIM:
        #     down_format = down_format.replace(
        #         sample_type=vs.FLOAT, bits_per_sample=32
        #     )

    denoised = denoised.resize.Bicubic(format=down_format.id, matrix=matrix, src_left=src_left)

    if not is_subsampled and 0 in planes:
        return denoised

    # if mode == CCDMode.NNEDI_SSIM and not i444:
    #     u = ssim_downsample(plane(denoised, 1), yuvw, yuvh, **ssim_kwargs)
    #     v = ssim_downsample(plane(denoised, 2), yuvw, yuvh, **ssim_kwargs)

    #     denoised = core.std.ShufflePlanes([denoised, u, v], [0, 0, 0], vs.YUV)
    # else:
    denoised = core.std.ShufflePlanes([src, denoised], [0, 1, 2], vs.YUV)

    return denoised if i444 else denoised.resize.Point(format=src.format.id)
