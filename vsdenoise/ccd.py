"""
This module contains a CCD implementation
"""

from __future__ import annotations

from enum import IntEnum
from math import sin, sqrt
from typing import List, Any

import vapoursynth as vs
from vsexprtools import PlanesT, norm_expr_planes, normalise_planes
from vskernels import Matrix
from vsscale import ssim_downsample
from vsutil import EXPR_VARS, get_peak_value, join, split, plane

core = vs.core

__all__ = ['ccd', 'CCDMode', 'CCDPoints']


class CCDMode(IntEnum):
    CHROMA_ONLY = 0
    BICUBIC_CHROMA = 1
    BICUBIC_LUMA = 2
    NNEDI_BICUBIC = 3
    NNEDI_SSIM = 4


class CCDPoints(IntEnum):
    LOW = 11
    MEDIUM = 22
    HIGH = 44
    ALL = 63


def ccd(
    src: vs.VideoNode, thr: float = 4, tr: int = 0, ref: vs.VideoNode | None = None,
    mode: int | CCDMode | None = None, scale: float | None = None, matrix: int | Matrix | None = None,
    ref_points: int | CCDPoints | None = CCDPoints.LOW | CCDPoints.MEDIUM,
    i444: bool = False, planes: PlanesT = None, **ssim_kwargs: Any
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
    is_subsampled = src.format.subsampling_h or src.format.subsampling_w

    if mode is not None:
        if not is_subsampled:
            raise ValueError('ccd: Mode is available only for subsampled video!')
        elif mode not in list(CCDMode):
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

        l_d, m_d, h_d = round(scale * 4), round(scale * 8), round(scale * 12)

        low_points = {
            'F': (-l_d, -l_d), 'G': (+l_d, -l_d),
            'J': (-l_d, +l_d), 'K': (+l_d, +l_d),
        }

        med_points = {
            'Q': (-m_d, -m_d), 'R': (0, -m_d), 'S': (+m_d, -m_d),
            'T': (-m_d, 0), '                   U': (+m_d, 0),
            'V': (-m_d, +m_d), 'W': (0, +m_d), 'X': (+m_d, +m_d),
        }

        high_points = {
            'A': (-h_d, -h_d), 'B': (-l_d, -h_d), 'C': (+l_d, -h_d), 'D': (+h_d, -h_d),
            'E': (-h_d, -l_d), '                                      H': (+h_d, -l_d),
            'I': (-h_d, +l_d), '                                      L': (+h_d, +l_d),
            'M': (-h_d, +h_d), 'N': (-l_d, +h_d), 'O': (+l_d, +h_d), 'P': (+h_d, +h_d),
        }

        if ref_points == CCDPoints.ALL:
            expr_points = low_points | med_points | high_points
        elif ref_points == (CCDPoints.LOW | CCDPoints.HIGH):
            expr_points = low_points | high_points
        elif ref_points == (CCDPoints.MEDIUM | CCDPoints.HIGH):
            expr_points = med_points | high_points
        else:
            expr_points = low_points | med_points

        tr_nclips = tr * 2 + 1
        num_points = len(expr_points.keys())

        plusses_plane = '+ ' * (tr_nclips - 1)
        plusses_points = '+ ' * (num_points - 1)

        def _get_weight_expr(x: int, y: int, c: str, weight: float | None = None) -> str:
            scale_str = peak != 1 and f'{peak} / ' or ''
            weigth_str = weight is not None and f'{weight_b} *' or ''

            return f'{c}[{x},{y}] {c} - {scale_str} 2 pow {weigth_str}'

        expression = list[str]()

        for char, (x, y) in expr_points.items():
            char = char.strip()
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

        expression.append(f'{plusses_points} 1 + WQ!')

        for char, (x, y) in expr_points.items():
            expression.append(f'{char}@ {thrs} < x[{x},{y}] 0 ?')

        expression.append(f'{plusses_points} x + WQ@ /')

        return core.akarin.Expr(
            expr_clips, norm_expr_planes(src, ' '.join(expression), planes), src444_format.id, True, False
        )

    if not is_yuv:
        return expr(ref or src, src)

    if matrix is None:
        matrix = Matrix.from_video(src)

    divw, divh = 1 << src.format.subsampling_w, 1 << src.format.subsampling_h

    if mode == CCDMode.BICUBIC_LUMA or not is_subsampled:
        yuvw, yuvh = src_width, src_height
        src_left = 0.0
    else:
        yuvw, yuvh = src_width // divw, src_height // divh

        src_left = 0.25 - 0.25 * divw

    yuv = yuvref = None

    if not is_subsampled:
        yuv = src
        yuvref = ref
    elif mode in {CCDMode.NNEDI_BICUBIC, CCDMode.NNEDI_SSIM}:
        ref_clips: List[List[vs.VideoNode] | None] = [split(src), ref and split(ref) or None]

        src_left += 0.125 * divw

        yuv, yuvref = [
            join(planes[:1] + [
                p.nnedi3.nnedi3(1, 1, 0, 0, 3, 2).std.Transpose()
                .nnedi3.nnedi3(1, 1, 0, 0, 3, 2).std.Transpose()
                .resize.Bicubic(src_top=-0.25 * divh) if divw == divh != 1 else (
                    p.nnedi3.nnedi3(1, 1, 0, 0, 3, 2).resize.Bicubic(src_top=-0.125 * divw)
                    if divh != 1 else p.std.Transpose().nnedi3.nnedi3(1, 1, 0, 0, 3, 2)
                    .std.Transpose().resize.Bicubic(src_top=-0.125 * divh)
                ) for p in planes[1:]
            ]) if planes else None for planes in ref_clips
        ]
    else:
        yuv = src.resize.Bicubic(yuvw, yuvh, src444_format.id)
        yuvref = ref and ref.resize.Bicubic(yuvw, yuvh, src444_format.id)

    assert yuv and yuv.format

    rgb = yuv.resize.Point(
        format=yuv.format.replace(color_family=vs.RGB).id, matrix_in=matrix
    )

    denoised = expr(yuvref or yuv, rgb)

    down_format = src444_format

    if not i444:
        if mode == CCDMode.NNEDI_BICUBIC:
            down_format = src.format
        elif mode == CCDMode.NNEDI_SSIM:
            down_format = down_format.replace(
                sample_type=vs.FLOAT, bits_per_sample=32
            )

    denoised = denoised.resize.Bicubic(format=down_format.id, src_left=src_left)

    if not is_subsampled and 0 in planes:
        return denoised

    if mode == CCDMode.NNEDI_SSIM and not i444:
        u = ssim_downsample(plane(denoised, 1), yuvw, yuvh, **ssim_kwargs)
        v = ssim_downsample(plane(denoised, 2), yuvw, yuvh, **ssim_kwargs)

        denoised = core.std.ShufflePlanes([denoised, u, v], [0, 0, 0], vs.YUV)
    else:
        denoised = core.std.ShufflePlanes([src, denoised], [0, 1, 2], vs.YUV)

    return denoised if i444 else denoised.resize.Bicubic(format=src.format.id)
