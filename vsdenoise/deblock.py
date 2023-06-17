from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, SupportsFloat, cast

from vsexprtools import expr_func, norm_expr
from vskernels import Catrom, Kernel, KernelT, Point
from vsmasktools import FDoG, GenericMaskT, adg_mask, normalize_mask
from vsrgtools import gauss_blur
from vstools import (
    ColorRange, CustomStrEnum, DependencyNotFoundError, DitherType, FrameRangeN, FrameRangesN, InvalidColorFamilyError,
    KwargsT, LengthMismatchError, Matrix, MatrixT, UnsupportedVideoFormatError, check_variable, core, depth, fallback,
    get_depth, get_nvidia_version, get_y, join, replace_ranges, vs
)

__all__ = [
    'dpir', 'dpir_mask'
]


class _dpir(CustomStrEnum):
    DEBLOCK: _dpir = 'deblock'  # type: ignore
    DENOISE: _dpir = 'denoise'  # type: ignore

    def __call__(
        self, clip: vs.VideoNode, strength: SupportsFloat | vs.VideoNode | None | tuple[
            SupportsFloat | vs.VideoNode | None, SupportsFloat | vs.VideoNode | None
        ] = 10, matrix: MatrixT | None = None, cuda: bool | Literal['trt'] | None = None, i444: bool = False,
        tiles: int | tuple[int, int] | None = None, overlap: int | tuple[int, int] | None = 8,
        zones: list[tuple[FrameRangeN | FrameRangesN | None, SupportsFloat | vs.VideoNode | None]] | None = None,
        fp16: bool | None = None, num_streams: int | None = None, device_id: int = 0, kernel: KernelT = Catrom,
        **kwargs: Any
    ) -> vs.VideoNode:
        func = 'dpir'

        try:
            from vsmlrt import Backend, DPIRModel, backendT, calc_tilesize, inference, models_path  # type: ignore
        except ModuleNotFoundError as e:
            raise DependencyNotFoundError(func, e)

        assert check_variable(clip, func)

        if isinstance(strength, tuple):
            if clip.format.num_planes > 1:
                args = (matrix, cuda, i444, tiles, overlap, zones, fp16, num_streams, device_id, kernel)
                return join(dpir(get_y(clip), strength[0], *args), dpir(clip, strength[1], *args))
            strength = strength[0]

        kernel = Kernel.ensure_obj(kernel)

        if not strength:
            return kernel.resample(clip, clip.format.replace(subsampling_w=0, subsampling_h=0)) if i444 else clip

        bit_depth = get_depth(clip)
        is_rgb, is_gray = (clip.format.color_family is f for f in (vs.RGB, vs.GRAY))

        if self.value == 'deblock':
            model = DPIRModel.drunet_deblocking_grayscale if is_gray else DPIRModel.drunet_deblocking_color
        else:  # elif self.value == 'denoise':
            model = DPIRModel.drunet_color if not is_gray else DPIRModel.drunet_gray

        if None in {cuda, fp16}:
            try:
                info = cast(dict[str, int], core.trt.DeviceProperties(device_id))

                fp16_available = info['major'] >= 7
                trt_available = True
            except BaseException:
                fp16_available = False
                trt_available = False

        if cuda is None:
            cuda = 'trt' if trt_available else get_nvidia_version() is not None

        if fp16 is None:
            fp16 = fp16_available

        def _get_strength_clip(clip: vs.VideoNode, strength: SupportsFloat) -> vs.VideoNode:
            return clip.std.BlankClip(format=vs.GRAYH if fp16 else vs.GRAYS, color=float(strength) / 255, keep=True)

        def _norm_str_clip(str_clip: vs.VideoNode) -> vs.VideoNode:
            assert (fmt := str_clip.format)

            InvalidColorFamilyError.check(
                fmt, vs.GRAY, func, '"strength" must be of {correct} color family, not {wrong}!'
            )

            if fmt.id == vs.GRAY8:
                str_clip = expr_func(str_clip, 'x 255 /', vs.GRAYH if fp16 else vs.GRAYS)
            elif fmt.id not in {vs.GRAYH, vs.GRAYS}:
                raise UnsupportedVideoFormatError('`strength` must be GRAY8, GRAYH or GRAYS!', func)
            elif fp16 and fmt.id != vs.GRAYH:
                str_clip = depth(str_clip, 16, vs.FLOAT)

            if str_clip.width != clip.width or str_clip.height != clip.height:
                str_clip = kernel.scale(str_clip, clip.width, clip.height)  # type: ignore

            if str_clip.num_frames != clip.num_frames:
                raise LengthMismatchError(func, '`strength` must be of the same length as \'clip\'')

            return str_clip

        if isinstance(strength, vs.VideoNode):
            strength = _norm_str_clip(strength)  # type: ignore
        elif isinstance(strength, SupportsFloat):
            strength = float(strength)
        else:
            raise UnsupportedVideoFormatError('`strength` must be a float or a GRAYS clip', func)

        if not is_rgb:
            targ_matrix = Matrix.from_param(matrix) or Matrix.from_video(clip)
        else:
            targ_matrix = Matrix.RGB

        targ_format = clip.format.replace(subsampling_w=0, subsampling_h=0) if i444 else clip.format

        clip_upsample = depth(clip, 16 if fp16 else 32, vs.FLOAT, dither_type=DitherType.ERROR_DIFFUSION)

        if is_rgb or is_gray:
            clip_rgb = clip_upsample
        else:
            clip_rgb = kernel.resample(clip_upsample, vs.RGBH if fp16 else vs.RGBS, matrix_in=targ_matrix)

        try:
            clip_rgb = clip_rgb.std.Limiter()
        except vs.Error:
            clip_rgb = norm_expr(clip_rgb, 'x 0 1 clamp')

        if overlap is None:
            overlap_w = overlap_h = 0
        elif isinstance(overlap, int):
            overlap_w = overlap_h = overlap
        else:
            overlap_w, overlap_h = overlap

        multiple = 8

        mod_w, mod_h = clip_rgb.width % multiple, clip_rgb.height % multiple

        if to_pad := any({mod_w, mod_h}):
            d_width, d_height = clip_rgb.width + mod_w, clip_rgb.height + mod_h

            clip_rgb = Point(src_width=d_width, src_height=d_height).scale(
                clip_rgb, d_width, d_height, (-mod_h, -mod_w)
            )

            if isinstance(strength, vs.VideoNode):
                strength = Point(src_width=d_width, src_height=d_height).scale(
                    strength, d_width, d_height, (-mod_h, -mod_w)  # type: ignore
                )

        if isinstance(tiles, tuple):
            tilesize = tiles
            tiles = None
        else:
            tilesize = None

        (tile_w, tile_h), (overlap_w, overlap_h) = calc_tilesize(
            multiple=multiple,
            tiles=tiles, tilesize=tilesize,
            width=clip_rgb.width, height=clip_rgb.height,
            overlap_w=overlap_w, overlap_h=overlap_h
        )

        strength_clip = cast(
            vs.VideoNode, strength if isinstance(
                strength, vs.VideoNode
            ) else _get_strength_clip(clip_rgb, strength)  # type: ignore
        )

        no_dpir_zones = list[FrameRangeN]()

        zoned_strength_clip = strength_clip

        if zones:
            cache_strength_clips = dict[float, vs.VideoNode]()

            dpir_zones = dict[int | tuple[int | None, int | None], vs.VideoNode]()

            for ranges, zstr in zones:
                if not zstr:
                    if isinstance(ranges, list):
                        no_dpir_zones.extend(ranges)
                    else:
                        no_dpir_zones.append(ranges)  # type: ignore

                    continue

                rstr_clip: vs.VideoNode

                if isinstance(zstr, vs.VideoNode):
                    rstr_clip = _norm_str_clip(zstr)  # type: ignore
                else:
                    zstr = float(zstr)  # type: ignore

                    if zstr not in cache_strength_clips:
                        cache_strength_clips[zstr] = _get_strength_clip(clip_rgb, zstr)

                    rstr_clip = cache_strength_clips[zstr]

                lranges = ranges if isinstance(ranges, list) else [ranges]

                for rrange in lranges:
                    if rrange:
                        dpir_zones[rrange] = rstr_clip

            if len(dpir_zones) <= 2:
                for rrange, sclip in dpir_zones.items():
                    if to_pad:
                        sclip = Point(src_width=d_width, src_height=d_height).scale(
                            sclip, d_width, d_height, (-mod_h, -mod_w)
                        )
                    zoned_strength_clip = replace_ranges(zoned_strength_clip, sclip, rrange)
            else:
                dpir_ranges_zones = {
                    range(*(
                        (r, r + 1) if isinstance(r, int) else (r[0] or 0, r[1] + 1 if r[1] else clip.num_frames)
                    )): sclip for r, sclip in dpir_zones.items()
                }

                dpir_ranges_zones = {k: dpir_ranges_zones[k] for k in sorted(dpir_ranges_zones, key=lambda x: x.start)}
                dpir_ranges_keys = list(dpir_ranges_zones.keys())

                def _select_sclip(n: int) -> vs.VideoNode:
                    nonlocal dpir_ranges_zones, dpir_ranges_keys

                    for i, ranges in enumerate(dpir_ranges_keys):
                        if n in ranges:
                            if i > 0:
                                dpir_ranges_keys = dpir_ranges_keys[i:] + dpir_ranges_keys[:i]
                            return dpir_ranges_zones[ranges]

                    return strength_clip

                zoned_strength_clip = strength_clip.std.FrameEval(_select_sclip)

        backend: backendT
        bkwargs = kwargs | KwargsT(fp16=fp16, device_id=device_id)

        # All this will eventually be in vs-nn
        if cuda is None or trt_available:
            try:
                data: KwargsT = core.trt.DeviceProperties(device_id)  # type: ignore
                memory = data.get('total_global_memory', 0)
                def_num_streams = data.get('async_engine_count', 1)

                cuda = 'trt'

                bkwargs = KwargsT(
                    workspace=memory / (1 << 22) if memory else None,
                    use_cuda_graph=True, use_cublas=True, use_cudnn=True,
                    use_edge_mask_convolutions=True, use_jit_convolutions=True,
                    static_shape=True, heuristic=True, output_format=int(fp16),
                    tf32=not fp16, force_fp16=fp16, num_streams=def_num_streams
                ) | bkwargs

                streams_info = 'OK' if bkwargs['num_streams'] == def_num_streams else 'MISMATCH'

                core.log_message(
                    vs.MESSAGE_TYPE_DEBUG,
                    f'Selected [{data.get("name", b"<unknown>").decode("utf8")}] '
                    f'with {f"{(memory / (1 << 30))}GiB" if memory else "<unknown>"} of VRAM, '
                    f'num_streams={def_num_streams} ({streams_info})'
                )
            except Exception:
                cuda = get_nvidia_version() is not None

        if bkwargs.get('num_streams', None) is None:
            bkwargs.update(num_streams=fallback(num_streams, 1))

        if cuda is True:
            if hasattr(core, 'ort'):
                backend = Backend.ORT_CUDA(**bkwargs)
            else:
                backend = Backend.OV_GPU(**bkwargs)
        elif cuda is False:
            if hasattr(core, 'ncnn'):
                backend = Backend.NCNN_VK(**bkwargs)
            else:
                bkwargs.pop('device_id')

                if hasattr(core, 'ort'):
                    backend = Backend.ORT_CPU(**bkwargs)
                else:
                    backend = Backend.OV_CPU(**bkwargs)
        else:
            channels = 2 << (not is_gray)

            backend = Backend.TRT((tile_w, tile_h), **bkwargs)
            backend._channels = channels

        network_path = Path(models_path) / 'dpir' / f'{tuple(DPIRModel.__members__)[model]}.onnx'

        run_dpir = inference(
            [clip_rgb, zoned_strength_clip], str(network_path), (overlap_w, overlap_h), (tile_w, tile_h), backend
        )

        if no_dpir_zones:
            run_dpir = replace_ranges(run_dpir, clip_rgb, no_dpir_zones)

        if to_pad:
            run_dpir = run_dpir.std.Crop(0, mod_w, mod_h, 0)

        if is_rgb or is_gray:
            return depth(run_dpir, bit_depth)

        return kernel.resample(run_dpir, targ_format, targ_matrix)


dpir = _dpir.DEBLOCK


def dpir_mask(
    clip: vs.VideoNode, low: float = 5, high: float = 10, lines: float | None = None,
    luma_scaling: float = 12, linemask: GenericMaskT | bool = True, relative: bool = False
) -> vs.VideoNode:
    y = depth(get_y(clip), 32, range_out=ColorRange.FULL)

    if linemask is True:
        linemask = FDoG

    mask = adg_mask(y, luma_scaling, relative, func=dpir_mask)

    if relative:
        mask = gauss_blur(mask, 1.5)

    mask = norm_expr(mask, f'{high} 255 / x {low} 255 / * -')

    if linemask:
        lines = fallback(lines, high)
        linemask = normalize_mask(linemask, y)

        lines_clip = mask.std.BlankClip(color=lines / 255)

        mask = mask.std.MaskedMerge(lines_clip, linemask)

    return mask
