from __future__ import annotations

from pathlib import Path
from typing import Literal, SupportsFloat, cast

from vsexprtools import expr_func
from vskernels import Catrom, Kernel, KernelT, Point
from vstools import (
    CustomStrEnum, DependencyNotFoundError, DitherType, FrameRangeN, FrameRangesN, InvalidColorFamilyError,
    LengthMismatchError, Matrix, MatrixT, UnsupportedVideoFormatError, check_variable, core, depth, get_depth,
    get_nvidia_version, replace_ranges, vs
)

__all__ = [
    'dpir',
]


class _dpir(CustomStrEnum):
    DEBLOCK = 'deblock'
    DENOISE = 'denoise'

    def __call__(
        self, clip: vs.VideoNode, strength: SupportsFloat | vs.VideoNode | None = 10,
        matrix: MatrixT | None = None, cuda: bool | Literal['trt'] | None = None, i444: bool = False,
        tiles: int | tuple[int, int] | None = None, overlap: int | tuple[int, int] | None = 8,
        zones: list[tuple[FrameRangeN | FrameRangesN | None, SupportsFloat | vs.VideoNode | None]] | None = None,
        fp16: bool | None = None, num_streams: int = 1, device_id: int = 0, kernel: KernelT = Catrom
    ) -> vs.VideoNode:
        func = 'dpir'

        try:
            from vsmlrt import Backend, DPIRModel, backendT, calc_tilesize, inference, models_path  # type: ignore
        except ModuleNotFoundError as e:
            raise DependencyNotFoundError(func, e)

        assert check_variable(clip, func)

        kernel = Kernel.ensure_obj(kernel)

        bit_depth = get_depth(clip)
        is_rgb, is_gray = (clip.format.color_family is f for f in (vs.RGB, vs.GRAY))

        clip_32 = depth(clip, 32, dither_type=DitherType.ERROR_DIFFUSION)

        if self.value == 'deblock':
            model = DPIRModel.drunet_deblocking_grayscale if is_gray else DPIRModel.drunet_deblocking_color
        else:  # elif self.value == 'denoise':
            model = DPIRModel.drunet_color if not is_gray else DPIRModel.drunet_gray

        def _get_strength_clip(clip: vs.VideoNode, strength: SupportsFloat) -> vs.VideoNode:
            return clip.std.BlankClip(format=vs.GRAYS, color=float(strength) / 255, keep=True)

        if isinstance(strength, vs.VideoNode):
            str_clip: vs.VideoNode = strength  # type: ignore

            assert (fmt := str_clip.format)

            InvalidColorFamilyError.check(
                fmt, vs.GRAY, func, '"strength" must be of {correct} color family, not {wrong}!')

            if fmt.id == vs.GRAY8:
                str_clip = expr_func(str_clip, 'x 255 /', vs.GRAYS)
            elif fmt.id != vs.GRAYS:
                raise UnsupportedVideoFormatError('`strength` must be GRAY8 or GRAYS!', func)

            if str_clip.width != clip.width or str_clip.height != clip.height:
                str_clip = kernel.scale(str_clip, clip.width, clip.height)

            if str_clip.num_frames != clip.num_frames:
                raise LengthMismatchError(func, '`strength` must be of the same length as \'clip\'')

            strength = str_clip
        elif isinstance(strength, SupportsFloat):
            strength = float(strength)
        else:
            raise UnsupportedVideoFormatError('`strength` must be a float or a GRAYS clip', func)

        if not is_rgb:
            targ_matrix = Matrix.from_param(matrix) or Matrix.from_video(clip)
        else:
            targ_matrix = Matrix.RGB

        targ_format = clip.format.replace(subsampling_w=0, subsampling_h=0) if i444 else clip.format

        if is_rgb or is_gray:
            clip_rgb = clip_32
        else:
            clip_rgb = kernel.resample(clip_32, vs.RGBS, matrix_in=targ_matrix)

        clip_rgb = clip_rgb.std.Limiter()

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
                    rstr_clip = zstr  # type: ignore
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

        backend: backendT

        if cuda == 'trt':
            channels = 2 << (not is_gray)

            backend = Backend.TRT(
                (tile_w, tile_h), fp16=fp16, num_streams=num_streams, device_id=device_id, verbose=False
            )
            backend._channels = channels
        elif cuda:
            backend = Backend.ORT_CUDA(fp16=fp16, num_streams=num_streams, device_id=device_id, verbosity=False)
        else:
            backend = Backend.OV_CPU(fp16=fp16)

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
