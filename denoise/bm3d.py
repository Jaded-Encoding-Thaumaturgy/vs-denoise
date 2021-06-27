import sys
from enum import Enum
from typing import Optional, Union

import vapoursynth as vs

core = vs.core


class BM3DProfile(Enum):
    FAST = "fast"
    LC = "lc"
    NP = "np"
    HIGH = "high"
    VN = "vn"

    # should this be included in the initialisations? I thought no
    # because I didn't want to make it too busy, but now im not too sure
    @property
    def radius(self) -> int:
        return {"fast": 1, "lc": 2, "np": 3, "high": 4, "vn": 4}[self.name]

    def _set_vbm3d_args(self, radius: int, args: dict[str, int]) -> dict[str, int]:
        if radius:
            args.update({"fast": dict(bm_range=7), "np": dict(bm_range=12), "vn": dict(bm_range=12)}.get(self.name, {}))

    def basic_cuda_args(self, radius: int) -> dict[str, int]:
        args = {
            "fast": dict(block_step=8, bm_range=9, ps_range=4),
            "lc": dict(block_step=6, bm_range=9, ps_range=4),
            "np": dict(block_step=4, bm_range=16, ps_range=5),
            "high": dict(block_step=2, bm_range=16, ps_range=7),
            "vn": dict(block_step=4, bm_range=16, ps_range=5),
        }[self.name]
        self._set_vbm3d_args(radius, args)
        return args

    def final_cuda_args(self, radius: int) -> dict[str, int]:
        if self.name == "vn":
            print("WARNING: BM3DCUDA doesn't directly support the vn profile for the final estimate."
                  "Emulating nearest parameters", file=sys.stderr)
        args = {
            "fast": dict(block_step=7, bm_range=9, ps_range=5),
            "lc": dict(block_step=5, bm_range=9, ps_range=5),
            "np": dict(block_step=3, bm_range=16, ps_range=6),
            "high": dict(block_step=2, bm_range=16, ps_range=8),
            # original vn profile for final uses a block size of 11, where cuda only supports 8. vn used 11,
            # and 11-4 = an overlap of 7, meaning the closest I can really get with a block size of 8 is a step of 1.
            # still probably isn't ideal, a larger block size would be far better for noisy content.
            "vn": dict(block_step=1, bm_range=16, ps_range=6),
        }[self.name]
        self._set_vbm3d_args(radius, args)
        return args


def BM3D(clip: vs.VideoNode,
         sigma: Union[float, list[float], list[list[float]]] = 1.5,
         profile: Union[Optional[BM3DProfile], list[Optional[BM3DProfile]]] = BM3DProfile.LC,
         radius: Union[Optional[int], list[Optional[int]]] = None,
         refine: int = 1,
         pre: Optional[vs.VideoNode] = None,
         ref: Optional[vs.VideoNode] = None,
         matrix: str = "709",
         CUDA: Optional[Union[bool, list[bool]]] = None) -> vs.VideoNode:

    # TODO: Allow passing args to bm3d/-cuda - how to handle auto detection with bad params?
    # TODO: Figure out and handle individual plane denoising
    # TODO: Autodetect matrix / colour range
    # TODO: Write docstring
    # TODO: Support for _rtc and new BM3DCPU?
    # TODO: Actually test this :P

    src = clip

    def is_gray(clip: vs.VideoNode) -> bool:
        return clip.format.color_family == vs.GRAY

    if CUDA is None:
        CUDA = [hasattr(core, "bm3dcuda")] * 2
    elif isinstance(CUDA, bool):
        CUDA = [CUDA, CUDA]

    if not isinstance(sigma, list):
        sigma = [sigma]
    if not all(isinstance(elem, list) for elem in sigma):
        sigma = [sigma, sigma]
    sigma: list[list[float]] = [(s + [s[-1]] * 3)[:3] for s in sigma]
    for i in [0, 1]:
        # multiply luma sigmas by 0.8, if using BM3DCUDA. This seemed to give closer results to the original.
        if CUDA[i]:
            sigma[i] = [s * 0.8 for s in sigma[i]]

    if not isinstance(profile, tuple):
        profile = (profile, profile)

    if not isinstance(radius, list):
        radius = [radius, radius]
    for i, r in enumerate(radius):
        if r is None:
            radius[i] = profile[i].radius

    if not (sigma[0][1] or sigma[0][2] or sigma[1][1] or sigma[1][2]) and clip.format.color_family != vs.RGB:
        clip = core.std.ShufflePlanes(clip, [0], vs.GRAY)

    if CUDA[0] and pre is not None:
        print("WARNING: BM3DCUDA doesn't accept a pre for the basic estimate, ignoring", file=sys.stderr)

    def to_opp(clip: vs.VideoNode) -> vs.VideoNode:
        clip = core.resize.Bicubic(clip,
                                   format=vs.RGBS if not is_gray(clip) else vs.GRAYS,
                                   filter_param_a=0, filter_param_b=0.5,
                                   matrix_in_s=matrix)
        return core.bm3d.RGB2OPP(clip, 1) if not is_gray(clip) else clip

    clips = {k: to_opp(v) for k, v in dict(src=clip, pre=pre, ref=ref).items() if v is not None}

    if all(c not in clips.keys() for c in ["pre", "ref"]):
        clips["pre"] = clips["src"]

    cudaargs = [None, None]
    if CUDA[0]:
        cudaargs[0] = profile[0].basic_cuda_args(radius[0])
    if CUDA[1]:
        cudaargs[1] = profile[1].final_cuda_args(radius[1])

    if "ref" in clips.keys():
        basic = clips["ref"]
    else:
        if CUDA[0]:
            basic = core.bm3dcuda.BM3D(clips["src"], sigma=sigma[0], radius=radius[0], **cudaargs[0])
            # bm3dcuda seems to set the luma to garbage if it isnt processed
            if not sigma[0][0] and (sigma[0][1] or sigma[0][2]):
                basic = core.std.ShufflePlanes([clips["src"], basic], [0, 1, 2], src.format.color_family)
        if not CUDA[0]:
            basicargs = dict(input=clips["src"], ref=clips["pre"], profile=profile[0], sigma=sigma[0], matrix=100)
            if radius[0]:
                basic = core.bm3d.VBasic(radius=radius[0], **basicargs)
            else:
                basic = core.bm3d.Basic(**basicargs)
        if radius[0]:
            basic = core.bm3d.VAggregate(basic, radius[0], 1)

    final = basic
    for _ in range(refine):
        if CUDA[1]:
            final = core.bm3dcuda.BM3D(clips["src"], ref=final, sigma=sigma[1], radius=radius[1], **cudaargs[1])
            # as before, bm3dcuda seems to set the luma to garbage if it isnt processed
            if not sigma[1][0] and (sigma[1][1] or sigma[1][2]):
                final = core.std.ShufflePlanes([clips["src"], final], [0, 1, 2], src.format.color_family)
        if not CUDA[1]:
            finalargs = dict(input=clips["src"], ref=final, profile=profile[1], sigma=sigma[1], matrix=100)
            if radius[1]:
                final = core.bm3d.VFinal(**finalargs, radius=radius[1])
            else:
                final = core.bm3d.Final(**finalargs)
        if radius[1]:
            final = core.bm3d.VAggregate(final, radius[1], 1)

    out = core.bm3d.OPP2RGB(final, 1) if not is_gray(final) else final
    out = core.resize.Bicubic(out, format=clip.format, filter_param_a=0, filter_param_b=0.5, matrix_s=matrix)

    if src.format.num_planes == 3 and clip.format.color_family == vs.GRAY:
        out = core.std.ShufflePlanes([out, src], [0, 1, 2], src.format.color_family)

    return out
