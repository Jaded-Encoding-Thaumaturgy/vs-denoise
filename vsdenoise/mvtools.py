"""
This module implements wrappers for mvtool
"""

from __future__ import annotations

from enum import Enum, IntEnum, auto
from itertools import chain
from math import ceil, exp
from typing import Any, Callable, Dict, List, Sequence, Tuple, cast

import vapoursynth as vs
from vsutil import Range as CRange
from vsutil import depth, disallow_variable_format, disallow_variable_resolution, fallback

from vsdenoise.utils import check_ref_clip

from .prefilters import Prefilter, prefilter_to_full_range
from .types import LambdaVSFunction
from .utils import planes_to_mvtools

__all__ = ['MVTools', 'SourceType', 'PelType', 'Prefilter']

core = vs.core
blackman_args = dict[str, Any](filter_param_a=-0.6, filter_param_b=0.4)


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


class PelType(IntEnum):
    AUTO = auto()
    NONE = auto()
    BICUBIC = auto()
    WIENER = auto()
    NNEDI3 = auto()


class MVToolPlugin(Enum):
    INTEGER = 0
    FLOAT_OLD = 1
    FLOAT_NEW = 2

    @property
    def namespace(self) -> Any:
        if self == MVToolPlugin.INTEGER:
            return core.mv
        else:
            return core.mvsf

    @property
    def Super(self) -> Callable[..., vs.VideoNode]:
        return cast(Callable[..., vs.VideoNode], self.namespace.Super)

    @property
    def Analyse(self) -> Callable[..., vs.VideoNode]:
        if self == MVToolPlugin.FLOAT_NEW:
            return cast(Callable[..., vs.VideoNode], self.namespace.Analyze)
        else:
            return cast(Callable[..., vs.VideoNode], self.namespace.Analyse)

    @property
    def Recalculate(self) -> Callable[..., vs.VideoNode]:
        return cast(Callable[..., vs.VideoNode], self.namespace.Recalculate)

    @property
    def Compensate(self) -> Callable[..., vs.VideoNode]:
        return cast(Callable[..., vs.VideoNode], self.namespace.Compensate)

    @property
    def Mask(self) -> Callable[..., vs.VideoNode]:
        return cast(Callable[..., vs.VideoNode], self.namespace.Mask)

    def Degrain(self, radius: int | None = None) -> Callable[..., vs.VideoNode]:
        if radius is None and self != MVToolPlugin.FLOAT_NEW:
            raise ValueError(f"{self.name}.Degrain needs radius")

        try:
            return cast(Callable[..., vs.VideoNode], getattr(
                self.namespace, f"Degrain{fallback(radius, '')}"
            ))
        except AttributeError:
            raise ValueError(f"{self.name}.Degrain doesn't support a radius of {radius}")

    def __eq__(self, o: Any) -> bool:
        if not isinstance(o, MVToolPlugin):
            raise NotImplementedError

        return self.value == o.value


class MVTools:
    """MVTools wrapper for motion analysis / degrain / compensation"""
    super_args: Dict[str, Any]
    analyze_args: Dict[str, Any]
    recalculate_args: Dict[str, Any]
    compensate_args: Dict[str, Any]
    degrain_args: Dict[str, Any]

    vectors: Dict[str, Any]

    clip: vs.VideoNode

    is_hd: bool
    is_uhd: bool
    tr: int
    refine: int
    source_type: SourceType
    prefilter: Prefilter | vs.VideoNode
    pel_type: Tuple[PelType, PelType]
    range_in: CRange
    pel: int
    subpixel: int
    chroma: bool
    is_gray: bool
    planes: List[int]
    mv_plane: int
    range_conversion: float
    hpad: int
    hpad_uhd: int
    vpad: int
    vpad_half: int
    rfilter: int
    mvtools: MVToolPlugin

    @disallow_variable_format
    @disallow_variable_resolution
    def __init__(
        self, clip: vs.VideoNode,
        tr: int = 2, refine: int = 3,
        source_type: SourceType = SourceType.PROGRESSIVE,
        prefilter: Prefilter | vs.VideoNode = Prefilter.AUTO,
        pel_type: PelType | Tuple[PelType, PelType] = PelType.AUTO,
        range_in: CRange = CRange.LIMITED,
        pel: int | None = None, subpixel: int = 3,
        planes: int | Sequence[int] | None = None,
        highprecision: bool = False,
        fix_fades: bool = False, range_conversion: float = 5.0,
        hpad: int | None = None, vpad: int | None = None,
        rfilter: int = 3, vectors: Dict[str, Any] | MVTools | None = None
    ) -> None:
        assert clip.format

        if clip.format.color_family not in {vs.GRAY, vs.YUV}:
            raise ValueError("MVTools: Only GRAY or YUV format clips supported")
        self.clip = clip

        self.is_hd = clip.width >= 1100 or clip.height >= 600
        self.is_uhd = self.clip.width >= 2600 or self.clip.height >= 1500

        self.tr = tr

        if refine > 6:
            raise ValueError("refine > 6 is not supported")
        self.refine = refine

        self.source_type = source_type
        self.prefilter = prefilter
        self.pel_type = pel_type if isinstance(pel_type, tuple) else (pel_type, pel_type)
        self.range_in = range_in
        self.pel = fallback(pel, 1 + int(not self.is_hd))
        self.subpixel = subpixel

        if planes is not None and isinstance(planes, int):
            planes = [planes]

        if clip.format.color_family == vs.GRAY:
            planes = [0]
        elif planes is None:
            planes = [0, 1, 2]

        self.is_gray = planes == [0]

        self.planes, self.mv_plane = planes_to_mvtools(planes)

        self.chroma = 1 in self.planes or 2 in self.planes

        self.range_conversion = range_conversion

        if isinstance(vectors, MVTools):
            self.vectors = vectors.vectors
        elif vectors:
            self.vectors = cast(Dict[str, Any], vectors)
        else:
            self.vectors = {}

        self.super_args = {}
        self.analyze_args = {}
        self.recalculate_args = {}
        self.compensate_args = {}
        self.degrain_args = {}

        self.hpad = fallback(hpad, 8 if self.is_hd else 16)
        self.hpad_uhd = self.hpad // 2 if self.is_uhd else self.hpad

        self.vpad = fallback(vpad, 8 if self.is_hd else 16)
        self.vpad_half = self.vpad // 2 if self.is_uhd else self.vpad

        self.rfilter = rfilter

        self.DCT = 5 if fix_fades else 0

        if self.source_type == SourceType.PROGRESSIVE:
            self.workclip = self.clip
        else:
            self.workclip = self.clip.std.SeparateFields(int(self.source_type))

        fmt = self.workclip.format
        assert fmt

        if highprecision or fmt.bits_per_sample == 32 or fmt.sample_type == vs.FLOAT or refine == 6 or tr > 3:
            self.workclip = depth(self.workclip, 32)
            self.mvtools = MVToolPlugin.FLOAT_NEW
            if not hasattr(core, 'mvsf'):
                raise ImportError(
                    "MVTools: With the current settings, the processing has to be done in float precision, "
                    "but you're missing mvsf."
                    "\n\tPlease download it from: https://github.com/IFeelBloated/vapoursynth-mvtools-sf"
                )
            if not hasattr(core.mvsf, 'Degrain'):
                if tr > 24:
                    raise ImportError(
                        "MVTools: With the current settings, (temporal radius > 24) you're gonna need the latest "
                        "master of mvsf and you're using an older version."
                        "\n\tPlease build it from: https://github.com/IFeelBloated/vapoursynth-mvtools-sf"
                    )
                self.mvtools = MVToolPlugin.FLOAT_OLD
        else:
            if not hasattr(core, 'mv'):
                raise ImportError(
                    "MVTools: You're missing mvtools."
                    "\n\tPlease download it from: https://github.com/dubhater/vapoursynth-mvtools"
                )
            self.mvtools = MVToolPlugin.INTEGER

        if not isinstance(prefilter, Prefilter):
            check_ref_clip(self.workclip, prefilter)

    def analyze(
        self, ref: vs.VideoNode | None = None,
        blksize: int | None = None, overlap: int | None = None,
        search: int | None = None, pelsearch: int | None = None,
        searchparam: int | None = None, truemotion: bool | None = None
    ) -> None:
        ref = fallback(ref, self.workclip)

        check_ref_clip(self.workclip, ref)

        truemotion = fallback(truemotion, not self.is_hd)

        searchparam = fallback(
            searchparam, (2 if self.is_uhd else 5) if (
                self.refine and truemotion
            ) else (1 if self.is_uhd else 2)
        )

        searchparamr = max(0, round(exp(0.69 * searchparam - 1.79) - 0.67))

        pelsearch = fallback(pelsearch, max(0, searchparam * 2 - 2))

        blocksize = max(
            self.refine and 2 ** (self.refine + 2),
            fallback(blksize, 16 if self.is_hd else 8)
        )

        halfblocksize = max(8, blocksize // 2)
        halfoverlap = max(2, halfblocksize // 2)

        overlap = fallback(overlap, halfblocksize)

        search = fallback(search, 4 if self.refine else 2)

        if isinstance(self.prefilter, vs.VideoNode):
            pref = self.prefilter
        else:
            pref = self.prefilter(ref, self.planes)

            if self.range_in == CRange.LIMITED:
                pref = prefilter_to_full_range(pref, self.range_conversion, self.planes)

        pelclip, pelclip2 = self.get_subpel_clips(pref, ref)

        common_args = dict[str, Any](
            sharp=min(self.subpixel, 2), pel=self.pel,
            vpad=self.vpad_half, hpad=self.hpad_uhd,
            chroma=self.chroma
        ) | self.super_args
        super_render_args = common_args | dict(
            levels=1,
            hpad=self.hpad, vpad=self.vpad,
            chroma=not self.is_gray
        )

        if pelclip or pelclip2:
            common_args |= dict(pelclip=pelclip)
            super_render_args |= dict(pelclip=pelclip2)

        super_search = self.mvtools.Super(ref, **common_args, rfilter=self.rfilter)
        super_render = self.mvtools.Super(self.workclip, **super_render_args)
        super_recalculate = self.mvtools.Super(pref, **common_args, levels=1) if self.refine else super_render

        recalculate_SAD = round(exp(-101. / (150 * 0.83)) * 360)
        t2 = (self.tr * 2 if self.tr > 1 else self.tr) if self.source_type.is_inter else self.tr

        analyse_args = dict[str, Any](
            plevel=0, pglobal=11, pelsearch=pelsearch,
            blksize=blocksize, overlap=overlap, search=search,
            truemotion=truemotion, searchparam=searchparam,
            chroma=self.chroma, dct=self.DCT
        ) | self.analyze_args

        recalculate_args = dict[str, Any](
            search=0, dct=5, thsad=recalculate_SAD,
            blksize=halfblocksize, overlap=halfoverlap,
            truemotion=truemotion, searchparam=searchparamr,
            chroma=self.chroma
        ) | self.recalculate_args

        if self.mvtools == MVToolPlugin.FLOAT_NEW:
            vmulti = self.mvtools.Analyse(super_search, radius=t2, **analyse_args)

            if self.source_type.is_inter:
                vmulti = vmulti.std.SelectEvery(4, 2, 3)

            self.vectors['vmulti'] = vmulti

            for i in range(self.refine):
                recalculate_args.update(
                    blksize=blocksize / 2 ** i, overlap=blocksize / 2 ** (i + 1)
                )
                self.vectors['vmulti'] = self.mvtools.Recalculate(
                    super_recalculate, self.vectors['vmulti'], **recalculate_args
                )
        else:
            def _add_vector(delta: int, recalculate: bool = False) -> None:
                if recalculate:
                    vects = {
                        'b': self.mvtools.Recalculate(
                            super_recalculate, self.vectors[f'bv{delta}'], **recalculate_args
                        ),
                        'f': self.mvtools.Recalculate(
                            super_recalculate, self.vectors[f'fv{delta}'], **recalculate_args
                        )
                    }
                else:
                    vects = {
                        'b': self.mvtools.Analyse(super_search, isb=True, delta=delta, **analyse_args),
                        'f': self.mvtools.Analyse(super_search, isb=False, delta=delta, **analyse_args)
                    }

                for k, vect in vects.items():
                    self.vectors[f'{k}v{delta}'] = vect

            for i in range(1, self.tr + 1):
                _add_vector(i)

            if self.refine:
                refblks = blocksize
                for i in range(1, t2 + 1):
                    if not self.vectors[f'bv{i}'] or not self.vectors[f'fv{i}']:
                        continue

                    for j in range(1, self.refine):
                        val = (refblks / 2 ** j)
                        if val > 128:
                            refblks = 128
                        elif val < 4:
                            refblks = blocksize

                        recalculate_args.update(
                            blksize=refblks / 2 ** j, overlap=refblks / 2 ** (j + 1)
                        )

                        _add_vector(i, True)

        self.vectors['super_render'] = super_render

    def get_vectors_bf(self, func_name: str = '') -> Tuple[List[vs.VideoNode], List[vs.VideoNode]]:
        if not self.vectors:
            raise RuntimeError(
                f"MVTools{'.' if func_name else ''}{func_name}: you first need to analyze the clip!"
            )

        t2 = (self.tr * 2 if self.tr > 1 else self.tr) if self.source_type.is_inter else self.tr

        vectors_backward = list[vs.VideoNode]()
        vectors_forward = list[vs.VideoNode]()

        if self.mvtools == MVToolPlugin.FLOAT_NEW:
            vmulti = self.vectors['vmulti']

            for i in range(0, t2 * 2, 2):
                vectors_backward.append(vmulti.std.SelectEvery(t2 * 2, i))
                vectors_forward.append(vmulti.std.SelectEvery(t2 * 2, i + 1))
        else:
            it = 1 + int(self.source_type.is_inter)
            for i in range(it, t2 + 1, it):
                vectors_backward.append(self.vectors[f'bv{i}'])
                vectors_forward.append(self.vectors[f'fv{i}'])

        return (vectors_backward, vectors_forward)

    def compensate(
        self, func: LambdaVSFunction,
        ref: vs.VideoNode | None = None,
        thSAD: int = 150, **kwargs: Any
    ) -> vs.VideoNode:
        ref = fallback(ref, self.workclip)

        check_ref_clip(self.workclip, ref)

        vect_b, vect_f = self.get_vectors_bf('compensate')

        compensate_args = dict(
            super=self.vectors['super_render'], thsad=thSAD,
            tff=self.source_type.is_inter and self.source_type.value or None
        ) | self.compensate_args

        comp_back, comp_forw = tuple(
            map(
                lambda vect: self.mvtools.Compensate(ref, vectors=vect, **compensate_args), vectors
            ) for vectors in (reversed(vect_b), vect_f)
        )

        comp_clips = [*comp_forw, ref, *comp_back]
        n_clips = len(comp_clips)

        interleaved = core.std.Interleave(comp_clips)

        processed = func(interleaved, **kwargs)

        return processed.std.SelectEvery(cycle=n_clips, offsets=ceil(n_clips / 2))

    def degrain(
        self, ref: vs.VideoNode | None = None,
        thSAD: int = 300, thSADC: int | None = None,
        thSCD1: int | None = None, thSCD2: int = 130,
        limit: int | None = None, limitC: float | None = None
    ) -> vs.VideoNode:
        check_ref_clip(self.workclip, ref)

        limit = fallback(limit, 2 if self.is_uhd else 255)
        limitC = fallback(limitC, limit)

        thrSAD_luma = round(exp(-101. / (thSAD * 0.83)) * 360)
        thrSAD_chroma = fallback(thSADC, round(thSAD * 0.18875 * exp(2 * 0.693)))

        thrSCD_first = fallback(thSCD1, round(0.35 * thSAD + 260))
        thrSCD_second = fallback(thSCD2, 130)

        vect_b, vect_f = self.get_vectors_bf('degrain')

        # Finally, MDegrain

        degrain_args = dict[str, Any](
            thscd1=thrSCD_first, thscd2=thrSCD_second, plane=self.mv_plane
        ) | self.degrain_args

        if self.mvtools == MVToolPlugin.INTEGER:
            degrain_args.update(
                thsad=thrSAD_luma, thsadc=thrSAD_chroma,
                limit=limit, limitc=limitC
            )
        else:
            degrain_args.update(
                thsad=[thrSAD_luma, thrSAD_chroma, thrSAD_chroma],
                limit=[limit, limitC]
            )

            if self.mvtools == MVToolPlugin.FLOAT_NEW:
                degrain_args.update(thsad2=[thrSAD_luma / 2, thrSAD_chroma / 2])

        to_degrain = ref or self.workclip

        if self.mvtools == MVToolPlugin.FLOAT_NEW:
            output = self.mvtools.Degrain()(
                to_degrain, self.vectors['super_render'], self.vectors['vmulti'], **degrain_args
            )
        else:
            output = self.mvtools.Degrain(self.tr)(
                to_degrain, self.vectors['super_render'], *chain.from_iterable(zip(vect_b, vect_f)), **degrain_args
            )

        return output.std.DoubleWeave(self.source_type.value) if self.source_type.is_inter else output

    def subpel_clip(self, clip: vs.VideoNode, pel_type: PelType) -> vs.VideoNode | None:
        bicubic_args = dict[str, Any](width=clip.width * self.pel, height=(clip.height * self.pel))

        if pel_type == PelType.BICUBIC or pel_type == PelType.WIENER:
            if pel_type == PelType.WIENER:
                bicubic_args |= blackman_args
            return clip.resize.Bicubic(**bicubic_args)
        elif pel_type == PelType.NNEDI3:
            nnargs = dict[str, Any](nsize=0, nns=1, qual=1, pscrn=2)

            plugin: Any = core.znedi3 if hasattr(core, 'znedi3') else core.nnedi3

            nnedi3_cpu = plugin.nnedi3(
                plugin.nnedi3(clip.std.Transpose(), 0, True, **nnargs).std.Transpose(), 0, True, **nnargs
            )

            if hasattr(core, 'nnedi3cl'):
                upscale = core.std.Interleave([
                    nnedi3_cpu[::2], clip[1::2].nnedi3cl.NNEDI3CL(0, True, True, **nnargs)
                ])
            else:
                upscale = nnedi3_cpu

            return upscale.resize.Bicubic(src_top=.5, src_left=.5)

        return None

    def get_subpel_clips(
        self, pref: vs.VideoNode, ref: vs.VideoNode
    ) -> Tuple[vs.VideoNode | None, vs.VideoNode | None]:
        pel_types = list(self.pel_type)

        for i, val in enumerate(pel_types):
            if val != PelType.AUTO:
                continue

            if i == 0:
                if self.prefilter == Prefilter.NONE:
                    pel_types[i] = PelType.NONE
                else:
                    if self.subpixel == 4:
                        pel_types[i] = PelType.NNEDI3
                    else:
                        pel_types[i] = PelType.BICUBIC
            else:
                pel_types[i] = PelType.NNEDI3 if self.subpixel == 4 else PelType.WIENER

        pel_type, pel2_type = pel_types

        return self.subpel_clip(pref, pel_type), self.subpel_clip(ref, pel2_type)
