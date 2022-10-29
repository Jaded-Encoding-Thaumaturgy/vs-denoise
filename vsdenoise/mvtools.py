"""
This module implements wrappers for mvtool
"""

from __future__ import annotations

from itertools import chain
from math import ceil, exp
from typing import Any, Callable, Sequence, cast

from vstools import (
    ColorRange, FieldBased, FieldBasedT, GenericVSFunction, check_ref_clip, depth, disallow_variable_format,
    disallow_variable_resolution, fallback, vs, core, CustomIntEnum, CustomValueError, CustomNotImplementedError,
    InvalidColorFamilyError, check_variable, CustomOverflowError, CustomStrEnum
)

from .prefilters import PelType, Prefilter, prefilter_to_full_range
from .utils import planes_to_mvtools

__all__ = [
    'MVTools', 'MVToolsPlugin',
    'SADMode',
    'MVWay', 'MotionVectors'
]


class MVWay(CustomStrEnum):
    BACK = 'backward'
    FWRD = 'forward'

    @property
    def isb(self) -> bool:
        return self is MVWay.BACK


class MotionVectors:
    vmulti: vs.VideoNode
    super_render: vs.VideoNode

    temporal_vectors: dict[MVWay, dict[int, vs.VideoNode]]

    def __init__(self) -> None:
        self._init_vects()

    def _init_vects(self) -> None:
        self.temporal_vectors = {w: {} for w in MVWay}

    @property
    def got_vectors(self) -> bool:
        return bool(self.temporal_vectors[MVWay.BACK] and self.temporal_vectors[MVWay.FWRD])

    def got_mv(self, way: MVWay, delta: int) -> bool:
        return delta in self.temporal_vectors[way]

    def get_mv(self, way: MVWay, delta: int) -> vs.VideoNode:
        return self.temporal_vectors[way][delta]

    def set_mv(self, way: MVWay, delta: int, vect: vs.VideoNode) -> None:
        self.temporal_vectors[way][delta] = vect

    def clear(self) -> None:
        del self.vmulti
        del self.super_render
        self.temporal_vectors.clear()


class MVToolsPlugin(CustomIntEnum):
    INTEGER = 0
    FLOAT_OLD = 1
    FLOAT_NEW = 2

    @property
    def namespace(self) -> Any:
        if self == MVToolsPlugin.INTEGER:
            return core.mv
        else:
            return core.mvsf

    @property
    def Super(self) -> Callable[..., vs.VideoNode]:
        return cast(Callable[..., vs.VideoNode], self.namespace.Super)

    @property
    def Analyse(self) -> Callable[..., vs.VideoNode]:
        if self == MVToolsPlugin.FLOAT_NEW:
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
        if radius is None and self != MVToolsPlugin.FLOAT_NEW:
            raise CustomValueError('This implementation needs a radius!', f'{self.name}.Degrain')

        if radius is not None and radius > 24 and self is not MVToolsPlugin.FLOAT_NEW:
            raise ImportError(
                f"{self.name}.Degrain: With the current settings, temporal radius > 24, you're gonna need the latest "
                "master of mvsf and you're using an older version."
                "\n\tPlease build it from: https://github.com/IFeelBloated/vapoursynth-mvtools-sf"
            )

        try:
            return cast(Callable[..., vs.VideoNode], getattr(
                self.namespace, f"Degrain{fallback(radius, '')}"
            ))
        except AttributeError:
            raise CustomValueError(f'This radius isn\'t supported! ({radius})', f'{self.name}.Degrain')

    def __eq__(self, o: Any) -> bool:
        if not isinstance(o, MVToolsPlugin):
            raise CustomNotImplementedError

        return self.value == o.value

    @classmethod
    def from_video(cls, clip: vs.VideoNode) -> MVToolsPlugin:
        assert clip.format

        if clip.format.sample_type == vs.FLOAT:
            if not hasattr(core, 'mvsf'):
                raise ImportError(
                    "MVTools: With the current settings, the processing has to be done in float precision, "
                    "but you're missing mvsf."
                    "\n\tPlease download it from: https://github.com/IFeelBloated/vapoursynth-mvtools-sf"
                )

            if hasattr(core.mvsf, 'Degrain'):
                return MVToolsPlugin.FLOAT_NEW

            return MVToolsPlugin.FLOAT_OLD
        elif not hasattr(core, 'mv'):
            raise ImportError(
                "MVTools: You're missing mvtools."
                "\n\tPlease download it from: https://github.com/dubhater/vapoursynth-mvtools"
            )

        return MVToolsPlugin.INTEGER


class SADMode(CustomIntEnum):
    SAT = 0
    BLOCK = 1
    MIXED_SAT_DCT = 2
    ADAPTIVE_SAT_MIXED = 3
    ADAPTIVE_SAT_DCT = 4

    SATD = 5
    MIXED_SATD_DCT = 6
    ADAPTIVE_SATD_MIXED = 7
    ADAPTIVE_SATD_DCT = 8
    MIXED_SATEQSATD_DCT = 9
    ADAPTIVE_SATD_MAJLUMA = 10

    def is_satd(self) -> bool:
        return self >= SADMode.SATD


class MVTools:
    """MVTools wrapper for motion analysis / degrain / compensation"""
    super_args: dict[str, Any]
    analyze_args: dict[str, Any]
    recalculate_args: dict[str, Any]
    compensate_args: dict[str, Any]
    degrain_args: dict[str, Any]

    vectors: MotionVectors

    clip: vs.VideoNode

    is_hd: bool
    is_uhd: bool
    tr: int
    refine: int
    source_type: FieldBased
    prefilter: Prefilter | vs.VideoNode
    pel_type: tuple[PelType, PelType]
    range_in: ColorRange
    pel: int
    sharp: int
    chroma: bool
    is_gray: bool
    planes: list[int]
    mv_plane: int
    range_conversion: float
    hpad: int
    hpad_uhd: int
    vpad: int
    vpad_half: int
    rfilter: int
    mvtools: MVToolsPlugin

    @disallow_variable_format
    @disallow_variable_resolution
    def __init__(
        self, clip: vs.VideoNode,
        tr: int = 2, refine: int = 3,
        source_type: FieldBasedT | None = None,
        prefilter: Prefilter | vs.VideoNode = Prefilter.AUTO,
        pel_type: PelType | tuple[PelType, PelType] = PelType.AUTO,
        range_in: ColorRange = ColorRange.LIMITED,
        pel: int | None = None, sharp: int = 3,
        planes: int | Sequence[int] | None = None,
        highprecision: bool = False,
        sad_mode: SADMode | tuple[SADMode, SADMode] = SADMode.SATD,
        range_conversion: float = 5.0,
        hpad: int | None = None, vpad: int | None = None,
        rfilter: int = 3, vectors: MotionVectors | MVTools | None = None,
        **analyze_kwargs: Any
    ) -> None:
        assert check_variable(clip, self.__class__)

        InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), self.__class__)

        self.clip = clip

        self.is_hd = clip.width >= 1100 or clip.height >= 600
        self.is_uhd = self.clip.width >= 2600 or self.clip.height >= 1500

        self.tr = tr

        if refine > 6:
            raise CustomOverflowError(f'Refine > 6 is not supported! ({refine})', self.__class__)

        self.refine = refine

        self.source_type = FieldBased.from_param(source_type, MVTools) or FieldBased.from_video(self.clip)
        self.prefilter = prefilter
        self.pel_type = pel_type if isinstance(pel_type, tuple) else (pel_type, pel_type)
        self.range_in = range_in
        self.pel = fallback(pel, 1 + int(not self.is_hd))
        self.sharp = sharp

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
        elif isinstance(vectors, MotionVectors):
            self.vectors = vectors
        else:
            self.vectors = MotionVectors()

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

        if isinstance(sad_mode, tuple):
            if not sad_mode[1].is_satd:
                raise CustomValueError('The SADMode for recalculation must use SATD!', self.__class__)
            self.sad_mode, self.recalc_sad_mode = sad_mode
        else:
            self.sad_mode, self.recalc_sad_mode = sad_mode, SADMode.SATD

        if self.source_type is not FieldBased.PROGRESSIVE:
            self.workclip = self.clip.std.SeparateFields(self.source_type.is_tff)
        else:
            self.workclip = self.clip

        if highprecision:
            self.workclip = depth(self.workclip, 32)

        if not isinstance(prefilter, Prefilter):
            check_ref_clip(self.workclip, prefilter)

        self.mvtools = MVToolsPlugin.from_video(self.workclip)

        self.analyze_func_kwargs = analyze_kwargs

    def analyze(
        self, ref: vs.VideoNode | None = None,
        blksize: int | None = None, overlap: int | None = None,
        search: int | None = None, pelsearch: int | None = None,
        searchparam: int | None = None, truemotion: bool | None = None,
        *, inplace: bool = False
    ) -> MotionVectors:
        if self.analyze_func_kwargs:
            if blksize is None:
                blksize = self.analyze_func_kwargs.get('blksize', None)

            if overlap is None:
                overlap = self.analyze_func_kwargs.get('overlap', None)

            if search is None:
                search = self.analyze_func_kwargs.get('search', None)

            if pelsearch is None:
                pelsearch = self.analyze_func_kwargs.get('pelsearch', None)

            if searchparam is None:
                searchparam = self.analyze_func_kwargs.get('searchparam', None)

            if truemotion is None:
                truemotion = self.analyze_func_kwargs.get('truemotion', None)

        vectors = MotionVectors() if inplace else self.vectors

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

            if self.range_in == ColorRange.LIMITED:
                pref = prefilter_to_full_range(pref, self.range_conversion, self.planes)

        pelclip, pelclip2 = self.get_subpel_clips(pref, ref)

        common_args = dict[str, Any](
            sharp=self.sharp, pel=self.pel, vpad=self.vpad_half, hpad=self.hpad_uhd, chroma=self.chroma
        ) | self.super_args
        super_render_args = common_args | dict(levels=1, hpad=self.hpad, vpad=self.vpad, chroma=not self.is_gray)

        if pelclip or pelclip2:
            common_args |= dict(pelclip=pelclip)
            super_render_args |= dict(pelclip=pelclip2)

        super_search = self.mvtools.Super(ref, **(dict(rfilter=self.rfilter) | common_args))
        super_render = self.mvtools.Super(self.workclip, **super_render_args)
        super_recalculate = self.mvtools.Super(pref, **(dict(levels=1) | common_args)) if self.refine else super_render

        recalculate_SAD = round(exp(-101. / (150 * 0.83)) * 360)
        t2 = (self.tr * 2 if self.tr > 1 else self.tr) if self.source_type.is_inter else self.tr

        analyse_args = dict[str, Any](
            plevel=0, pglobal=11, pelsearch=pelsearch, blksize=blocksize, overlap=overlap, search=search,
            truemotion=truemotion, searchparam=searchparam, chroma=self.chroma, dct=self.sad_mode
        ) | self.analyze_args

        recalc_args = dict[str, Any](
            search=0, dct=5, thsad=recalculate_SAD, blksize=halfblocksize, overlap=halfoverlap,
            truemotion=truemotion, searchparam=searchparamr, chroma=self.chroma
        ) | self.recalculate_args

        if self.mvtools == MVToolsPlugin.FLOAT_NEW:
            vmulti = self.mvtools.Analyse(super_search, radius=t2, **analyse_args)

            if self.source_type.is_inter:
                vmulti = vmulti.std.SelectEvery(4, 2, 3)

            vectors.vmulti = vmulti

            for i in range(self.refine):
                recalc_args.update(blksize=blocksize / 2 ** i, overlap=blocksize / 2 ** (i + 1))
                vectors.vmulti = self.mvtools.Recalculate(super_recalculate, vectors.vmulti, **recalc_args)
        else:
            def _add_vector(delta: int, analyze: bool = True) -> None:
                for way in MVWay:
                    if analyze:
                        vect = self.mvtools.Analyse(super_search, isb=way.isb, delta=delta, **analyse_args)
                    else:
                        vect = self.mvtools.Recalculate(super_recalculate, vectors.get_mv(way, delta), **recalc_args)

                    vectors.set_mv(way, delta, vect)

            for i in range(1, self.tr + 1):
                _add_vector(i)

            if self.refine:
                refblks = blocksize
                for i in range(1, t2 + 1):
                    if not vectors.got_mv(MVWay.BACK, i) or not vectors.got_mv(MVWay.FWRD, i):
                        continue

                    for j in range(1, self.refine):
                        val = (refblks / 2 ** j)
                        if val > 128:
                            refblks = 128
                        elif val < 4:
                            refblks = blocksize

                        recalc_args.update(blksize=refblks / 2 ** j, overlap=refblks / 2 ** (j + 1))

                        _add_vector(i, False)

        vectors.super_render = super_render

        return vectors

    def get_vectors_bf(self, *, inplace: bool = False) -> tuple[list[vs.VideoNode], list[vs.VideoNode]]:
        vectors = self.vectors if self.vectors.got_vectors else self.analyze(inplace=inplace)

        t2 = (self.tr * 2 if self.tr > 1 else self.tr) if self.source_type.is_inter else self.tr

        vectors_backward = list[vs.VideoNode]()
        vectors_forward = list[vs.VideoNode]()

        if self.mvtools == MVToolsPlugin.FLOAT_NEW:
            vmulti = vectors.vmulti

            for i in range(0, t2 * 2, 2):
                vectors_backward.append(vmulti.std.SelectEvery(t2 * 2, i))
                vectors_forward.append(vmulti.std.SelectEvery(t2 * 2, i + 1))
        else:
            it = 1 + int(self.source_type.is_inter)
            for i in range(it, t2 + 1, it):
                vectors_backward.append(vectors.get_mv(MVWay.BACK, i))
                vectors_forward.append(vectors.get_mv(MVWay.FWRD, i))

        return (vectors_backward, vectors_forward)

    def compensate(
        self, func: GenericVSFunction,
        ref: vs.VideoNode | None = None,
        thSAD: int = 150, **kwargs: Any
    ) -> vs.VideoNode:
        ref = fallback(ref, self.workclip)

        check_ref_clip(self.workclip, ref)

        vect_b, vect_f = self.get_vectors_bf()

        compensate_args = dict(
            super=self.vectors.super_render, thsad=thSAD,
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

        vect_b, vect_f = self.get_vectors_bf()

        degrain_args = dict[str, Any](
            thscd1=thrSCD_first, thscd2=thrSCD_second, plane=self.mv_plane
        ) | self.degrain_args

        if self.mvtools == MVToolsPlugin.INTEGER:
            degrain_args.update(thsad=thrSAD_luma, thsadc=thrSAD_chroma, limit=limit, limitc=limitC)
        else:
            degrain_args.update(thsad=[thrSAD_luma, thrSAD_chroma, thrSAD_chroma], limit=[limit, limitC])

            if self.mvtools == MVToolsPlugin.FLOAT_NEW:
                degrain_args.update(thsad2=[thrSAD_luma / 2, thrSAD_chroma / 2])

        to_degrain = ref or self.workclip

        if self.mvtools == MVToolsPlugin.FLOAT_NEW:
            output = self.mvtools.Degrain()(to_degrain, self.vectors.super_render, self.vectors.vmulti, **degrain_args)
        else:
            output = self.mvtools.Degrain(self.tr)(
                to_degrain, self.vectors.super_render, *chain.from_iterable(zip(vect_b, vect_f)), **degrain_args
            )

        return output.std.DoubleWeave(self.source_type.value) if self.source_type.is_inter else output

    def get_subpel_clips(
        self, pref: vs.VideoNode, ref: vs.VideoNode
    ) -> tuple[vs.VideoNode | None, vs.VideoNode | None]:
        return tuple(  # type: ignore[return-value]
            None if ptype == PelType.NONE else ptype(
                clip, self.pel, default=PelType.WIENER if is_ref else PelType.BICUBIC
            ) for is_ref, ptype, clip in zip((False, True), self.pel_type, (pref, ref))
        )
