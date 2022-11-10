"""
This module implements wrappers for mvtool
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from math import ceil, exp
from typing import Any, Literal, Sequence, cast, overload

from vstools import (
    MISSING, ColorRange, ConstantFormatVideoNode, CustomIntEnum, CustomOverflowError, CustomStrEnum, CustomValueError,
    FieldBased, FieldBasedT, FuncExceptT, GenericVSFunction, InvalidColorFamilyError, MissingT, VSFunction,
    check_ref_clip, check_variable, core, depth, disallow_variable_format, disallow_variable_resolution, fallback,
    kwargs_fallback, normalize_planes, normalize_seq, vs
)

from .prefilters import PelType, Prefilter, prefilter_to_full_range
from .utils import planes_to_mvtools

__all__ = [
    'MVTools', 'MVToolsPlugin',
    'SADMode', 'SearchMode', 'MotionMode',
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
        self._init_vects()


class MVToolsPlugin(CustomIntEnum):
    INTEGER = 0
    FLOAT_OLD = 1
    FLOAT_NEW = 2

    @property
    def namespace(self) -> Any:
        return core.mv if self is MVToolsPlugin.INTEGER else core.mvsf

    @property
    def Super(self) -> VSFunction:
        return cast(VSFunction, self.namespace.Super)

    @property
    def Analyse(self) -> VSFunction:
        return cast(VSFunction, self.namespace.Analyze if self is MVToolsPlugin.FLOAT_NEW else self.namespace.Analyse)

    @property
    def Recalculate(self) -> VSFunction:
        return cast(VSFunction, self.namespace.Recalculate)

    @property
    def Compensate(self) -> VSFunction:
        return cast(VSFunction, self.namespace.Compensate)

    @property
    def Mask(self) -> VSFunction:
        return cast(VSFunction, self.namespace.Mask)

    def Degrain(self, radius: int | None = None) -> VSFunction:
        if radius is None and self is not MVToolsPlugin.FLOAT_NEW:
            raise CustomValueError('This implementation needs a radius!', f'{self.name}.Degrain')

        if radius is not None and radius > 24 and self is not MVToolsPlugin.FLOAT_NEW:
            raise ImportError(
                f"{self.name}.Degrain: With the current settings, temporal radius > 24, you're gonna need the latest "
                "master of mvsf and you're using an older version."
                "\n\tPlease build it from: https://github.com/IFeelBloated/vapoursynth-mvtools-sf"
            )

        try:
            return cast(VSFunction, getattr(self.namespace, f"Degrain{fallback(radius, '')}"))
        except AttributeError:
            raise CustomValueError(f'This radius isn\'t supported! ({radius})', f'{self.name}.Degrain')

    @classmethod
    def from_video(cls, clip: vs.VideoNode) -> MVToolsPlugin:
        assert clip.format

        if clip.format.sample_type is vs.FLOAT:
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


class MotionMode:
    @dataclass
    class Config:
        truemotion: bool
        coherence: int
        sad_limit: int
        pnew: int
        plevel: int
        global_motion: bool

    HIGH_SAD = Config(False, 0, 400, 0, 0, False)
    VECT_COHERENCE = Config(True, 1000, 1200, 50, 1, True)
    VECT_NOSCALING = Config(True, 1000, 1200, 50, 0, True)

    class _CustomConfig:
        def __call__(
            self, coherence: int | None = None, sad_limit: int | None = None,
            pnew: int | None = None, plevel: int | None = None, global_motion: bool | None = None,
            truemotion: bool = True
        ) -> MotionMode.Config:
            ref = MotionMode.from_param(truemotion)

            return MotionMode.Config(
                truemotion,
                fallback(coherence, ref.coherence),
                fallback(sad_limit, ref.sad_limit),
                fallback(pnew, ref.pnew),
                fallback(plevel, ref.plevel),
                fallback(global_motion, ref.global_motion)
            )

    MANUAL = _CustomConfig()

    @classmethod
    def from_param(cls, truemotion: bool) -> Config:
        return MotionMode.VECT_COHERENCE if truemotion else MotionMode.HIGH_SAD


class SearchModeBase:
    @dataclass
    class Config:
        mode: SearchMode
        param: int
        param_recalc: int
        pel: int


class SearchMode(SearchModeBase, CustomIntEnum):
    AUTO = -1
    ONETIME = 0
    NSTEP = 1
    DIAMOND = 2
    HEXAGON = 4
    UMH = 5
    EXHAUSTIVE = 3
    EXHAUSTIVE_H = 6
    EXHAUSTIVE_V = 7

    @overload
    def __call__(  # type: ignore
        self: Literal[ONETIME], step: int | tuple[int, int] = ..., pel: int = ..., /, **kwargs: Any
    ) -> SearchMode.Config:
        ...

    @overload
    def __call__(  # type: ignore
        self: Literal[NSTEP], times: int | tuple[int, int] = ..., pel: int = ..., /, **kwargs: Any
    ) -> SearchMode.Config:
        ...

    @overload
    def __call__(  # type: ignore
        self: Literal[DIAMOND], init_step: int | tuple[int, int] = ..., pel: int = ..., /, **kwargs: Any
    ) -> SearchMode.Config:
        ...

    @overload
    def __call__(  # type: ignore
        self: Literal[HEXAGON], range: int | tuple[int, int] = ..., pel: int = ..., /, **kwargs: Any
    ) -> SearchMode.Config:
        ...

    @overload
    def __call__(  # type: ignore
        self: Literal[UMH], range: int | tuple[int, int] = ..., pel: int = ..., /, **kwargs: Any
    ) -> SearchMode.Config:
        ...

    @overload
    def __call__(  # type: ignore
        self: Literal[EXHAUSTIVE] | Literal[EXHAUSTIVE_H] | Literal[EXHAUSTIVE_V],
        radius: int | tuple[int, int] = ..., pel: int = ..., /, **kwargs: Any
    ) -> SearchMode.Config:
        ...

    @overload
    def __call__(self, param: int | tuple[int, int] = ..., pel: int = ..., /, **kwargs: Any) -> SearchMode.Config:
        ...

    def __call__(
        self, param: int | tuple[int, int] | MissingT = MISSING, pel: int | MissingT = MISSING, /, **kwargs: Any
    ) -> SearchMode.Config:
        is_uhd = kwargs.get('is_uhd', False)
        refine = kwargs.get('refine', 3)
        truemotion = kwargs.get('truemotion', False)

        if self is SearchMode.AUTO:
            self = SearchMode.DIAMOND

        param_recalc: int | MissingT

        if isinstance(param, int):
            param, param_recalc = param, MISSING
        elif isinstance(param, tuple):
            param, param_recalc = param
        else:
            param = param_recalc = MISSING

        if param is MISSING:
            param = (2 if is_uhd else 5) if (refine and truemotion) else (1 if is_uhd else 2)

        if param_recalc is MISSING:
            param_recalc = max(0, round(exp(0.69 * param - 1.79) - 0.67))

        if pel is MISSING:
            pel = min(8, max(0, param * 2 - 2))

        return SearchMode.Config(self, param, param_recalc, pel)  # type: ignore


class MVTools:
    """MVTools wrapper for motion analysis / degrain / compensation"""

    super_args: dict[str, Any]
    analyze_args: dict[str, Any]
    recalculate_args: dict[str, Any]
    compensate_args: dict[str, Any]

    vectors: MotionVectors

    clip: vs.VideoNode

    @disallow_variable_format
    @disallow_variable_resolution
    def __init__(
        self, clip: vs.VideoNode,
        tr: int = 2, refine: int = 3, pel: int | None = None,
        planes: int | Sequence[int] | None = None,
        range_in: ColorRange = ColorRange.LIMITED,
        source_type: FieldBasedT | None = None,
        high_precision: bool = False,
        hpad: int | None = None, vpad: int | None = None,
        vectors: MotionVectors | MVTools | None = None,
        *,
        # kwargs for mvtools calls
        super_args: dict[str, Any] | None = None,
        analyze_args: dict[str, Any] | None = None,
        recalculate_args: dict[str, Any] | None = None,
        compensate_args: dict[str, Any] | None = None,
        # Analyze kwargs
        block_size: int | None = None,
        overlap: int | None = None,
        range_conversion: float | None = None,
        search: SearchMode | SearchMode.Config | None = None,
        sharp: int | None = None, rfilter: int | None = None,
        sad_mode: SADMode | tuple[SADMode, SADMode] | None = None,
        motion: MotionMode.Config | None = None,
        prefilter: Prefilter | vs.VideoNode | None = None,
        pel_type: PelType | tuple[PelType, PelType] | None = None
    ) -> None:
        assert check_variable(clip, self.__class__)

        InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), self.__class__)

        self.clip = clip

        self.is_hd = clip.width >= 1100 or clip.height >= 600
        self.is_uhd = self.clip.width >= 2600 or self.clip.height >= 1500

        self.tr = tr

        self.refine = refine

        if self.refine > 6:
            raise CustomOverflowError(f'Refine > 6 is not supported! ({refine})', self.__class__)

        self.source_type = FieldBased.from_param(source_type, MVTools) or FieldBased.from_video(self.clip)
        self.range_in = range_in
        self.pel = fallback(pel, 1 + int(not self.is_hd))

        planes = normalize_planes(self.clip, planes)

        self.is_gray = planes == [0]

        self.planes, self.mv_plane = planes_to_mvtools(planes)

        self.chroma = 1 in self.planes or 2 in self.planes

        if isinstance(vectors, MVTools):
            self.vectors = vectors.vectors
        elif isinstance(vectors, MotionVectors):
            self.vectors = vectors
        else:
            self.vectors = MotionVectors()

        self.super_args = fallback(super_args, dict[str, Any]())
        self.analyze_args = fallback(analyze_args, dict[str, Any]())
        self.recalculate_args = fallback(recalculate_args, dict[str, Any]())
        self.compensate_args = fallback(compensate_args, dict[str, Any]())

        self.hpad = fallback(hpad, 8 if self.is_hd else 16)
        self.hpad_uhd = self.hpad // 2 if self.is_uhd else self.hpad

        self.vpad = fallback(vpad, 8 if self.is_hd else 16)
        self.vpad_half = self.vpad // 2 if self.is_uhd else self.vpad

        if self.source_type is not FieldBased.PROGRESSIVE:
            self.workclip = self.clip.std.SeparateFields(self.source_type.is_tff)
        else:
            self.workclip = self.clip

        self.high_precision = high_precision

        if self.high_precision:
            self.workclip = depth(self.workclip, 32)

        self.mvtools = MVToolsPlugin.from_video(self.workclip)

        self.analyze_func_kwargs = dict(
            rfilter=rfilter, overlap=overlap, range_conversion=range_conversion, search=search, sharp=sharp,
            block_size=block_size, sad_mode=sad_mode, motion=motion, prefilter=prefilter, pel_type=pel_type
        )

    def analyze(
        self,
        block_size: int | None = None,
        overlap: int | None = None,
        range_conversion: float | None = None,
        search: SearchMode | SearchMode.Config | None = None,
        sharp: int | None = None, rfilter: int | None = None,
        sad_mode: SADMode | tuple[SADMode, SADMode] | None = None,
        motion: MotionMode.Config | None = None,
        prefilter: Prefilter | vs.VideoNode | None = None,
        pel_type: PelType | tuple[PelType, PelType] | None = None,
        *, ref: vs.VideoNode | None = None, inplace: bool = False
    ) -> MotionVectors:
        ref = self.get_ref_clip(ref, self.__class__.analyze)

        block_size = kwargs_fallback(block_size, (self.analyze_func_kwargs, 'block_size'), 16 if self.is_hd else 8)
        blocksize = max(self.refine and 2 ** (self.refine + 2), block_size)

        halfblocksize = max(8, blocksize // 2)
        halfoverlap = max(2, halfblocksize // 2)

        overlap = kwargs_fallback(overlap, (self.analyze_func_kwargs, 'overlap'), halfblocksize)

        rfilter = kwargs_fallback(rfilter, (self.analyze_func_kwargs, 'rfilter'), 3)

        range_conversion = kwargs_fallback(range_conversion, (self.analyze_func_kwargs, 'range_conversion'), 5.0)

        sharp = kwargs_fallback(sharp, (self.analyze_func_kwargs, 'sharp'), 2)

        search = kwargs_fallback(  # type: ignore[assignment]
            search, (self.analyze_func_kwargs, 'search'),
            SearchMode.HEXAGON if self.refine else SearchMode.DIAMOND
        )

        motion = kwargs_fallback(
            motion, (self.analyze_func_kwargs, 'motion'),
            MotionMode.VECT_NOSCALING if (
                ref.format.bits_per_sample == 32
            ) else MotionMode.from_param(not self.is_hd)
        )

        if isinstance(search, SearchMode):
            search = search(is_uhd=self.is_uhd, refine=self.refine, truemotion=motion.truemotion)

        assert search

        sad_mode = kwargs_fallback(  # type: ignore[assignment]
            sad_mode, (self.analyze_func_kwargs, 'sad_mode'), SADMode.SATD
        )

        prefilter = kwargs_fallback(  # type: ignore[assignment]
            prefilter, (self.analyze_func_kwargs, 'prefilter'), Prefilter.AUTO
        )

        pel_type = kwargs_fallback(  # type: ignore[assignment]
            pel_type, (self.analyze_func_kwargs, 'pel_type'), PelType.AUTO
        )

        if not isinstance(pel_type, tuple):
            pel_type = (pel_type, pel_type)  # type: ignore[assignment]

        vectors = MotionVectors() if inplace else self.vectors

        if isinstance(sad_mode, tuple):
            if not sad_mode[1].is_satd:
                raise CustomValueError('The SADMode for recalculation must use SATD!', self.__class__)
            sad_mode, recalc_sad_mode = sad_mode
        else:
            sad_mode, recalc_sad_mode = sad_mode, SADMode.SATD

        if isinstance(prefilter, Prefilter):
            prefilter = prefilter(ref, self.planes)

            if self.range_in.is_limited:
                prefilter = prefilter_to_full_range(prefilter, range_conversion, self.planes)

        assert prefilter is not None

        if self.high_precision:
            prefilter = depth(prefilter, 32)

        check_ref_clip(ref, prefilter)

        pelclip, pelclip2 = self.get_subpel_clips(prefilter, ref, pel_type)  # type: ignore[arg-type]

        common_args = dict[str, Any](
            sharp=sharp, pel=self.pel, vpad=self.vpad_half, hpad=self.hpad_uhd, chroma=self.chroma
        ) | self.super_args
        super_render_args = common_args | dict(levels=1, hpad=self.hpad, vpad=self.vpad, chroma=not self.is_gray)

        if pelclip or pelclip2:
            common_args |= dict(pelclip=pelclip)
            super_render_args |= dict(pelclip=pelclip2)

        super_search = self.mvtools.Super(ref, **(dict(rfilter=rfilter) | common_args))
        super_render = self.mvtools.Super(self.workclip, **super_render_args)
        super_recalculate = self.mvtools.Super(
            prefilter, **(dict(levels=1) | common_args)
        ) if self.refine else super_render

        recalculate_SAD = round(exp(-101. / (150 * 0.83)) * 360)
        t2 = (self.tr * 2 if self.tr > 1 else self.tr) if self.source_type.is_inter else self.tr

        analyse_args = dict[str, Any](
            dct=sad_mode, pelsearch=search.pel, blksize=blocksize, overlap=overlap, search=search.mode,
            truemotion=motion.truemotion, searchparam=search.param, chroma=self.chroma,
            plevel=0, pglobal=11
        ) | self.analyze_args

        recalc_args = dict[str, Any](
            search=0, dct=recalc_sad_mode, thsad=recalculate_SAD, blksize=halfblocksize, overlap=halfoverlap,
            truemotion=motion.truemotion, searchparam=search.param_recalc, chroma=self.chroma
        ) | self.recalculate_args

        if self.mvtools is MVToolsPlugin.FLOAT_NEW:
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

        if self.mvtools is MVToolsPlugin.FLOAT_NEW:
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
        self, func: GenericVSFunction, thSAD: int = 150, *, ref: vs.VideoNode | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        ref = self.get_ref_clip(ref, self.__class__.compensate)

        vect_b, vect_f = self.get_vectors_bf()

        compensate_args = dict(
            super=self.vectors.super_render, thsad=thSAD,
            tff=self.source_type.is_inter and self.source_type.value or None
        ) | self.compensate_args

        comp_back, comp_forw = [
            [self.mvtools.Compensate(ref, vectors=vect, **compensate_args) for vect in vectors]
            for vectors in (reversed(vect_b), vect_f)
        ]

        comp_clips = [*comp_forw, ref, *comp_back]
        n_clips = len(comp_clips)

        interleaved = core.std.Interleave(comp_clips)

        processed = func(interleaved, **kwargs)

        return processed.std.SelectEvery(cycle=n_clips, offsets=ceil(n_clips / 2))

    def degrain(
        self, thSAD: int | tuple[int, int] = 300, limit: int | tuple[int, int] = 255,
        thSCD: tuple[int | None, int | None] = (None, 130),
        *, ref: vs.VideoNode | None = None
    ) -> vs.VideoNode:
        ref = self.get_ref_clip(ref, self.__class__.degrain)

        thSAD, thSADC = normalize_seq(thSAD, 2)
        limit, limitC = normalize_seq(limit, 2)
        thSCD1, thSCD2 = thSCD

        thSAD = round(exp(-101. / (thSAD * 0.83)) * 360)
        thSADC = fallback(thSADC, round(thSAD * 0.18875 * exp(2 * 0.693)))

        thSCD1 = fallback(thSCD1, round(0.35 * thSAD + 260))
        thSCD2 = fallback(thSCD2, 130)

        vect_b, vect_f = self.get_vectors_bf()

        degrain_args = dict[str, Any](thscd1=thSCD1, thscd2=thSCD2, plane=self.mv_plane)

        if self.mvtools is MVToolsPlugin.INTEGER:
            degrain_args.update(thsad=thSAD, thsadc=thSADC, limit=limit, limitc=limitC)
        else:
            degrain_args.update(thsad=[thSAD, thSADC, thSADC], limit=[limit, limitC])

            if self.mvtools is MVToolsPlugin.FLOAT_NEW:
                degrain_args.update(thsad2=[thSAD / 2, thSADC / 2])

        if self.mvtools is MVToolsPlugin.FLOAT_NEW:
            output = self.mvtools.Degrain()(ref, self.vectors.super_render, self.vectors.vmulti, **degrain_args)
        else:
            output = self.mvtools.Degrain(self.tr)(
                ref, self.vectors.super_render, *chain.from_iterable(zip(vect_b, vect_f)), **degrain_args
            )

        return output.std.DoubleWeave(self.source_type.value) if self.source_type.is_inter else output

    def get_ref_clip(self, ref: vs.VideoNode | None, func: FuncExceptT) -> ConstantFormatVideoNode:
        ref = fallback(ref, self.workclip)

        check_ref_clip(self.workclip, ref)

        if self.high_precision:
            ref = depth(ref, 32)

        assert check_variable(ref, func)

        return ref

    def get_subpel_clips(
        self, pref: vs.VideoNode, ref: vs.VideoNode, pel_type: tuple[PelType, PelType]
    ) -> tuple[vs.VideoNode | None, vs.VideoNode | None]:
        return tuple(  # type: ignore[return-value]
            None if ptype is PelType.NONE else ptype(  # type: ignore[misc]
                clip, self.pel, default=PelType.WIENER if is_ref else PelType.BICUBIC
            ) for is_ref, ptype, clip in zip((False, True), pel_type, (pref, ref))
        )
