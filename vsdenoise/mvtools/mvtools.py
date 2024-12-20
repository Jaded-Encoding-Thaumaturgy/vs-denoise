from __future__ import annotations

from fractions import Fraction
from itertools import chain
from typing import Any, Callable, Concatenate, Sequence, Union, overload

from vstools import (
    ConstantFormatVideoNode, CustomOverflowError, CustomRuntimeError, FieldBased, FieldBasedT, FuncExceptT,
    InvalidColorFamilyError, KwargsT, OutdatedPluginError, P, PlanesT,
    check_ref_clip, check_variable, clamp, core, depth, disallow_variable_format,
    disallow_variable_resolution, fallback, kwargs_fallback, normalize_planes, normalize_seq, vs
)

from ..prefilters import PelType, Prefilter, prefilter_to_full_range
from .enums import FinestMode, FlowMode, MotionMode, MVDirection, MVToolsPlugin, SADMode, SearchMode
from .motion import MotionVectors, SuperClips
from .utils import normalize_thscd, planes_to_mvtools

__all__ = [
    'MVTools'
]


class MVTools:
    """MVTools wrapper for motion analysis / degrain / compensation"""

    super_args: KwargsT
    """Arguments passed to all the :py:attr:`MVToolsPlugin.Super` calls."""

    analyze_args: KwargsT
    """Arguments passed to all the :py:attr:`MVToolsPlugin.Analyze` calls."""

    recalculate_args: KwargsT
    """Arguments passed to all the :py:attr:`MVToolsPlugin.Recalculate` calls."""

    compensate_args: KwargsT
    """Arguments passed to all the :py:attr:`MVToolsPlugin.Compensate` calls."""

    vectors: MotionVectors
    """Motion vectors analyzed and used for all operations."""

    clip: vs.VideoNode
    """Clip to process."""

    @disallow_variable_format
    @disallow_variable_resolution
    def __init__(
        self, clip: vs.VideoNode,
        tr: int = 2, refine: int = 1, pel: int | None = None,
        planes: int | Sequence[int] | None = None,
        source_type: FieldBasedT | None = None,
        high_precision: bool = False,
        hpad: int | None = None, vpad: int | None = None,
        vectors: MotionVectors | MVTools | None = None,
        *,
        # kwargs for mvtools calls
        super_args: KwargsT | None = None,
        analyze_args: KwargsT | None = None,
        recalculate_args: KwargsT | None = None,
        compensate_args: KwargsT | None = None,
        flow_args: KwargsT | None = None,
        # super kwargs
        range_conversion: float | None = None, sharp: int | None = None,
        rfilter: int | None = None, prefilter: Prefilter | vs.VideoNode | None = None,
        pel_type: PelType | tuple[PelType, PelType] | None = None,
        # analyze kwargs
        block_size: int | None = None, overlap: int | None = None,
        thSAD: int | None = None, search: SearchMode | SearchMode.Config | None = None,
        sad_mode: SADMode | tuple[SADMode, SADMode] | None = None,
        motion: MotionMode.Config | None = None, finest_mode: FinestMode = FinestMode.NONE
    ) -> None:
        """
        MVTools is a wrapper around the Motion Vector Tools plugin for VapourSynth,
        used for estimation and compensation of object motion in video clips.

        This may be used for strong temporal denoising, degraining,
        advanced framerate conversions, image restoration, and other similar tasks.

        The plugin uses block-matching method of motion estimation (similar methods are used in MPEG2, MPEG4, etc).

        Of course, the motion estimation and compensation is not ideal and precise.\n
        In some complex cases (video with fading, ultra-fast motion, or periodic structures)
        the motion estimation may be completely wrong, and the compensated frame will be blocky and(/or) ugly.

        Severe difficulty is also due to objects mutual screening (occlusion) or reverse opening.\n
        Complex scripts with many motion compensation functions may eat huge amounts of memory
        which results in very slow processing.

        It's not simple to use, but it's quite an advanced plugin.
        The goal of this wrapper is to make it more accessible to your average user.
        However, use it for appropriate cases only, and try tuning its (many) parameters.

        :param clip:                Input clip to process. Must be either a GRAY or YUV format.
        :param tr:                  Temporal radius of the processing.
        :param refine:              This represents the times the analyzed clip will be recalculated.\n
                                    With every recalculation step, the block size will be further refined.\n
                                    i.e. `refine=3` it will analyze at `block_size=32`, then refine at 16, 8, 4.
                                    Set `refine=0` to disable recalculation completely.
        :param pel:                 Pixel EnLargement value, a.k.a. subpixel accuracy of the motion estimation.\n
                                    Value can only be 1, 2 or 4.
                                     * 1 means a precision to the pixel.
                                     * 2 means a precision to half a pixel.
                                     * 4 means a precision to quarter a pixel.
                                    `pel=4` is produced by spatial interpolation which is more accurate,
                                    but slower and not always better due to big level scale step.
        :param planes:              Planes to process.
        :param source_type:         Source type of the input clip.
        :param high_precision:      Whether to process everything in float32 (very slow).
                                    If set to False, it will process it in the input clip's bitdepth.
        :param hpad:                Horizontal padding added to source frame (both left and right).\n
                                    Small padding is added for more correct motion estimation near frame borders.
        :param vpad:                Vertical padding added to source frame (both top and bottom).
        :param vectors:             Precalculated vectors, either a custom instance or another MVTools instance.

        :param super_args:          Arguments passed to all the :py:attr:`MVToolsPlugin.Super` calls.
        :param analyze_args:        Arguments passed to all the :py:attr:`MVToolsPlugin.Analyze` calls.
        :param recalculate_args:    Arguments passed to all the :py:attr:`MVToolsPlugin.Recalculate` calls.
        :param compensate_args:     Arguments passed to all the :py:attr:`MVToolsPlugin.Compensate` calls.
        :param flow_args:           Arguments passed to all the :py:attr:`MVToolsPlugin.Flow` calls.

        :param block_size:          Block size to be used as smallest portion of the picture for analysis.
        :param overlap:             N block overlap value. Must be even to or lesser than the block size.\n
                                    The step between blocks for motion estimation is equal to `block_size - overlap`.\n
                                    N blocks cover the size `(block_size - overlap) * N + overlap` on the frame.\n
                                    Try using overlap value from `block_size / 4` to `block_size / 2`.\n
                                    The greater the overlap, the higher the amount of blocks,
                                    and the longer the processing will take.\n
                                    However the default value of 0 may cause blocking-like artefacts.\n
        :param thSAD:               During the recalculation, only bad quality new vectors with SAD above this thSAD
                                    will be re-estimated by search. thSAD value is scaled to 8x8 block size.
                                    Good vectors are not changed, but their SAD will be re-calculated and updated.
        :param range_conversion:    If the input is limited, it will be converted to full range
                                    to allow the motion analysis to use a wider array of information.\n
                                    This is for deciding what range conversion method to use.
                                     * >= 1.0 - Expansion with expr based on this coefficient.
                                     * >  0.0 - Expansion with retinex.
                                     * <= 0.0 - Simple conversion with resize plugin.
        :param search:              Decides the type of search at every level of the hierarchial
                                    analysis made while searching for motion vectors.
        :param sharp:               Subpixel interpolation method for pel = 2 or 4. Possible values are 0, 1, 2.\n
                                     * 0 - for soft interpolation (bilinear).
                                     * 1 - for bicubic interpolation (4 tap Catmull-Rom).
                                     * 2 - for sharper Wiener interpolation (6 tap, similar to Lanczos).
                                    This parameter controls the calculation of the first level only.
                                    When pel = 4, bilinear interpolation is always used to compute the second level.
        :param rfilter:             Hierarchical levels smoothing and reducing (halving) filter.\n
                                     * 0 - Simple 4 pixels averaging.
                                     * 1 - Triangle (shifted) for more smoothing (decrease aliasing).
                                     * 2 - Triangle filter like Bilinear for even more smoothing.
                                     * 3 - Quadratic filter for even more smoothing.
                                     * 4 - Cubic filter like Bicubic(b=1, c=0) for even more smoothing.
        :param sad_mode:            SAD Calculation mode.
        :param motion:              A preset or custom parameters values for truemotion/motion analysis mode.
        :param prefilter:           Prefilter to use for motion estimation. Can be a prefiltered clip instead.
                                    The ideal prefiltered clip will be one that has little to not
                                    temporal instability or dynamic grain, but retains most of the detail.
        :param pel_type:            Type of interpolation to use for upscaling the pel clip.
        """

        assert check_variable(clip, self.__class__)

        InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), self.__class__)

        self.clip = clip
        self.workclip = self.clip

        self.source_type = FieldBased.from_param_or_video(source_type, self.clip, False, self.__class__)
        self.is_hd = clip.width >= 1100 or clip.height >= 600
        self.is_uhd = self.clip.width >= 2600 or self.clip.height >= 1500

        self.tr = tr
        self.refine = refine
        self.pel = fallback(pel, 1 + int(not self.is_hd))
        self.planes = normalize_planes(self.clip, planes)

        self.is_gray = self.planes == [0]
        self.mv_plane = planes_to_mvtools(self.planes)
        self.chroma = self.mv_plane != 0
        self.high_precision = high_precision

        self.hpad = fallback(hpad, 8 if self.is_hd else 16)
        self.hpad_uhd = self.hpad // 2 if self.is_uhd else self.hpad
        self.vpad = fallback(vpad, 8 if self.is_hd else 16)
        self.vpad_half = self.vpad // 2 if self.is_uhd else self.vpad

        self.super_args = fallback(super_args, KwargsT())
        self.analyze_args = fallback(analyze_args, KwargsT())
        self.recalculate_args = fallback(recalculate_args, KwargsT())
        self.compensate_args = fallback(compensate_args, KwargsT())
        self.flow_args = fallback(flow_args, KwargsT())

        if self.refine > 6:
            raise CustomOverflowError(f'Refine > 6 is not supported! ({refine})', self.__class__)

        if self.high_precision:
            self.workclip = depth(self.workclip, 32)

        self.mvtools = MVToolsPlugin.from_video(self.workclip)

        if self.source_type.is_inter:
            self.workclip = self.workclip.std.SeparateFields(self.source_type.is_tff)

            if self.mvtools is MVToolsPlugin.INTEGER:
                if 'time' not in str(core.mv.Compensate.signature):
                    raise OutdatedPluginError(self.__class__, f'{self.__class__.__name__} {self.mvtools.name}')
            elif self.mvtools in (MVToolsPlugin.FLOAT_OLD, MVToolsPlugin.FLOAT_NEW):
                if not hasattr(self.mvtools.namespace, 'Flow'):
                    raise OutdatedPluginError(self.__class__, f'{self.__class__.__name__} {self.mvtools.name}')

        self.super_func_kwargs = dict(
            rfilter=rfilter, range_conversion=range_conversion, sharp=sharp,
            prefilter=prefilter, pel_type=pel_type
        )

        self.supers: SuperClips | None = None

        self.analyze_func_kwargs = dict(
            overlap=overlap, search=search, block_size=block_size, sad_mode=sad_mode,
            motion=motion, thSAD=thSAD
        )

        self.finest_mode = finest_mode

        if self.mvtools is MVToolsPlugin.INTEGER and self.finest_mode is not FinestMode.NONE:
            raise CustomRuntimeError(
                'finest_mode != NONE is only supported in the float plugin!', reason=dict(finest_mode=self.finest_mode)
            )

        if isinstance(vectors, MVTools):
            self.vectors = vectors.vectors
        elif isinstance(vectors, MotionVectors):
            self.vectors = vectors
        else:
            self.vectors = MotionVectors()

    def super(
        self, range_conversion: float | None = None, sharp: int | None = None,
        rfilter: int | None = None, prefilter: Prefilter | vs.VideoNode | None = None,
        pel_type: PelType | tuple[PelType, PelType] | None = None,
        *, ref: vs.VideoNode | None = None, inplace: bool = False
    ) -> SuperClips:
        """
        Calculates Super clips for rendering, searching, and recalculating.

        :param range_conversion:    If the input is limited, it will be converted to full range
                                    to allow the motion analysis to use a wider array of information.\n
                                    This is for deciding what range conversion method to use.
                                     * >= 1.0 - Expansion with expr based on this coefficient.
                                     * >  0.0 - Expansion with retinex.
                                     * <= 0.0 - Simple conversion with resize plugin.
        :param sharp:               Subpixel interpolation method for pel = 2 or 4. Possible values are 0, 1, 2.\n
                                     * 0 - for soft interpolation (bilinear).
                                     * 1 - for bicubic interpolation (4 tap Catmull-Rom).
                                     * 2 - for sharper Wiener interpolation (6 tap, similar to Lanczos).
                                    This parameter controls the calculation of the first level only.
                                    When pel = 4, bilinear interpolation is always used to compute the second level.
        :param rfilter:             Hierarchical levels smoothing and reducing (halving) filter.\n
                                     * 0 - Simple 4 pixels averaging.
                                     * 1 - Triangle (shifted) for more smoothing (decrease aliasing).
                                     * 2 - Triangle filter like Bilinear for even more smoothing.
                                     * 3 - Quadratic filter for even more smoothing.
                                     * 4 - Cubic filter like Bicubic(b=1, c=0) for even more smoothing.
        :param prefilter:           Prefilter to use for motion estimation. Can be a prefiltered clip instead.
                                    The ideal prefiltered clip will be one that has little to not
                                    temporal instability or dynamic grain, but retains most of the detail.
        :param pel_type:            Type of interpolation to use for upscaling the pel clip.
        :param ref:                 Reference clip to use for creating super clips.

        :return:                    SuperClips tuple containing the render, search, and recalculate super clips.
        """

        ref = self.get_ref_clip(ref, self.super)
        rfilter = kwargs_fallback(rfilter, (self.super_func_kwargs, 'rfilter'), 3)
        range_conversion = kwargs_fallback(range_conversion, (self.super_func_kwargs, 'range_conversion'), 5.0)

        sharp = kwargs_fallback(sharp, (self.super_func_kwargs, 'sharp'), 2)

        prefilter = kwargs_fallback(  # type: ignore[assignment]
            prefilter, (self.super_func_kwargs, 'prefilter'), Prefilter.AUTO
        )

        pel_type = kwargs_fallback(  # type: ignore[assignment]
            pel_type, (self.super_func_kwargs, 'pel_type'), PelType.AUTO
        )

        if not isinstance(pel_type, tuple):
            pel_type = (pel_type, pel_type)  # type: ignore[assignment]

        if isinstance(prefilter, Prefilter):
            prefilter = prefilter(ref, self.planes)

            prefilter = prefilter_to_full_range(prefilter, range_conversion, self.planes)

        assert prefilter is not None

        if self.high_precision:
            prefilter = depth(prefilter, 32)

        check_ref_clip(ref, prefilter)
        pelclip, pelclip2 = self.get_subpel_clips(prefilter, ref, pel_type)  # type: ignore[arg-type]

        common_args = KwargsT(
            sharp=sharp, pel=self.pel, vpad=self.vpad_half, hpad=self.hpad_uhd, chroma=self.chroma
        ) | self.super_args
        super_render_args = common_args | dict(levels=1, hpad=self.hpad, vpad=self.vpad, chroma=not self.is_gray)

        if pelclip or pelclip2:
            common_args |= dict(pelclip=pelclip)  # type: ignore
            super_render_args |= dict(pelclip=pelclip2)  # type: ignore

        super_render = self.mvtools.Super(ref if inplace else self.workclip, **super_render_args)
        super_search = self.mvtools.Super(ref, **(dict(rfilter=rfilter) | common_args))
        super_recalc = self.refine and self.mvtools.Super(prefilter, **(dict(levels=1) | common_args)) or super_render

        supers = SuperClips(ref, super_render, super_search, super_recalc)

        if not inplace:
            self.supers = supers

        return supers

    def analyze(
        self, block_size: int | None = None, overlap: int | None = None, thSAD: int | None = None,
        search: SearchMode | SearchMode.Config | None = None,
        sad_mode: SADMode | tuple[SADMode, SADMode] | None = None,
        motion: MotionMode.Config | None = None, supers: SuperClips | None = None,
        *, ref: vs.VideoNode | None = None, inplace: bool = False
    ) -> MotionVectors:
        """
        During the analysis stage, the plugin divides frames by small blocks and for every block in current frame
        it tries to find the most similar (matching) block in the second frame (previous or next).\n
        The relative shift of these blocks is represented by a motion vector.

        The main measure of block similarity is the sum of absolute differences (SAD) of all pixels
        of the two compared blocks. SAD is a value which says how good the motion estimation was.

        :param block_size:          Block size to be used as smallest portion of the picture for analysis.
        :param overlap:             N block overlap value. Must be even to or lesser than the block size.\n
                                    The step between blocks for motion estimation is equal to `block_size - overlap`.\n
                                    N blocks cover the size `(block_size - overlap) * N + overlap` on the frame.\n
                                    Try using overlap value from `block_size / 4` to `block_size / 2`.\n
                                    The greater the overlap, the higher the amount of blocks,
                                    and the longer the processing will take.\n
                                    However the default value of 0 may cause blocking-like artefacts.\n
        :param thSAD:               During the recalculation, only bad quality new vectors with SAD above this thSAD
                                    will be re-estimated by search. thSAD value is scaled to 8x8 block size.
                                    Good vectors are not changed, but their SAD will be re-calculated and updated.
        :param search:              Decides the type of search at every level of the hierarchial
                                    analysis made while searching for motion vectors.
        :param sad_mode:            SAD Calculation mode.
        :param motion:              A preset or custom parameters values for truemotion/motion analysis mode.
        :param supers:              Custom super clips to be used for analyze.
        :param ref:                 Reference clip to use for analyzes over the main clip.
        :param inplace:             Whether to save the analysis in the MVTools instance or not.

        :return:                    :py:class:`MotionVectors` object with the analyzed motion vectors.
        """

        ref = self.get_ref_clip(ref, self.analyze)

        block_size = kwargs_fallback(block_size, (self.analyze_func_kwargs, 'block_size'), 16 if self.is_hd else 8)
        blocksize = max(self.refine and 2 ** (self.refine + 1), block_size)

        halfblocksize = max(2, blocksize // 2)
        halfoverlap = max(2, halfblocksize // 2)

        overlap = kwargs_fallback(overlap, (self.analyze_func_kwargs, 'overlap'), halfblocksize)

        thSAD = kwargs_fallback(thSAD, (self.analyze_func_kwargs, 'thSAD'), 300)

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

        vectors = MotionVectors() if inplace else self.vectors

        if isinstance(sad_mode, tuple):
            sad_mode, recalc_sad_mode = sad_mode
        else:
            sad_mode, recalc_sad_mode = sad_mode, SADMode.SATD

        supers = supers or self.get_supers(ref, inplace=inplace)

        thSAD_recalc = thSAD

        t2 = (self.tr * 2 if self.tr > 1 else self.tr) if self.source_type.is_inter else self.tr

        analyze_args = KwargsT(
            dct=sad_mode, pelsearch=search.pel, blksize=blocksize, overlap=overlap, search=search.mode,
            truemotion=motion.truemotion, searchparam=search.param, chroma=self.chroma,
            plevel=motion.plevel, pglobal=motion.pglobal, pnew=motion.pnew,
            lambda_=motion.block_coherence(blocksize), lsad=motion.sad_limit,
            fields=self.source_type.is_inter
        ) | self.analyze_args

        if self.mvtools is MVToolsPlugin.FLOAT_NEW:
            vectors.vmulti = self.mvtools.Analyse(supers.search, radius=t2, **analyze_args)
        else:
            for i in range(1, t2 + 1):
                vectors.calculate_vectors(i, self.mvtools, supers, False, self.finest_mode, **analyze_args)

        if self.refine:
            self.recalculate(
                self.refine, self.tr, blocksize, halfoverlap, thSAD_recalc,
                search, recalc_sad_mode, motion, vectors, supers, ref=ref
            )

        vectors.kwargs.update(thSAD=thSAD)

        return vectors

    def recalculate(
        self, refine: int = 1, tr: int | None = None, block_size: int | None = None,
        overlap: int | None = None, thSAD: int | None = None,
        search: SearchMode | SearchMode.Config | None = None, sad_mode: SADMode = SADMode.SATD,
        motion: MotionMode.Config | None = None, vectors: MotionVectors | MVTools | None = None,
        supers: SuperClips | None = None, *, ref: vs.VideoNode | None = None
    ) -> None:
        ref = self.get_ref_clip(ref, self.recalculate)

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        if not vectors.has_vectors:
            raise CustomRuntimeError('You need to first run analyze before recalculating!', self.recalculate)

        tr = min(tr, self.tr) if tr else self.tr
        t2 = (tr * 2 if tr > 1 else tr) if self.source_type.is_inter else tr

        blocksize = max(refine and 2 ** (refine + 1), fallback(block_size, 16 if self.is_hd else 8))
        halfblocksize = max(2, blocksize // 2)

        overlap = fallback(overlap, max(2, max(2, blocksize // 2) // 2))

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

        recalc_args = KwargsT(
            search=search.recalc_mode, dct=sad_mode, thsad=thSAD, blksize=halfblocksize,
            overlap=overlap, truemotion=motion.truemotion, searchparam=search.param_recalc,
            chroma=self.chroma, pnew=motion.pnew, lambda_=motion.block_coherence(halfblocksize),
            fields=self.source_type.is_inter
        ) | self.recalculate_args

        supers = supers or self.get_supers(ref, inplace=True)

        if self.mvtools is MVToolsPlugin.FLOAT_NEW:
            for i in range(refine):
                recalc_blksize = clamp(blocksize / 2 ** i, 4, 128)

                vectors.vmulti = self.mvtools.Recalculate(
                    supers.recalculate, vectors=vectors.vmulti, **(recalc_args | dict(
                        blksize=recalc_blksize, overlap=recalc_blksize / 2,
                        lambda_=motion.block_coherence(recalc_blksize)
                    ))
                )
        else:
            for i in range(1, t2 + 1):
                if not vectors.has_mv(MVDirection.BACK, i) or not vectors.has_mv(MVDirection.FWRD, i):
                    continue

                for j in range(0, refine):
                    recalc_blksize = clamp(blocksize / 2 ** j, 4, 128)

                    vectors.calculate_vectors(
                        i, self.mvtools, supers, True, self.finest_mode, **(recalc_args | dict(
                            blksize=recalc_blksize, overlap=recalc_blksize // 2,
                            lambda_=motion.block_coherence(recalc_blksize)
                        ))
                    )

    @overload
    def compensate(  # type: ignore
        self, func: Union[
            Callable[Concatenate[vs.VideoNode, P], vs.VideoNode],
            Callable[Concatenate[list[vs.VideoNode], P], vs.VideoNode]
        ] | None = None,
        tr: int | None = None, thSAD: int = 10000, thSCD: int | tuple[int | None, int | None] | None = None,
        supers: SuperClips | None = None, *args: P.args, ref: vs.VideoNode | None = None,
        **kwargs: P.kwargs
    ) -> vs.VideoNode:
        """
        At compensation stage, the plugin client functions read the motion vectors and use them to move blocks
        and form a motion compensated frame (or realize some other full- or partial motion compensation or
        interpolation function).

        Every block in this fully-compensated frame is placed in the same position as this block in current frame.

        So, we may (for example) use strong temporal denoising even for quite fast moving objects without producing
        annoying artefactes and ghosting (object's features and edges coincide if compensation is perfect).

        This function is for using compensated and original frames to create an interleaved clip,
        denoising it with the external temporal filter `func`, and select central cleaned original frames for output.

        :param func:    Temporal function to motion compensate.
        :param thSAD:   This is the SAD threshold for safe (dummy) compensation.\n
                        If block SAD is above thSAD, the block is bad, and we use source block
                        instead of the compensated block.
        :param thSCD:   The first value is a threshold for whether a block has changed
                        between the previous frame and the current one.\n
                        When a block has changed, it means that motion estimation for it isn't relevant.
                        It, for example, occurs at scene changes, and is one of the thresholds used to
                        tweak the scene changes detection engine.\n
                        Raising it will lower the number of blocks detected as changed.\n
                        It may be useful for noisy or flickered video. This threshold is compared to the SAD value.\n
                        For exactly identical blocks we have SAD = 0, but real blocks are always different
                        because of objects complex movement (zoom, rotation, deformation),
                        discrete pixels sampling, and noise.\n
                        Suppose we have two compared 8×8 blocks with every pixel different by 5.\n
                        It this case SAD will be 8×8×5 = 320 (block will not detected as changed for thSCD1 = 400).\n
                        Actually this parameter is scaled internally in MVTools,
                        and it is always relative to 8x8 block size.\n
                        The second value is a threshold of the percentage of how many blocks have to change for
                        the frame to be considered as a scene change. It ranges from 0 to 100 %.
        :param supers:  Custom super clips to be used for compensating.
        :param wargs:   Arguments passed to `func` to avoid using `partial`.
        :param ref:     Reference clip to use instead of main clip.
        :param kwargs:  Keyword arguments passed to `func` to avoid using `partial`.

        :return:        Motion compensated output of `func`.
        """

    @overload
    def compensate(
        self, func: None = None,
        tr: int | None = None, thSAD: int = 10000, thSCD: int | tuple[int | None, int | None] | None = None,
        supers: SuperClips | None = None, ref: vs.VideoNode | None = None
    ) -> tuple[vs.VideoNode, tuple[int, int]]:
        """
        At compensation stage, the plugin client functions read the motion vectors and use them to move blocks
        and form a motion compensated frame (or realize some other full- or partial motion compensation or
        interpolation function).

        Every block in this fully-compensated frame is placed in the same position as this block in current frame.

        So, we may (for example) use strong temporal denoising even for quite fast moving objects without producing
        annoying artefactes and ghosting (object's features and edges coincide if compensation is perfect).

        This function is for using compensated and original frames to create an interleaved clip,
        denoising it with the external temporal filter `func`, and select central cleaned original frames for output.

        :param thSAD:   This is the SAD threshold for safe (dummy) compensation.\n
                        If block SAD is above thSAD, the block is bad, and we use source block
                        instead of the compensated block.
        :param thSCD:   The first value is a threshold for whether a block has changed
                        between the previous frame and the current one.\n
                        When a block has changed, it means that motion estimation for it isn't relevant.
                        It, for example, occurs at scene changes, and is one of the thresholds used to
                        tweak the scene changes detection engine.\n
                        Raising it will lower the number of blocks detected as changed.\n
                        It may be useful for noisy or flickered video. This threshold is compared to the SAD value.\n
                        For exactly identical blocks we have SAD = 0, but real blocks are always different
                        because of objects complex movement (zoom, rotation, deformation),
                        discrete pixels sampling, and noise.\n
                        Suppose we have two compared 8×8 blocks with every pixel different by 5.\n
                        It this case SAD will be 8×8×5 = 320 (block will not detected as changed for thSCD1 = 400).\n
                        Actually this parameter is scaled internally in MVTools,
                        and it is always relative to 8x8 block size.\n
                        The second value is a threshold of the percentage of how many blocks have to change for
                        the frame to be considered as a scene change. It ranges from 0 to 100 %.
        :param supers:  Custom super clips to be used for compensating.
        :param ref:     Reference clip to use instead of main clip.

        :return:        A tuple of motion compensated clip, then a tuple of (cycle, offset) so that
                        compensated.std.SelectEvery(cycle, offsets) will give the original clip.
        """

    def compensate(  # type: ignore
        self, func: Union[
            Callable[Concatenate[vs.VideoNode, P], vs.VideoNode],
            Callable[Concatenate[list[vs.VideoNode], P], vs.VideoNode]
        ] | None = None,
        tr: int | None = None, thSAD: int = 10000, thSCD: int | tuple[int | None, int | None] | None = None,
        supers: SuperClips | None = None, *args: P.args, ref: vs.VideoNode | None = None,
        **kwargs: P.kwargs
    ) -> vs.VideoNode | tuple[vs.VideoNode, tuple[int, int]]:
        ref = self.get_ref_clip(ref, self.compensate)
        tr = min(tr, self.tr) if tr else self.tr

        thSCD1, thSCD2 = normalize_thscd(thSCD, self.compensate)
        supers = supers or self.get_supers(ref, inplace=True)

        vect_b, vect_f = self.get_vectors_bf(self.vectors, tr=tr)

        compensate_args = dict(
            super=supers.render, thsad=thSAD,
            thscd1=thSCD1, thscd2=thSCD2,
            fields=self.source_type.is_inter,
            tff=self.source_type.is_inter and self.source_type.is_tff or None
        ) | self.compensate_args

        comp_back, comp_fwrd = [
            [self.mvtools.Compensate(ref, vectors=vect, **compensate_args) for vect in vectors]
            for vectors in (reversed(vect_b), vect_f)
        ]

        comp_clips = [*comp_fwrd, ref, *comp_back]
        n_clips = len(comp_clips)
        offset = (n_clips - 1) // 2

        interleaved = core.std.Interleave(comp_clips)

        if func:
            processed = func(interleaved, *args, **kwargs)  # type: ignore

            return processed.std.SelectEvery(n_clips, offset)

        return interleaved, (n_clips, offset)

    @overload
    def flow(  # type: ignore
        self, func: Union[
            Callable[Concatenate[vs.VideoNode, P], vs.VideoNode],
            Callable[Concatenate[list[vs.VideoNode], P], vs.VideoNode]
        ] | None = None, 
        tr: int | None = None, time: float = 100, mode: FlowMode = FlowMode.ABSOLUTE,
        thSCD: int | tuple[int | None, int | None] | None = None,
        supers: SuperClips | None = None, *args: P.args, ref: vs.VideoNode | None = None,
        **kwargs: P.kwargs
    ) -> vs.VideoNode:
        ...

    @overload
    def flow(  # type: ignore
        self, func: None = None, 
        tr: int | None = None, time: float = 100, mode: FlowMode = FlowMode.ABSOLUTE,
        thSCD: int | tuple[int | None, int | None] | None = None,
        supers: SuperClips | None = None, *args: P.args, ref: vs.VideoNode | None = None,
        **kwargs: P.kwargs
    ) -> tuple[vs.VideoNode, tuple[int, int]]:
        ...

    def flow(  # type: ignore
        self, func: Union[
            Callable[Concatenate[vs.VideoNode, P], vs.VideoNode],
            Callable[Concatenate[list[vs.VideoNode], P], vs.VideoNode]
        ] | None = None,
        tr: int | None = None, time: float = 100, mode: FlowMode = FlowMode.ABSOLUTE,
        thSCD: int | tuple[int | None, int | None] | None = None,
        supers: SuperClips | None = None, *args: P.args, ref: vs.VideoNode | None = None,
        **kwargs: P.kwargs
    ) -> vs.VideoNode | tuple[vs.VideoNode, tuple[int, int]]:
        ref = self.get_ref_clip(ref, self.flow)
        tr = min(tr, self.tr) if tr else self.tr

        thSCD1, thSCD2 = normalize_thscd(thSCD, self.flow)
        supers = supers or self.get_supers(ref, inplace=True)

        vect_b, vect_f = self.get_vectors_bf(self.vectors, tr=tr)

        flow_args = KwargsT(  # type: ignore
            super=supers.render, time=time, mode=mode,
            thscd1=thSCD1, thscd2=thSCD2,
            fields=self.source_type.is_inter,
            tff=self.source_type.is_inter and self.source_type.is_tff or None
        ) | self.flow_args

        flow_back, flow_fwrd = [
            [self.mvtools.Flow(ref, vectors=vect, **flow_args) for vect in vectors]
            for vectors in (reversed(vect_b), vect_f)
        ]

        flow_clips = [*flow_fwrd, ref, *flow_back]
        n_clips = len(flow_clips)
        offset = (n_clips - 1) // 2

        interleaved = core.std.Interleave(flow_clips)

        if func:
            processed = func(interleaved, *args, **kwargs)  # type: ignore

            return processed.std.SelectEvery(n_clips, offset)

        return interleaved, (n_clips, offset)

    def degrain(
        self,
        tr: int | None = None,
        thSAD: int | tuple[int | None, int | None] | None = None,
        limit: int | tuple[int, int] | None = None,
        thSCD: int | tuple[int | None, int | None] | None = None,
        supers: SuperClips | None = None,
        *, vectors: MotionVectors | MVTools | None = None, ref: vs.VideoNode | None = None
    ) -> vs.VideoNode:
        """
        Makes a temporal denoising with motion compensation.

        Blocks of previous and next frames are motion compensated and then averaged with current
        frame with weigthing factors depended on block differences from current (SAD).

        :param thSAD:   Defines the soft threshold of the block sum absolute differences.\n
                        If an int is specified, it will be used for luma and chroma will be a scaled value.\n
                        If a tuple is specified, the first value is for luma, second is for chroma.\n
                        If None, the same `thSAD` used in the `analyze` step will be used.\n
                        Block with SAD above threshold thSAD have a zero weight for averaging (denoising).\n
                        Block with low SAD has highest weight. Rest of weight is taken from pixels of source clip.\n
                        The provided thSAD value is scaled to a 8x8 blocksize.\n
                        Low values can result in staggered denoising, large values can result in ghosting and artifacts.
        :param limit:   Maximum change of pixel. This is post-processing to prevent some artifacts.\n
                        Value ranges from 0 to 255.
        :param thSCD:   The first value is a threshold for whether a block has changed
                        between the previous frame and the current one.\n
                        When a block has changed, it means that motion estimation for it isn't relevant.
                        It, for example, occurs at scene changes, and is one of the thresholds used to
                        tweak the scene changes detection engine.\n
                        Raising it will lower the number of blocks detected as changed.\n
                        It may be useful for noisy or flickered video. This threshold is compared to the SAD value.\n
                        For exactly identical blocks we have SAD = 0, but real blocks are always different
                        because of objects complex movement (zoom, rotation, deformation),
                        discrete pixels sampling, and noise.\n
                        Suppose we have two compared 8×8 blocks with every pixel different by 5.\n
                        It this case SAD will be 8×8×5 = 320 (block will not detected as changed for thSCD1 = 400).\n
                        Actually this parameter is scaled internally in MVTools,
                        and it is always relative to 8x8 block size.\n
                        The second value is a threshold of the percentage of how many blocks have to change for
                        the frame to be considered as a scene change. It ranges from 0 to 100 %.
        :param ref:     Reference clip to use rather than the main clip. If passed,
                        the degraining will be applied to the ref clip rather than the original input clip.
        :param supers:  Custom super clips to be used for degraining.

        :return:        Degrained clip.
        """

        ref = self.get_ref_clip(ref, self.degrain)
        tr = min(tr, self.tr) if tr else self.tr

        if isinstance(vectors, MVTools):
            vectors = vectors.vectors
        elif vectors is None:
            vectors = self.vectors

        vect_b, vect_f = self.get_vectors_bf(vectors, supers=supers, ref=ref, tr=tr)
        supers = supers or self.get_supers(ref, inplace=True)

        thSAD, thSADC = (thSAD if isinstance(thSAD, tuple) else (thSAD, None))

        thSAD = kwargs_fallback(thSAD, (vectors.kwargs, 'thSAD'), 300)
        thSADC = fallback(thSADC, thSAD // 2)

        limit, limitC = normalize_seq(limit, 2)

        thSCD1, thSCD2 = normalize_thscd(thSCD, self.degrain)

        degrain_args = dict[str, Any](thscd1=thSCD1, thscd2=thSCD2, plane=self.mv_plane)

        if self.mvtools is MVToolsPlugin.INTEGER:
            degrain_args.update(thsad=thSAD, thsadc=thSADC, limit=limit, limitc=limitC)
        else:
            degrain_args.update(thsad=[thSAD, thSADC, thSADC], limit=[limit, limitC])

            if self.mvtools is MVToolsPlugin.FLOAT_NEW:
                degrain_args.update(thsad2=[thSAD / 2, thSADC / 2])

        if self.mvtools is MVToolsPlugin.FLOAT_NEW:
            output = self.mvtools.Degrain()(ref, supers.render, vectors.vmulti, **degrain_args)
        else:
            output = self.mvtools.Degrain(tr)(
                ref, supers.render, *chain.from_iterable(zip(vect_b, vect_f)), **degrain_args
            )

        if not self.source_type.is_inter:
            return output

        return output.std.DoubleWeave(self.source_type.is_tff)[::2]

    def flow_interpolate(
        self,
        time: float = 50, mask_scale: float = 100, blend: bool = False,
        thSCD: int | tuple[int | None, int | None] | None = None,
        supers: SuperClips | None = None, *, ref: vs.VideoNode | None = None
    ) -> vs.VideoNode:
        ref = self.get_ref_clip(ref, self.flow_interpolate)
        thSCD1, thSCD2 = normalize_thscd(thSCD, self.flow_interpolate)

        supers = supers or self.get_supers(ref, inplace=True)
        vect_b, vect_f = self.get_vectors_bf(self.vectors, tr=1)

        return self.mvtools.FlowInter(
            ref, supers.render, vect_b, vect_f, time, mask_scale, blend, thSCD1, thSCD2
        )

    def flow_blur(
        self,
        blur: float = 50, pixel_precision: int = 1,
        thSCD: int | tuple[int | None, int | None] | None = None,
        supers: SuperClips | None = None, *, ref: vs.VideoNode | None = None
    ) -> vs.VideoNode:
        ref = self.get_ref_clip(ref, self.flow_blur)
        thSCD1, thSCD2 = normalize_thscd(thSCD, self.flow_blur)

        supers = supers or self.get_supers(ref, inplace=True)
        vect_b, vect_f = self.get_vectors_bf(self.vectors, tr=1)

        return self.mvtools.FlowBlur(
            ref, supers.render, vect_b, vect_f, blur, pixel_precision, thSCD1, thSCD2
        )

    def flow_fps(
        self,
        fps: Fraction, mask_type: int = 2, mask_scale: float = 100, blend: bool = False,
        thSCD: int | tuple[int | None, int | None] | None = None,
        supers: SuperClips | None = None, *, ref: vs.VideoNode | None = None
    ) -> vs.VideoNode:
        ref = self.get_ref_clip(ref, self.flow_fps)
        thSCD1, thSCD2 = normalize_thscd(thSCD, self.flow_fps)

        supers = supers or self.get_supers(ref, inplace=True)
        vect_b, vect_f = self.get_vectors_bf(self.vectors, tr=1)

        return self.mvtools.FlowFPS(
            ref, supers.render, vect_b, vect_f, fps.numerator, fps.denominator,
            mask_type, mask_scale, blend, thSCD1, thSCD2
        )

    def block_fps(
        self,
        fps: Fraction, mask_type: int = 0, mask_scale: float = 100, blend: bool = False,
        thSCD: int | tuple[int | None, int | None] | None = None,
        supers: SuperClips | None = None, *, ref: vs.VideoNode | None = None
    ) -> vs.VideoNode:
        ref = self.get_ref_clip(ref, self.block_fps)
        thSCD1, thSCD2 = normalize_thscd(thSCD, self.block_fps)

        supers = supers or self.get_supers(ref, inplace=True)
        vect_b, vect_f = self.get_vectors_bf(self.vectors, tr=1)

        return self.mvtools.BlockFPS(
            ref, supers.render, vect_b, vect_f, fps.numerator, fps.denominator,
            mask_type, mask_scale, blend, thSCD1, thSCD2
        )

    def mask(
        self,
        mask_type: int = 0, mask_scale: float = 100, gamma: float = 1.0,
        scenechange_y: int = 0, time: float = 100, fwd: bool = True,
        thSCD: int | tuple[int | None, int | None] | None = None,
        *, ref: vs.VideoNode | None = None
    ) -> vs.VideoNode:
        ref = self.get_ref_clip(ref, self.mask)
        thSCD1, thSCD2 = normalize_thscd(thSCD, self.mask)

        vect_b, vect_f = self.get_vectors_bf(self.vectors, tr=1)

        return self.mvtools.Mask(
            ref, vect_f if fwd else vect_b, mask_scale, gamma, mask_type,
            time, scenechange_y, thSCD1, thSCD2
        )

    def sc_detection(
        self,
        fwd: bool = True,
        thSCD: int | tuple[int | None, int | None] | None = None,
        *, ref: vs.VideoNode | None = None
    ) -> vs.VideoNode:
        ref = self.get_ref_clip(ref, self.sc_detection)
        thSCD1, thSCD2 = normalize_thscd(thSCD, self.sc_detection)

        vect_b, vect_f = self.get_vectors_bf(self.vectors, tr=1)

        sc_detect = self.mvtools.SCDetection(ref, vect_f if fwd else vect_b, thSCD1, thSCD2)

        return sc_detect

    def finest(self) -> None:
        self.analyze().finest(self.mvtools)

    def get_supers(self, ref: vs.VideoNode, *, inplace: bool = False) -> SuperClips:
        """
        Get the super clips for the specified ref clip.

        If :py:attr:`analyze` wasn't previously called,
        it will do so here with default values or kwargs specified in the constructor.

        :param inplace:     Only return the SuperClips object, not modifying the internal state.

        :return:            SuperClips tuple.
        """

        if self.supers and self.supers.base == ref:
            return self.supers

        return self.super(ref=ref, inplace=inplace)

    def get_vectors_bf(
        self, vectors: MotionVectors, *, supers: SuperClips | None = None,
        ref: vs.VideoNode | None = None, tr: int | None = None, inplace: bool = False
    ) -> tuple[list[vs.VideoNode], list[vs.VideoNode]]:
        """
        Get the backwards and forward vectors.\n

        If :py:attr:`analyze` wasn't previously called,
        it will do so here with default values or kwargs specified in the constructor.

        :param inplace:     Only return the list, not modifying the internal state.\n
                            (Useful if you haven't called :py:attr:`analyze` previously)

        :return:            Two lists, respectively for backward and forwards, containing motion vectors.
        """

        if not vectors.has_vectors:
            vectors = self.analyze(supers=supers, ref=ref, inplace=inplace)

        tr = min(tr, self.tr) if tr else self.tr
        t2 = (tr * 2 if tr > 1 else tr) if self.source_type.is_inter else tr

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
                vectors_backward.append(vectors.get_mv(MVDirection.BACK, i))
                vectors_forward.append(vectors.get_mv(MVDirection.FWRD, i))

        return (vectors_backward, vectors_forward)

    def get_ref_clip(self, ref: vs.VideoNode | None, func: FuncExceptT) -> ConstantFormatVideoNode:
        """
        Utility for getting the ref clip and set it up with internal modifying.

        :param ref:     Input clip. If None, the workclip will be used.
        :param func:    Function this was called from.

        :return:        Clip to be used in this instance of MVTools.
        """

        ref = fallback(ref, self.workclip)

        if self.high_precision:
            ref = depth(ref, 32)

        check_ref_clip(self.workclip, ref)

        assert check_variable(ref, func)

        return ref

    def get_subpel_clips(
        self, pref: vs.VideoNode, ref: vs.VideoNode, pel_type: tuple[PelType, PelType]
    ) -> tuple[vs.VideoNode | None, vs.VideoNode | None]:
        """
        Get upscaled clips for the subpel param.

        :param pref:        Prefiltered clip.
        :param ref:         Input clip.
        :param pel_type:    :py:class:`PelType` to use for upscaling.\n
                            First is for the prefilter, the other is for normal clip.

        :return:            Two values. An upscaled clip or None if PelType.NONE.
        """

        return tuple(  # type: ignore[return-value]
            None if ptype is PelType.NONE else ptype(  # type: ignore[misc]
                clip, self.pel, PelType.WIENER if is_ref else PelType.BICUBIC
            ) for is_ref, ptype, clip in zip((False, True), pel_type, (pref, ref))
        )

    @classmethod
    def denoise(
        cls, clip: vs.VideoNode, thSAD: int | tuple[int, int | tuple[int, int]] | None = None,
        tr: int = 2, refine: int = 1, block_size: int | None = None, overlap: int | None = None,
        prefilter: Prefilter | vs.VideoNode | None = None, pel: int | None = None,
        sad_mode: SADMode | tuple[SADMode, SADMode] | None = None,
        search: SearchMode | SearchMode.Config | None = None, motion: MotionMode.Config | None = None,
        pel_type: PelType | tuple[PelType, PelType] | None = None,
        planes: PlanesT = None, source_type: FieldBasedT | None = None, high_precision: bool = False,
        limit: int | tuple[int, int] | None = None, thSCD: int | tuple[int | None, int | None] | None = None,
        *, super_args: KwargsT | None = None, analyze_args: KwargsT | None = None,
        recalculate_args: KwargsT | None = None, compensate_args: KwargsT | None = None,
        range_conversion: float | None = None, sharp: int | None = None,
        hpad: int | None = None, vpad: int | None = None,
        rfilter: int | None = None, vectors: MotionVectors | MVTools | None = None,
        supers: SuperClips | None = None, ref: vs.VideoNode | None = None
    ) -> vs.VideoNode:
        mvtools = cls(
            clip, tr, refine, pel, planes, source_type, high_precision, hpad, vpad,
            vectors, super_args=super_args, analyze_args=analyze_args,
            recalculate_args=recalculate_args, compensate_args=compensate_args
        )

        if not isinstance(thSAD, Sequence):
            thSADA = thSADD = thSAD
        else:
            thSADA, thSADD = thSAD  # type: ignore

        supers = supers or mvtools.super(
            range_conversion, sharp, rfilter, prefilter, pel_type, inplace=True
        )

        vectors = vectors or mvtools.analyze(
            block_size, overlap, thSADA, search, sad_mode, motion, supers, inplace=True
        )

        return mvtools.degrain(tr, thSADD, limit, thSCD, supers, vectors=vectors, ref=ref)
