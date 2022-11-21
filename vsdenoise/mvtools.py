"""
This module implements wrappers for mvtool
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from math import ceil, exp
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeVar, cast, overload

from _collections_abc import dict_items, dict_keys, dict_values
from vstools import (
    MISSING, ColorRange, ConstantFormatVideoNode, CustomEnum, CustomIntEnum, CustomOverflowError, CustomStrEnum,
    CustomValueError, FieldBased, FieldBasedT, FuncExceptT, GenericVSFunction, InvalidColorFamilyError, KwargsNotNone,
    KwargsT, MissingT, VSFunction, check_ref_clip, check_variable, clamp, core, depth, disallow_variable_format,
    disallow_variable_resolution, fallback, inject_self, kwargs_fallback, normalize_planes, normalize_seq, scale_value,
    vs
)

from .prefilters import PelType, Prefilter, prefilter_to_full_range
from .utils import planes_to_mvtools

__all__ = [
    'MVTools', 'MVToolsPlugin',
    'SADMode', 'SearchMode', 'MotionMode',
    'MVDirection', 'MotionVectors',
    'MVToolsPresets'
]


class MVDirection(CustomStrEnum):
    """Motion vector analyze direction."""

    BACK = 'backward'
    """Backwards motion detection."""

    FWRD = 'forward'
    """Forwards motion detection."""

    @property
    def isb(self) -> bool:
        """Whether it's using backwards motion detection."""

        return self is MVDirection.BACK


class MotionVectors:
    vmulti: vs.VideoNode
    """Super analyzed clip."""

    super_render: vs.VideoNode
    """Super clip used for analyzing."""

    kwargs: dict[str, Any]

    temporal_vectors: dict[MVDirection, dict[int, vs.VideoNode]]
    """Dict containing backwards and forwards motion vectors."""

    def __init__(self) -> None:
        self._init_vects()
        self.kwargs = dict[str, Any]()

    def _init_vects(self) -> None:
        self.temporal_vectors = {w: {} for w in MVDirection}

    @property
    def got_vectors(self) -> bool:
        """Whether the instance uses bidirectional motion vectors."""

        return bool(self.temporal_vectors[MVDirection.BACK] and self.temporal_vectors[MVDirection.FWRD])

    def got_mv(self, direction: MVDirection, delta: int) -> bool:
        """
        Returns whether the motion vector exists.

        :param direction:   Which direction the motion vector was analyzed.
        :param delta:       Delta with which the motion vector was analyzed.

        :return:            Whether the motion vector exists.
        """

        return delta in self.temporal_vectors[direction]

    def get_mv(self, direction: MVDirection, delta: int) -> vs.VideoNode:
        """
        Get the motion vector.

        :param direction:   Which direction the motion vector was analyzed.
        :param delta:       Delta with which the motion vector was analyzed.

        :return:            Motion vector.
        """

        return self.temporal_vectors[direction][delta]

    def set_mv(self, direction: MVDirection, delta: int, vect: vs.VideoNode) -> None:
        """
        Sets the motion vector.

        :param direction:   Which direction the motion vector was analyzed.
        :param delta:       Delta with which the motion vector was analyzed.
        """

        self.temporal_vectors[direction][delta] = vect

    def clear(self) -> None:
        """Deletes all :py:class:`vsdenoise.mvtools.MotionVectors` attributes."""

        del self.vmulti
        del self.super_render
        self.kwargs.clear()
        self.temporal_vectors.clear()
        self._init_vects()


class MVToolsPlugin(CustomIntEnum):
    """Abstraction around the three versions of mvtools plugins that exist."""

    INTEGER = 0
    """Original plugin. Only accepts integer 8-16 bits clips."""

    FLOAT_OLD = 1
    """New plugin by IFeelBloated. Latest release. Only works with float single precision clips."""

    FLOAT_NEW = 2
    """Latest git master of :py:attr:`FLOAT_OLD`. Must be compiled by yourself."""

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
        """
        Automatically select the appropriate plugin for the specified clip.

        :param clip:    Clip you will use the plugin on.

        :return:        Correct MVToolsPlugin for the specified clip.
        """

        assert clip.format

        if clip.format.sample_type is vs.FLOAT:
            if not hasattr(core, 'mvsf'):
                raise ImportError(
                    "MVTools: With the current clip, the processing has to be done in float precision, "
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
    """
    SAD Calculation mode for MVTools.

    Decides the using of block pure Spatial Data, DCT, SAD, or SATD for SAD calculation.

    SAD => Sum of Absolute Difference (The main parameter mvtools uses).\n
    This is calculated over the 2 macroblocks that get compared.

    DCT => Discrete Cosine Transform (Frequency Spectrum).\n
    Transform both the current blocks and the reference block to frequency domain,
    then calculate the sum of the absolute difference between each pair of transformed samples in that domain.

    SATD => Sum of HAdamard Transformed Differences.\n
    Get the difference block between the current blocks and the reference block,
    and transform that difference block to frequency domain and calculate the sum of the absolute value
    of each sample in that transformed difference block.

    You can read more about the algorithm SATD uses `here <https://en.wikipedia.org/wiki/Hadamard_transform>`_.\n
    The actual implementation is a recursive Hadamard Ordered Walsh-Hadamard Transform.

    The usage of DCT in particular, can improve motion vector estimation in the event of luma flicker and fades.
    """

    SPATIAL = 0
    """Regular usage of spatial block data only, does not use DCT."""

    DCT = 1
    """Use block DCT instead of spatial data (slow for block size of 8x8 and very slow for smaller sizes)."""

    MIXED_SPATIAL_DCT = 2
    """Mixed spatial and DCT data; weight is dependent on mean frame luma difference."""

    ADAPTIVE_SPATIAL_MIXED = 3
    """Adaptive per-block switching from spatial to equal-weighted mixed mode."""

    ADAPTIVE_SPATIAL_DCT = 4
    """Adaptive per-block switching from spatial to mixed mode with more weight given to DCT."""

    SATD = 5
    """SATD instead of SAD for luma."""

    MIXED_SATD_DCT = 6
    """Same as 2, except use SATD instead of SAD."""

    ADAPTIVE_SATD_MIXED = 7
    """Same as 3, except use SATD instead of SAD."""

    ADAPTIVE_SATD_DCT = 8
    """Same as 4, except use SATD instead of SAD."""

    MIXED_SATEQSATD_DCT = 9
    """Similar to 2, use SATD and weight ranges from SAD only to equal SAD & SATD."""

    ADAPTIVE_SATD_MAJLUMA = 10
    """Similar to 3 and 4, use SATD weight is on SAD, only on strong luma changes."""

    @property
    def is_satd(self) -> bool:
        """Returns wether this SADMode uses SATD rather than SAD."""

        return self >= SADMode.SATD

    @property
    def same_recalc(self: SelfSADMode) -> tuple[SelfSADMode, SelfSADMode]:
        return (self, self)


SelfSADMode = TypeVar('SelfSADMode', bound=SADMode)


class MotionMode:
    """
    A preset or custom parameters values for truemotion/motion analysis modes of mvtools.

    Presets allows easy to switch default values of all "true motion" parameters at once.
    """

    @dataclass
    class Config:
        """Dataclass to represent all the "true motion" parameters."""

        truemotion: bool
        """Straight MVTools preset parameter."""

        coherence: int
        """
        Coherence of the field of vectors. The higher, the more coherent.

        However, if set too high, some best motion vectors can be missed.

        Values around 400 - 2000 (for block size 8) are strongly recommended.

        Internally it is coefficient for SAD penalty of vector squared
        difference from predictors (neighbors), scaled by 256.
        """

        sad_limit: int
        """
        SAD limit for coherence using.

        Local coherence is decreased if SAD value of vector predictor (formed from neighbor blocks)
        is greater than the limit. It prevents bad predictors using but decreases the motion coherence.

        Values above 1000 (for block size=8) are recommended for true motion.
        """

        pnew: int
        """
        Relative penalty (scaled to 256) to SAD cost for new candidate vector.

        New candidate vector will be accepted as new vector only if its SAD with penalty (SAD + SAD*pnew/256)
        is lower then predictor cost (old SAD).

        It prevent replacing of quite good predictors by new vector
        with a slightly better SAD but different length and direction.
        """

        plevel: int
        """
        Penalty factor coherence level scaling mode.
         * 0 - No scaling.
         * 1 - Linear.
         * 2 - Quadratic dependence from hierarchical level size.

        Note that vector length is smaller at lower level.
        """

        pglobal: bool
        """
        Relative penalty (scaled to 8 bit) to SAD cost for global predictor vector.\n
        Coherence is not used for global vector.
        """

    HIGH_SAD = Config(False, 0, 400, 0, 0, False)
    """Use to search motion vectors with best SAD."""

    VECT_COHERENCE = Config(True, 1000, 1200, 50, 1, True)
    """Use for true motion search (high vector coherence)."""

    VECT_NOSCALING = Config(True, 1000, 1200, 50, 0, True)
    """Same as :py:attr:`VECT_COHERENCE` but with plevel set to no scaling (lower penality factor)."""

    class _CustomConfig:
        def __call__(
            self, coherence: int | None = None, sad_limit: int | None = None,
            pnew: int | None = None, plevel: int | None = None, pglobal: bool | None = None,
            truemotion: bool = True
        ) -> MotionMode.Config:
            """
            Create a custom :py:class:`MotionMode.Config`.\n
            Default values will depend on `truemotion`.

            For parameters, please refer to :py:class:`MotionMode.Config`
            """

            ref = MotionMode.from_param(truemotion)

            return MotionMode.Config(
                truemotion,
                fallback(coherence, ref.coherence),
                fallback(sad_limit, ref.sad_limit),
                fallback(pnew, ref.pnew),
                fallback(plevel, ref.plevel),
                fallback(pglobal, ref.pglobal)
            )

    MANUAL = _CustomConfig()
    """Construct a custom config."""

    @classmethod
    def from_param(cls, truemotion: bool) -> Config:
        """
        Get a default :py:class:`MotionMode.Config`.

        :param truemotion:  Whether to have a true motion config or not.

        :return:            A :py:class:`MotionMode.Config`.
        """

        return MotionMode.VECT_COHERENCE if truemotion else MotionMode.HIGH_SAD


class SearchModeBase:
    @dataclass
    class Config:
        """Dataclass to represent all the search related parameters."""

        mode: SearchMode
        """SearchMode that decides which analysis mode to use for search of motion vectors."""

        recalc_mode: SearchMode
        """SearchMode that decides which analysis mode to use for recalculation of motion vectors."""

        param: int | None
        """Parameter used by the search mode in analysis."""

        param_recalc: int | None
        """Parameter used by the search mode in recalculation."""

        pel: int | None
        """Parameter used by search mode for subpixel accuracy."""


class SearchMode(SearchModeBase, CustomIntEnum):
    """Decides the type of search at every level of the hierarchial analysis made while searching for motion vectors."""

    AUTO = -1
    """Automatically select a SearchMode."""

    ONETIME = 0
    """One time search."""

    NSTEP = 1
    """N step searches. The most well-known of the MV search algorithms."""

    DIAMOND = 2
    """Logarithmic search, also known as Diamond Search."""

    HEXAGON = 4
    """Hexagon search (similar to x264's)."""

    UMH = 5
    """Uneven Multi Hexagon search (similar to x264's)."""

    EXHAUSTIVE = 3
    """Exhaustive search, square side is 2 * radius + 1. It's slow, but gives the best results SAD-wise."""

    EXHAUSTIVE_H = 6
    """Pure horizontal exhaustive search, width is 2 * radius + 1."""

    EXHAUSTIVE_V = 7
    """Pure vertical exhaustive search, height is 2 * radius + 1."""

    @overload
    def __call__(  # type: ignore
        self: Literal[ONETIME], step: int | tuple[int, int] | None = ..., pel: int | None = ...,
        recalc_mode: SearchMode = ..., /, **kwargs: Any
    ) -> SearchMode.Config:
        """
        Get the :py:class:`SearchMode.Config` from this mode and params.

        :param step:    Step between each vector tried. If > 1, step will be progressively refined.
        :param pel:     Search pixel enlargement, for subpixel precision.

        :return:        :py:class:`SearchMode.Config` from this mode, param and accuracy.
        """

    @overload
    def __call__(  # type: ignore
        self: Literal[NSTEP], times: int | tuple[int, int] | None = ..., pel: int | None = ...,
        recalc_mode: SearchMode = ..., /, **kwargs: Any
    ) -> SearchMode.Config:
        """
        Get the :py:class:`SearchMode.Config` from this mode and params.

        :param times:   Number of step for search.
        :param pel:     Search pixel enlargement, for subpixel precision.

        :return:        :py:class:`SearchMode.Config` from this mode, param and accuracy.
        """

    @overload
    def __call__(  # type: ignore
        self: Literal[DIAMOND], init_step: int | tuple[int, int] | None = ..., pel: int | None = ...,
        recalc_mode: SearchMode = ..., /, **kwargs: Any
    ) -> SearchMode.Config:
        """
        Get the :py:class:`SearchMode.Config` from this mode and params.

        :param init_step:   Initial step search, then refined progressively.
        :param pel:         Search pixel enlargement, for subpixel precision.

        :return:            :py:class:`SearchMode.Config` from this mode, param and accuracy.
        """

    @overload
    def __call__(  # type: ignore
        self: Literal[HEXAGON], range: int | tuple[int, int] | None = ..., pel: int | None = ...,
        recalc_mode: SearchMode = ..., /, **kwargs: Any
    ) -> SearchMode.Config:
        """
        Get the :py:class:`SearchMode.Config` from this mode and params.

        :param range:   Range of search.
        :param pel:     Search pixel enlargement, for subpixel precision.

        :return:        :py:class:`SearchMode.Config` from this mode, param and accuracy.
        """

    @overload
    def __call__(  # type: ignore
        self: Literal[UMH], range: int | tuple[int, int] | None = ..., pel: int | None = ...,
        recalc_mode: SearchMode = ..., /, **kwargs: Any
    ) -> SearchMode.Config:
        """
        Get the :py:class:`SearchMode.Config` from this mode and params.

        :param range:   Radius of the multi hexagonal search.
        :param pel:     Search pixel enlargement, for subpixel precision.

        :return:        :py:class:`SearchMode.Config` from this mode, param and accuracy.
        """

    @overload
    def __call__(  # type: ignore
        self: Literal[EXHAUSTIVE] | Literal[EXHAUSTIVE_H] | Literal[EXHAUSTIVE_V],
        radius: int | tuple[int, int] | None = ..., pel: int | None = ..., recalc_mode: SearchMode = ...,
        /, **kwargs: Any
    ) -> SearchMode.Config:
        """
        Get the :py:class:`SearchMode.Config` from this mode and params.

        :param radius:  Radius of the exhaustive (tesa) search.
        :param pel:     Search pixel enlargement, for subpixel precision.

        :return:        :py:class:`SearchMode.Config` from this mode, param and accuracy.
        """

    @overload
    def __call__(
        self, param: int | tuple[int, int] | None = ..., pel: int | None = ..., recalc_mode: SearchMode = ...,
        /, **kwargs: Any
    ) -> SearchMode.Config:
        """
        Get the :py:class:`SearchMode.Config` from this mode and params.

        :param param:   Parameter used by the search mode. Purpose depends on the mode.
        :param pel:     Search pixel enlargement, for subpixel precision.

        :return:        :py:class:`SearchMode.Config` from this mode, param and accuracy.
        """

    def __call__(
        self, param: int | tuple[int, int] | None | MissingT = MISSING, pel: int | None | MissingT = MISSING,
        recalc_mode: SearchMode | MissingT = MISSING, /, **kwargs: Any
    ) -> SearchMode.Config:
        """
        Get the :py:class:`SearchMode.Config` from this mode and params.

        :param step:    Parameter of the mvtools search mode.
        :param pel:     Search pixel enlargement, for subpixel precision.

        :return:        :py:class:`SearchMode.Config` from this mode, param and accuracy.
        """

        is_uhd = kwargs.get('is_uhd', False)
        refine = kwargs.get('refine', 3)
        truemotion = kwargs.get('truemotion', False)

        if self is SearchMode.AUTO:
            self = SearchMode.DIAMOND

        if recalc_mode is MISSING:
            recalc_mode = SearchMode.ONETIME

        param_recalc: int | MissingT | None

        if isinstance(param, int):
            param, param_recalc = param, MISSING
        elif isinstance(param, tuple):
            param, param_recalc = param
        else:
            param = param_recalc = param

        if param is MISSING:
            param = (2 if is_uhd else 5) if (refine and truemotion) else (1 if is_uhd else 2)

        param_c = fallback(param, 2)

        if param_recalc is MISSING:
            param_recalc = max(0, round(exp(0.69 * param_c - 1.79) - 0.67))

        if pel is MISSING:
            pel = min(8, max(0, param_c * 2 - 2))

        return SearchMode.Config(self, recalc_mode, param, param_recalc, pel)  # type: ignore

    @property
    def defaults(self) -> SearchMode.Config:
        return self(None, None, self)


if TYPE_CHECKING:
    class MVToolsPresetBase(dict[str, Any]):
        ...
else:
    MVToolsPresetBase = object


@dataclass(kw_only=True)
class MVToolsPreset(MVToolsPresetBase):
    """Base MVTools preset. Please refer to :py:class:`MVTools` documentation for members."""

    tr: property | int | None = None
    refine: property | int | None = None
    pel: property | int | None = None
    planes: property | int | Sequence[int] | None = None
    range_in: property | ColorRange | None = None
    source_type: property | FieldBasedT | None = None
    high_precision: property | bool = False
    hpad: property | int | None = None
    vpad: property | int | None = None
    params_curve: property | bool | None = None
    block_size: property | int | None = None
    overlap: property | int | None = None
    thSAD: property | int | None = None
    range_conversion: property | float | None = None
    search: property | SearchMode | SearchMode.Config | None = None
    sharp: property | int | None = None
    rfilter: property | int | None = None
    sad_mode: property | SADMode | tuple[SADMode, SADMode] | None = None
    motion: property | MotionMode.Config | None = None
    prefilter: property | Prefilter | vs.VideoNode | None = None
    pel_type: property | PelType | tuple[PelType, PelType] | None = None

    if TYPE_CHECKING:
        def __call__(
            self, *, tr: int | None = None, refine: int | None = None, pel: int | None = None,
            planes: int | Sequence[int] | None = None, range_in: ColorRange | None = None,
            source_type: FieldBasedT | None = None, high_precision: bool = False, hpad: int | None = None,
            vpad: int | None = None, params_curve: bool | None = None, block_size: int | None = None,
            overlap: int | None = None, thSAD: int | None = None, range_conversion: float | None = None,
            search: SearchMode | SearchMode.Config | None = None, motion: MotionMode.Config | None = None,
            sad_mode: SADMode | tuple[SADMode, SADMode] | None = None, rfilter: int | None = None,
            sharp: int | None = None, prefilter: Prefilter | vs.VideoNode | None = None,
            pel_type: PelType | tuple[PelType, PelType] | None = None
        ) -> MVToolsPreset:
            ...
    else:
        def __call__(self, **kwargs: Any) -> MVToolsPreset:
            return MVToolsPreset(**(dict(**self) | kwargs))

    def _get_dict(self) -> KwargsT:
        return KwargsNotNone(**{
            key: value.__get__(self) if isinstance(value, property) else value
            for key, value in (self._value_ if isinstance(self, CustomEnum) else self).__dict__.items()
        })

    def __getitem__(self, key: str) -> Any:
        return self._get_dict()[key]

    def __class_getitem__(cls, key: str) -> Any:
        return cls()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._get_dict().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._get_dict()

    def copy(self) -> MVToolsPreset:
        return MVToolsPreset(**self._get_dict())

    def keys(self) -> dict_keys[str, Any]:
        return self._get_dict().keys()

    def values(self) -> dict_values[str, Any]:
        return self._get_dict().values()

    def items(self) -> dict_items[str, Any]:
        return self._get_dict().items()

    def __eq__(self, other: Any) -> bool:
        return False

    @inject_self
    def as_dict(self) -> KwargsT:
        return KwargsT(**self._get_dict())


class MVToolsPresets:
    """Presets for MVTools analyzing/refining."""

    CUSTOM = MVToolsPreset
    """Create your own preset."""

    SMDE = MVToolsPreset(
        pel=2, prefilter=Prefilter.NONE, params_curve=False, sharp=2, rfilter=4,
        block_size=8, overlap=2, thSAD=300, sad_mode=SADMode.SPATIAL.same_recalc,
        motion=MotionMode.VECT_COHERENCE, search=SearchMode.HEXAGON.defaults,
        hpad=property(fget=lambda x: x.block_size), vpad=property(fget=lambda x: x.block_size),
        range_conversion=1.0
    )
    """SMDegrain by Caroliano & DogWay"""

    CMDE = MVToolsPreset(
        pel=1, prefilter=Prefilter.NONE, params_curve=False, sharp=2, rfilter=4,
        block_size=32, overlap=16, thSAD=200, sad_mode=SADMode.SPATIAL.same_recalc,
        motion=MotionMode.HIGH_SAD, search=SearchMode.HEXAGON.defaults,
        hpad=property(fget=lambda x: x.block_size), vpad=property(fget=lambda x: x.block_size),
        range_conversion=1.0
    )
    """CMDegrain from EoE."""


class MVTools:
    """MVTools wrapper for motion analysis / degrain / compensation"""

    super_args: dict[str, Any]
    """Arguments passed to all the :py:attr:`MVToolsPlugin.Super` calls."""

    analyze_args: dict[str, Any]
    """Arguments passed to all the :py:attr:`MVToolsPlugin.Analyze` calls."""

    recalculate_args: dict[str, Any]
    """Arguments passed to all the :py:attr:`MVToolsPlugin.Recalculate` calls."""

    compensate_args: dict[str, Any]
    """Arguments passed to all the :py:attr:`MVToolsPlugin.Compensate` calls."""

    vectors: MotionVectors
    """Motion vectors analyzed and used for all operations."""

    clip: vs.VideoNode
    """Clip to process."""

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
        params_curve: bool = True,
        *,
        # kwargs for mvtools calls
        super_args: dict[str, Any] | None = None,
        analyze_args: dict[str, Any] | None = None,
        recalculate_args: dict[str, Any] | None = None,
        compensate_args: dict[str, Any] | None = None,
        # analyze kwargs
        block_size: int | None = None, overlap: int | None = None,
        thSAD: int | None = None, range_conversion: float | None = None,
        search: SearchMode | SearchMode.Config | None = None,
        sharp: int | None = None, rfilter: int | None = None,
        sad_mode: SADMode | tuple[SADMode, SADMode] | None = None,
        motion: MotionMode.Config | None = None,
        prefilter: Prefilter | vs.VideoNode | None = None,
        pel_type: PelType | tuple[PelType, PelType] | None = None
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
                                    i.e. `refine=4` it will analyze at `block_size=32`, then refine at 16, 8, 4.
        :param pel:                 Pixel EnLargement value, a.k.a. subpixel accuracy of the motion estimation.\n
                                    Value can only be 1, 2 or 4.
                                     * 1 means a precision to the pixel.
                                     * 2 means a precision to half a pixel.
                                     * 4 means a precision to quarter a pixel.
                                    `pel=4` is produced by spatial interpolation which is more accurate,
                                    but slower and not always better due to big level scale step.
        :param planes:              Planes to process.
        :param range_in:            ColorRange of the input clip.
        :param source_type:         Source type of the input clip.
        :param high_precision:      Whether to process everything in float32 (very slow).
                                    If set to False, it will process it in the input clip's bitdepth.
        :param hpad:                Horizontal padding added to source frame (both left and right).\n
                                    Small padding is added for more correct motion estimation near frame borders.
        :param vpad:                Vertical padding added to source frame (both top and bottom).
        :param vectors:             Precalculated vectors, either a custom instance or another MVTools instance.
        :param params_curve:        Apply a curve to some parameters and apply a limit to Recalculate parameters.

        :param super_args:          Arguments passed to all the :py:attr:`MVToolsPlugin.Super` calls.
        :param analyze_args:        Arguments passed to all the :py:attr:`MVToolsPlugin.Analyze` calls.
        :param recalculate_args:    Arguments passed to all the :py:attr:`MVToolsPlugin.Recalculate` calls.
        :param compensate_args:     Arguments passed to all the :py:attr:`MVToolsPlugin.Compensate` calls.

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

        self.is_hd = clip.width >= 1100 or clip.height >= 600
        self.is_uhd = self.clip.width >= 2600 or self.clip.height >= 1500

        self.tr = tr

        self.refine = refine

        if self.refine > 6:
            raise CustomOverflowError(f'Refine > 6 is not supported! ({refine})', self.__class__)

        self.source_type = FieldBased.from_param(source_type, MVTools) or FieldBased.from_video(self.clip)
        self.range_in = range_in

        self.pel = fallback(pel, 1 + int(not self.is_hd))

        self.planes = normalize_planes(self.clip, planes)

        self.is_gray = self.planes == [0]

        self.mv_plane = planes_to_mvtools(self.planes)

        self.chroma = self.mv_plane != 0

        if isinstance(vectors, MVTools):
            self.vectors = vectors.vectors
        elif isinstance(vectors, MotionVectors):
            self.vectors = vectors
        else:
            self.vectors = MotionVectors()

        self.params_curve = params_curve

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
            block_size=block_size, sad_mode=sad_mode, motion=motion, prefilter=prefilter, pel_type=pel_type,
            thSAD=thSAD
        )

    def analyze(
        self,
        block_size: int | None = None, overlap: int | None = None,
        thSAD: int | None = None, range_conversion: float | None = None,
        search: SearchMode | SearchMode.Config | None = None,
        sharp: int | None = None, rfilter: int | None = None,
        sad_mode: SADMode | tuple[SADMode, SADMode] | None = None,
        motion: MotionMode.Config | None = None,
        prefilter: Prefilter | vs.VideoNode | None = None,
        pel_type: PelType | tuple[PelType, PelType] | None = None,
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
        :param ref:                 Reference clip to use for analyzes over the main clip.
        :param inplace:             Whether to save the analysis in the MVTools instance or not.

        :return:                    :py:class:`MotionVectors` object with the analyzed motion vectors.
        """

        ref = self.get_ref_clip(ref, self.__class__.analyze)

        block_size = kwargs_fallback(block_size, (self.analyze_func_kwargs, 'block_size'), 16 if self.is_hd else 8)
        blocksize = max(self.refine and 2 ** (self.refine + 1), block_size)

        halfblocksize = max(2, blocksize // 2)
        halfoverlap = max(2, halfblocksize // 2)

        overlap = kwargs_fallback(overlap, (self.analyze_func_kwargs, 'overlap'), halfblocksize)

        rfilter = kwargs_fallback(rfilter, (self.analyze_func_kwargs, 'rfilter'), 3)

        thSAD = kwargs_fallback(thSAD, (self.analyze_func_kwargs, 'thSAD'), 300)

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
        super_recalc = self.mvtools.Super(
            prefilter, **(dict(levels=1) | common_args)
        ) if self.refine else super_render

        if self.params_curve:
            thSAD_recalc = round(exp(-101. / (thSAD * 0.83)) * 360)
        else:
            thSAD_recalc = thSAD

        t2 = (self.tr * 2 if self.tr > 1 else self.tr) if self.source_type.is_inter else self.tr

        analyze_args = dict[str, Any](
            dct=sad_mode, pelsearch=search.pel, blksize=blocksize, overlap=overlap, search=search.mode,
            truemotion=motion.truemotion, searchparam=search.param, chroma=self.chroma,
            plevel=motion.plevel, pglobal=motion.pglobal
        ) | self.analyze_args

        recalc_args = dict[str, Any](
            search=search.recalc_mode, dct=recalc_sad_mode, thsad=thSAD_recalc, blksize=halfblocksize,
            overlap=halfoverlap, truemotion=motion.truemotion, searchparam=search.param_recalc,
            chroma=self.chroma
        ) | self.recalculate_args

        if self.mvtools is MVToolsPlugin.FLOAT_NEW:
            vmulti = self.mvtools.Analyse(super_search, radius=t2, **analyze_args)

            vectors.vmulti = vmulti

            for i in range(self.refine):
                recalc_args.update(blksize=blocksize / 2 ** i, overlap=blocksize / 2 ** (i + 1))
                vectors.vmulti = self.mvtools.Recalculate(super_recalc, vectors.vmulti, **recalc_args)
        else:
            def _add_vector(delta: int, analyze: bool = True) -> None:
                for direction in MVDirection:
                    if analyze:
                        vect = self.mvtools.Analyse(super_search, isb=direction.isb, delta=delta, **analyze_args)
                    else:
                        vect = self.mvtools.Recalculate(super_recalc, vectors.get_mv(direction, delta), **recalc_args)

                    vectors.set_mv(direction, delta, vect)

            for i in range(1, t2 + 1):
                _add_vector(i)

            if self.refine:
                for i in range(1, t2 + 1):
                    if not vectors.got_mv(MVDirection.BACK, i) or not vectors.got_mv(MVDirection.FWRD, i):
                        continue

                    for j in range(0, self.refine):
                        val = clamp(blocksize / 2 ** j, 4, 128)

                        recalc_args.update(blksize=val, overlap=val / 2)

                        _add_vector(i, False)

        vectors.super_render = super_render
        vectors.kwargs.update(thSAD=thSAD)

        return vectors

    def get_vectors_bf(self, *, inplace: bool = False) -> tuple[list[vs.VideoNode], list[vs.VideoNode]]:
        """
        Get the backwards and forward vectors.\n

        If :py:attr:`analyze` wasn't previously called,
        it will do so here with default values or kwargs specified in the constructor.

        :param inplace:     Only return the list, not modifying the internal state.\n
                            (Useful if you haven't called :py:attr:`analyze` previously)

        :return:            Two lists, respectively for backward and forwards, containing motion vectors.
        """

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
                vectors_backward.append(vectors.get_mv(MVDirection.BACK, i))
                vectors_forward.append(vectors.get_mv(MVDirection.FWRD, i))

        return (vectors_backward, vectors_forward)

    def compensate(
        self, func: GenericVSFunction, thSAD: int = 150, *, ref: vs.VideoNode | None = None, **kwargs: Any
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

        :param func:        Temporal function to motion compensate.
        :param thSAD:       This is the SAD threshold for safe (dummy) compensation.\n
                            If block SAD is above thSAD, the block is bad, and we use source block
                            instead of the compensated block.
        :param ref:         Reference clip to use instead of main clip.
        :param kwargs:      Keyword arguments passed to `func` to avoid using `partial`.

        :return:            Motion compensated output of `func`.
        """

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
        self,
        thSAD: int | tuple[int | None, int | None] | None = None,
        limit: int | tuple[int, int] = 255,
        thSCD: int | tuple[int | None, int | None] | None = (None, 51),
        *, ref: vs.VideoNode | None = None
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
                        Value ranges from 0 to 255. At 255, no pixel may be adjusted,
                        effectively preventing any degraining from occuring.
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

        :return:        Degrained clip.
        """

        ref = self.get_ref_clip(ref, self.__class__.degrain)

        vect_b, vect_f = self.get_vectors_bf()

        thSAD, thSADC = (thSAD if isinstance(thSAD, tuple) else (thSAD, None))

        thSAD = kwargs_fallback(thSAD, (self.vectors.kwargs, 'thSAD'), 300)
        thSADC = fallback(thSADC, round(thSAD * 0.18875 * exp(2 * 0.693)) if self.params_curve else thSAD // 2)

        limit, limitC = normalize_seq(limit, 2)

        if not all(0 <= x <= 255 for x in (limit, limitC)):
            raise CustomOverflowError(
                '"limit" values should be between 0 and 255 (inclusive)!', self.__class__.degrain
            )

        limitf, limitCf = scale_value(limit, 8, ref), scale_value(limitC, 8, ref)

        thSCD1, thSCD2 = thSCD if isinstance(thSCD, tuple) else (thSCD, None)

        thSCD1 = fallback(thSCD1, round(0.35 * thSAD + 300) if self.params_curve else 400)
        thSCD2 = fallback(thSCD2, 51)

        if not 1 <= thSCD2 <= 100:
            raise CustomOverflowError(
                '"thSCD[1]" must be between 1 and 100 (inclusive)!', self.__class__.degrain
            )

        thSCD2 = int(thSCD2 / 100 * 255)

        degrain_args = dict[str, Any](thscd1=thSCD1, thscd2=thSCD2, plane=self.mv_plane)

        if self.mvtools is MVToolsPlugin.INTEGER:
            degrain_args.update(thsad=thSAD, thsadc=thSADC, limit=limitf, limitc=limitCf)
        else:
            degrain_args.update(thsad=[thSAD, thSADC, thSADC], limit=[limitf, limitCf])

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
