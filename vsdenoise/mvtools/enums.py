from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Literal, TypeVar, cast, overload

from vstools import (
    MISSING, CustomIntEnum, CustomStrEnum, CustomValueError, MissingT, VSFunctionAllArgs, VSFunctionKwArgs, core,
    fallback, vs
)

__all__ = [
    'MVDirection',

    'MVToolsPlugin',

    'SADMode', 'SearchMode', 'MotionMode', 'FlowMode', 'FinestMode'
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
    def Super(self) -> VSFunctionKwArgs:
        return cast(VSFunctionKwArgs, self.namespace.Super)

    @property
    def Analyse(self) -> VSFunctionKwArgs:
        return cast(
            VSFunctionKwArgs, self.namespace.Analyze if self is MVToolsPlugin.FLOAT_NEW else self.namespace.Analyse
        )

    @property
    def Recalculate(self) -> VSFunctionAllArgs:
        return cast(VSFunctionAllArgs, self.namespace.Recalculate)

    @property
    def Compensate(self) -> VSFunctionKwArgs:
        return cast(VSFunctionKwArgs, self.namespace.Compensate)

    @property
    def Flow(self) -> VSFunctionAllArgs:
        return cast(VSFunctionAllArgs, self.namespace.Flow)

    @property
    def FlowInter(self) -> VSFunctionAllArgs:
        return cast(VSFunctionAllArgs, self.namespace.FlowInter)

    @property
    def FlowBlur(self) -> VSFunctionAllArgs:
        return cast(VSFunctionAllArgs, self.namespace.FlowBlur)

    @property
    def FlowFPS(self) -> VSFunctionAllArgs:
        return cast(VSFunctionAllArgs, self.namespace.FlowFPS)

    @property
    def BlockFPS(self) -> VSFunctionAllArgs:
        return cast(VSFunctionAllArgs, self.namespace.BlockFPS)

    @property
    def Mask(self) -> VSFunctionAllArgs:
        return cast(VSFunctionAllArgs, self.namespace.Mask)

    @property
    def SCDetection(self) -> VSFunctionAllArgs:
        return cast(VSFunctionAllArgs, self.namespace.SCDetection)

    @property
    def Finest(self) -> VSFunctionAllArgs:
        return cast(VSFunctionAllArgs, self.namespace.Finest)

    def Degrain(self, radius: int | None = None) -> VSFunctionAllArgs:
        if radius is None and self is not MVToolsPlugin.FLOAT_NEW:
            raise CustomValueError('This implementation needs a radius!', f'{self.name}.Degrain')

        if radius is not None and radius > 24 and self is not MVToolsPlugin.FLOAT_NEW:
            raise ImportError(
                f"{self.name}.Degrain: With the current settings, temporal radius > 24, you're gonna need the latest "
                "master of mvsf and you're using an older version."
                "\n\tPlease build it from: https://github.com/IFeelBloated/vapoursynth-mvtools-sf"
            )

        try:
            return cast(VSFunctionAllArgs, getattr(self.namespace, f"Degrain{fallback(radius, '')}"))
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

    SATD => Sum of Hadamard Transformed Differences.\n
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
        """Returns whether this SADMode uses SATD rather than SAD."""

        return self >= SADMode.SATD

    @property
    def same_recalc(self: SelfSADMode) -> tuple[SelfSADMode, SelfSADMode]:
        return (self, self)


SelfSADMode = TypeVar('SelfSADMode', bound=SADMode)


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

        This is the "lambda" parameter of MVTools.
        """

        sad_limit: int
        """
        SAD limit for coherence using.

        Local coherence is decreased if SAD value of vector predictor (formed from neighbor blocks)
        is greater than the limit. It prevents bad predictors using but decreases the motion coherence.

        Values above 1000 (for block size=8) are recommended for true motion.

        This is the "lsad" parameter of MVTools.
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

        pglobal: int
        """
        Relative penalty (scaled to 8 bit) to SAD cost for global predictor vector.\n
        Coherence is not used for global vector.
        """

        def block_coherence(self, block_size: int) -> int:
            """Method to calculate coherence (lambda) based on blocksize."""
            return (self.coherence * block_size ** 2) // 64

    HIGH_SAD = Config(False, 0, 400, 0, 0, False)
    """Use to search motion vectors with best SAD."""

    VECT_COHERENCE = Config(True, 1000, 1200, 50, 1, True)
    """Use for true motion search (high vector coherence)."""

    VECT_NOSCALING = Config(True, 1000, 1200, 50, 0, True)
    """Same as :py:attr:`VECT_COHERENCE` but with plevel set to no scaling (lower penalty factor)."""

    class _CustomConfig:
        def __call__(
            self, coherence: int | None = None, sad_limit: int | None = None,
            pnew: int | None = None, plevel: int | None = None, pglobal: int | None = None,
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


class FlowMode(CustomIntEnum):
    """Decide where from the pixels be taken from the two frames when calculating the "flow" of the vector."""

    ABSOLUTE = 0
    """Fetch pixels to every place of destination."""

    RELATIVE = 1
    """Shift pixels from every place of source"""


class FinestMode(CustomIntEnum):
    """Decide when to calculate Finest type of a vector"""

    NONE = 0
    """Disabled"""

    ANALYZE = 1
    """After Analyze"""

    RECALCULATE = 2
    """After recalculation"""

    BOTH = 3
    """Every step"""

    @property
    def after_analyze(self) -> bool:
        return self in {self.ANALYZE, self.BOTH}

    @property
    def after_recalculate(self) -> bool:
        return self in {self.RECALCULATE, self.BOTH}
