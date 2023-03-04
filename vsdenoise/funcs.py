"""
This module contains general denoising functions built on top of base denoisers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import count, zip_longest
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, cast, overload

from vsexprtools import norm_expr
from vskernels import Bilinear, Catrom, Scaler, ScalerT
from vsrgtools import RemoveGrainMode, contrasharpening, contrasharpening_dehalo, removegrain
from vsrgtools.util import norm_rmode_planes
from vstools import (
    CustomIntEnum, FuncExceptT, FunctionUtil, KwargsT, P, PlanesT, VSFunction, depth, expect_bits, fallback, get_h,
    get_w, normalize_planes, vs
)

from .fft import DFTTest, fft3d
from .knlm import nl_means
from .mvtools import MotionMode, MotionVectors, MVTools, MVToolsPreset, MVToolsPresets, SADMode, SearchMode
from .mvtools.enums import SearchModeBase
from .prefilters import PelType, Prefilter

__all__ = [
    'mlm_degrain',

    'temporal_degrain',

    'PostProcessFFT', 'TemporalLimiter'
]


def mlm_degrain(
    clip: vs.VideoNode, tr: int = 3, refine: int = 3, thSAD: int | tuple[int, int] = 200,
    factors: Iterable[float] | range = [1 / 3, 2 / 3],
    scaler: ScalerT = Bilinear, downscaler: ScalerT = Catrom,
    mv_kwargs: KwargsT | list[KwargsT] | None = None,
    analyze_kwargs: KwargsT | list[KwargsT] | None = None,
    degrain_kwargs: KwargsT | list[KwargsT] | None = None,
    soften: Callable[..., vs.VideoNode] | bool | None = False,
    planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode:
    """
    Multi Level scaling Motion compensated Degrain. Original idea by Didée.

    The observation was that when downscaling the source to a smaller resolution,
    a normal MVTools Degrain pass can produce a much stabler result.

    The approach taken here is to first make a small-but-stable-denoised clip,
    then work our way upwards to the original resolution, averaging their differences.

    :param clip:                Clip to be denoised.
    :param tr:                  Temporal radius of the denoising.
    :param refine:              Refine param of :py:class:`MVTools`.
    :param thSAD:               thSAD param of :py:attr:`MVTools.analyze`/:py:attr:`MVTools.degrain`.
    :param factors:             Scaling factors.
                                 * If floats, they will be interpreted as size * factor.
                                 * If a range, it will first be normalized as a list of float with 1 / factor.
    :param scaler:              Scaler to use for scaling the downscaled clips up when diffing them.
    :param downscaler:          Scaler to use for downscaling the clip to various levels.
    :param mv_kwargs:           Keyword arguments to pass to :py:class:`MVTools`.
    :param analyze_kwargs:      Keyword arguments to pass to :py:attr:`MVTools.analyze`.
    :param degrain_kwargs:      Keyword arguments to pass to :py:attr:`MVTools.degrain`.
    :param soften:              Use a softening function to sharpen the output; recommended only for live content.
    :param planes:              Planes to process.

    :return:                    Denoised clip.
    """

    planes = normalize_planes(clip, planes)

    scaler = Scaler.ensure_obj(scaler, mlm_degrain)
    downscaler = Scaler.ensure_obj(downscaler, mlm_degrain)

    do_soft = bool(soften)

    if isinstance(thSAD, tuple):
        thSADA, thSADD = thSAD
    else:
        thSADA = thSADD = thSAD

    mkwargs_def = dict[str, Any](tr=tr, refine=refine, planes=planes)
    akwargs_def = dict[str, Any](motion=MotionMode.HIGH_SAD, thSAD=thSADA, pel_type=PelType.WIENER)
    dkwargs_def = dict[str, Any](thSAD=thSADD)

    mkwargs, akwargs, dkwargs = [
        [default] if kwargs is None else [
            (default | val) for val in
            (kwargs if isinstance(kwargs, list) else [kwargs])
        ] for default, kwargs in (
            (mkwargs_def, mv_kwargs),
            (akwargs_def, analyze_kwargs),
            (dkwargs_def, degrain_kwargs)
        )
    ]

    if isinstance(factors, range):
        factors = [1 / x for x in factors if x >= 1]
    else:
        factors = list(factors)

    mkwargs_fact, akwargs_fact, dkwargs_fact = [
        cast(
            list[tuple[float, dict[str, Any]]],
            list(zip_longest(
                factors, kwargs[:len(factors)], fillvalue=kwargs[-1]
            ))
        ) for kwargs in (mkwargs, akwargs, dkwargs)
    ]

    factors = set(sorted(factors)) - {0, 1}

    norm_mkwargs, norm_akwargs, norm_dkwargs = [
        [
            next(x[1] for x in kwargs if x[0] == factor)
            for factor in factors
        ] for kwargs in (mkwargs_fact, akwargs_fact, dkwargs_fact)
    ]

    norm_mkwargs, norm_akwargs, norm_dkwargs = [
        norm_mkwargs + norm_mkwargs[-1:], norm_akwargs + norm_akwargs[-1:], norm_dkwargs + norm_dkwargs[-1:]
    ]

    def _degrain(clip: vs.VideoNode, ref: vs.VideoNode | None, idx: int) -> vs.VideoNode:
        mvtools_arg = dict(**norm_mkwargs[idx])

        if do_soft and idx in {0, last_idx}:
            if soften is True:
                softened = removegrain(clip, norm_rmode_planes(
                    clip, RemoveGrainMode.SQUARE_BLUR if clip.width < 1200 else RemoveGrainMode.BOX_BLUR, planes
                ))
            elif callable(soften):
                try:
                    softened = soften(clip, planes=planes)
                except BaseException:
                    softened = soften(clip)

            mvtools_arg |= dict(prefilter=softened)

        mvtools_arg |= dict(
            pel=2 if idx == 0 else 1, block_size=16 if clip.width > 960 else 8
        ) | norm_akwargs[idx] | kwargs

        mv = MVTools(clip, **mvtools_arg)
        mv.analyze(ref=ref)
        return mv.degrain(**norm_dkwargs[idx])

    clip, bits = expect_bits(clip, 16)
    resolutions = [
        (get_w(clip.height * factor, clip), get_h(clip.width * factor, clip))
        for factor in factors
    ]

    scaled_clips = [clip]
    for width, height in resolutions[::-1]:
        scaled_clips.insert(0, downscaler.scale(scaled_clips[0], width, height))

    diffed_clips = [
        scaler.scale(clip, nclip.width, nclip.height).std.MakeDiff(nclip)
        for clip, nclip in zip(scaled_clips[:-1], scaled_clips[1:])
    ]

    last_idx = len(diffed_clips)

    new_resolutions = [(c.width, c.height) for c in diffed_clips]

    base_denoise = _degrain(scaled_clips[0], None, 0)
    ref_den_clips = [base_denoise]
    for width, height in new_resolutions:
        ref_den_clips.append(scaler.scale(ref_den_clips[-1], width, height))

    ref_denoise = ref_den_clips[1]

    for i, diff, ref_den, ref_den_next in zip(
        count(1), diffed_clips, ref_den_clips[1:], ref_den_clips[2:] + ref_den_clips[-1:]
    ):
        ref_denoise = ref_denoise.std.MakeDiff(_degrain(diff, ref_den, i))

        if not i == last_idx:
            ref_denoise = scaler.scale(ref_denoise, ref_den_next.width, ref_den_next.height)

    return depth(ref_denoise, bits)


@dataclass
class PostProcessConfig:
    mode: PostProcessFFT
    kwargs: KwargsT

    _sigma: float | None = None
    _tr: int | None = None
    _block_size: int | None = None
    merge_strength: int = 0

    @property
    def sigma(self) -> float:
        sigma = fallback(self._sigma, 1.0)

        if self.mode is PostProcessFFT.DFTTEST:
            return sigma * 4

        if self.mode is PostProcessFFT.NL_MEANS:
            return sigma / 2

        return sigma

    @property
    def tr(self) -> int:
        if self.mode <= 0:
            return 0

        tr = fallback(self._tr, 1)

        if self.mode is PostProcessFFT.DFTTEST:
            return min(tr, 3)

        if self.mode in {PostProcessFFT.FFT3D_MED, PostProcessFFT.FFT3D_HIGH}:
            return min(tr, 2)

        return tr

    @property
    def block_size(self) -> int:
        if self.mode is PostProcessFFT.DFTTEST:
            from .fft import BackendInfo

            backend_info = BackendInfo.from_param(self.kwargs.pop('plugin', DFTTest.Backend.AUTO))

            if backend_info.resolved_backend.is_dfttest2:
                return 16

        return fallback(self._block_size, [0, 48, 32, 12, 0][self.mode.value])

    def __call__(self, clip: vs.VideoNode, planes: PlanesT = None, func: FuncExceptT | None = None) -> vs.VideoNode:
        func = func or self.__class__

        if self.mode is PostProcessFFT.REPAIR:
            return removegrain(clip, norm_rmode_planes(clip, RemoveGrainMode.MINMAX_AROUND1, planes))

        if self.mode in {PostProcessFFT.FFT3D_MED, PostProcessFFT.FFT3D_HIGH}:
            return fft3d(clip, func, bw=self.block_size, bh=self.block_size, bt=self.tr * 2 + 1, **self.kwargs)

        if self.mode is PostProcessFFT.DFTTEST:
            return DFTTest.denoise(
                clip, self.sigma, tr=self.tr, block_size=self.block_size,
                planes=planes, **(KwargsT(overlap=int(self.block_size * 9 / 12)) | self.kwargs)  # type: ignore
            )

        if self.mode is PostProcessFFT.NL_MEANS:
            return nl_means(
                clip, self.sigma, self.tr, planes=planes, **(KwargsT(sr=2) | self.kwargs)  # type: ignore
            )

        return clip


class PostProcessFFT(CustomIntEnum):
    REPAIR = 0
    FFT3D_HIGH = 1
    FFT3D_MED = 2
    DFTTEST = 3
    NL_MEANS = 4

    if TYPE_CHECKING:
        from .funcs import PostProcessFFT

        @overload
        def __call__(  # type: ignore
            self: Literal[PostProcessFFT.REPAIR], *, merge_strength: int = 0
        ) -> PostProcessConfig:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[PostProcessFFT.NL_MEANS], *, sigma: float = 1.0, tr: int | None = None,
            merge_strength: int = 0, **kwargs: Any
        ) -> PostProcessConfig:
            ...

        @overload
        def __call__(
            self, *, sigma: float = 1.0, tr: int | None = None, block_size: int | None = None,
            merge_strength: int = 0, **kwargs: Any
        ) -> PostProcessConfig:
            ...

        def __call__(
            self, *, sigma: float = 1.0, tr: int | None = None, block_size: int | None = None,
            merge_strength: int = 0, **kwargs: Any
        ) -> PostProcessConfig:
            ...
    else:
        def __call__(
            self, *, sigma: float = 1.0, tr: int | None = None, block_size: int | None = None,
            merge_strength: int = 0, **kwargs: Any
        ) -> PostProcessConfig:
            return PostProcessConfig(self, kwargs, sigma, tr, block_size, merge_strength)


@dataclass
class TemporalLimiterConfig:
    limiter: VSFunction

    def __call__(
        self, clip: vs.VideoNode,
        thSAD1: int | tuple[int, int], thSAD2: int | tuple[int, int],
        thSCD: int | tuple[int | None, int | None] | None = None, mv_planes: PlanesT = None,
        vectors: MotionVectors | None = None, preset: MVToolsPreset = MVToolsPresets.SMDE
    ) -> vs.VideoNode:
        preset = preset(planes=mv_planes)

        NR1 = MVTools(clip, vectors=vectors, **preset).degrain(thSAD=thSAD1, thSCD=thSCD)

        NR1x = norm_expr([clip, self.limiter(clip), NR1], 'x y - abs x z - abs < y z ?', 0)

        return MVTools(NR1x, vectors=vectors, **preset).degrain(thSAD=thSAD2, thSCD=thSCD)


class TemporalLimiter(CustomIntEnum):
    CUSTOM = -1
    FFT3D = 0

    if TYPE_CHECKING:
        from .funcs import TemporalLimiter

        @overload
        def __call__(  # type: ignore
            self: Literal[TemporalLimiter.CUSTOM],
            limiter: Callable[P, vs.VideoNode], /, *args: P.args, **kwargs: P.kwargs
        ) -> TemporalLimiterConfig:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[TemporalLimiter.CUSTOM], limiter: vs.VideoNode, /,
        ) -> TemporalLimiterConfig:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[TemporalLimiter.FFT3D], *, sigma: float, block_size: int, ov: int
        ) -> TemporalLimiterConfig:
            ...

        def __call__(self, *args: Any, **kwargs: Any) -> TemporalLimiterConfig:  # type: ignore
            ...
    else:
        def __call__(
            self, limiter: vs.VideoNode | VSFunction | None = None, *args: Any, **kwargs: Any
        ) -> TemporalLimiterConfig:
            if self is self.CUSTOM:
                if isinstance(limiter, vs.VideoNode):
                    def limit_func(clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
                        return limiter
                else:
                    if limiter is None:
                        def _limiter(clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
                            return clip

                        limiter = _limiter

                    limit_func = partial(limiter, *args, **kwargs)
            elif self is self.FFT3D:
                sigma = kwargs.get('sigma')
                block_size = kwargs.get('block_size')
                ov = kwargs.get('ov')

                def limit_func(clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
                    return fft3d(
                        clip, sigma=sigma, sigma2=sigma * 0.625,
                        sigma3=sigma * 0.375, sigma4=sigma * 0.250,
                        bt=3, bw=block_size, bh=block_size, ow=ov, oh=ov,
                        **kwargs
                    )

            return TemporalLimiterConfig(limit_func)


def temporal_degrain(
    clip: vs.VideoNode, tr: int = 1, grain_level: int = 2,
    post: PostProcessFFT | PostProcessConfig = PostProcessFFT.REPAIR,
    limiter: TemporalLimiter | TemporalLimiterConfig = TemporalLimiter.FFT3D,
    block_size: int | None = None, refine: int = 0,
    thSAD1: int | None = None, thSAD2: int | None = None,
    thSCD1: int | None = None, thSCD2: int = 50,
    search_mode: SearchMode | SearchMode.Config = SearchMode.HEXAGON,
    sad_mode: SADMode | tuple[SADMode, SADMode] = SADMode.SPATIAL.same_recalc,
    prefilter: Prefilter | vs.VideoNode | None = None,
    truemotion: bool = False, global_motion: bool = True, chroma_motion: bool = True,
    contra: bool | int | float = False, planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode:
    func = FunctionUtil(clip, temporal_degrain, planes, (vs.GRAY, vs.YUV))

    if func.luma_only:
        chroma_motion = False
        planes = [0]

    longlat = max(func.clip.width, func.clip.height)
    shortlat = min(func.clip.width, func.clip.height)

    grain_level += 2

    if (longlat <= 1050 and shortlat <= 576):
        autoTune = 0
    elif (longlat <= 1280 and shortlat <= 720):
        autoTune = 1
    elif (longlat <= 2048 and shortlat <= 1152):
        autoTune = 2
    else:
        autoTune = 3

    postConf = post if isinstance(post, PostProcessConfig) else post()

    maxTR = max(tr, postConf.tr)

    prefilter = fallback(
        prefilter, [*([Prefilter.NONE] * 3), *([Prefilter.GAUSSBLUR2] * 3)][grain_level]  # type: ignore
    )

    if not isinstance(search_mode, SearchModeBase.Config):
        meAlgPar = 5 if refine and truemotion else 2
        meSubpel = [4, 2, 2, 1][autoTune]

        search_mode = search_mode(meAlgPar, meSubpel, search_mode)

    block_size = fallback(block_size, [8, 8, 16, 32][autoTune])
    hpad = vpad = block_size

    motion_lambda = (1000 if truemotion else 100) * (block_size ** 2) // 64

    thSAD1f, thSAD2f, thSCD1f = [
        d * 64.0 if x is None else float(x)
        for x, d in zip((thSAD1, thSAD2, thSCD1), list[tuple[int, int, int]]([
            (3, 2, 3), (5, 4, 3), (7, 5, 3), (9, 6, 4), (11, 7, 5), (13, 8, 6)
        ])[grain_level])
    ]

    if (sad_mode[0] if isinstance(sad_mode, tuple) else sad_mode) is SADMode.SATD:
        thSAD1f, thSAD2f = thSAD2f * 1.7, thSAD2f * 1.7

    thSAD1, thSAD2, thSCD1 = [int(x) for x in (thSAD1f, thSAD2f, thSCD1f)]

    if isinstance(limiter, TemporalLimiterConfig):
        limitConf = limiter
    elif limiter is TemporalLimiter.FFT3D:
        limitVal = [6, 8, 12, 16, 32, 48][[-1, -1, 0, 0, 0, 1][grain_level] + autoTune + 1]
        ov = 2 * round(limitVal * 2 / [4, 4, 4, 3, 2, 2][grain_level] * 0.5)

        limitConf = limiter(sigma=limitVal, block_size=limitVal * 2, ov=ov)  # type: ignore
    else:
        limitConf = limiter()  # type: ignore

    class MotionModeCustom(MotionMode.Config):
        def block_coherence(self, block_size: int) -> int:
            return motion_lambda // 4

    motion_ref = MotionMode.from_param(truemotion)

    preset = MVToolsPresets.CUSTOM(
        tr=tr, refine=refine, prefilter=prefilter(clip) if isinstance(prefilter, Prefilter) else prefilter,
        pel=meSubpel, hpad=hpad, vpad=vpad, block_size=block_size,
        overlap=property(lambda self: self.block_size // 2), search=search_mode,
        motion=MotionModeCustom(
            truemotion, motion_lambda, motion_ref.sad_limit, 50 if truemotion else 25, motion_ref.plevel, global_motion
        ), super_args=dict(chroma=chroma_motion), analyze_args=dict(chroma=chroma_motion),
        recalculate_args=dict(thsad=thSAD1 // 2), planes=planes
    )

    maxMV = MVTools(clip, **preset(tr=maxTR, **kwargs))
    maxMV.analyze()

    NR2 = limitConf(
        clip, thSAD1, thSAD2, (thSCD1, thSCD2),
        list(set(func.norm_planes + [1, 2])) if chroma_motion else func.norm_planes, maxMV.vectors, preset
    ) if tr > 0 else clip

    if postConf.tr > 0:
        dnWindow = MVTools(NR2, vectors=maxMV, **preset(tr=postConf.tr)).compensate(
            postConf, thSAD=thSAD2, thSCD=(thSCD1, thSCD2)
        )
    else:
        dnWindow = postConf(NR2)

    if contra:
        if contra is True:
            contra = 3

        if isinstance(contra, int):
            sharpened = contrasharpening(dnWindow, clip, contra, 13, planes)
        else:
            sharpened = contrasharpening_dehalo(dnWindow, clip, contra, planes=planes)
    else:
        sharpened = dnWindow

    if postConf.tr > 0 and postConf.merge_strength:
        sharpened = clip.std.Merge(sharpened, postConf.merge_strength / 100)

    return sharpened
