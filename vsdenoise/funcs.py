"""
This module contains general denoising functions built on top of base denoisers.
"""

from __future__ import annotations

from functools import partial
from itertools import count, zip_longest
from typing import Any, Callable, Iterable, cast

from vskernels import Bilinear, Catrom, Scaler, ScalerT
from vsrgtools import RemoveGrainMode, contrasharpening, removegrain
from vsrgtools.util import norm_rmode_planes
from vstools import (
    KwargsT, PlanesT, VSFunction, core, depth, expect_bits, fallback, get_color_family, get_h, get_neutral_value,
    get_sample_type, get_w, normalize_planes, vs
)

from .fft import DFTTest, fft3d
from .knlm import nl_means
from .mvtools import MotionMode, MVTools, MVToolsPresets, SADMode, SearchMode
from .prefilters import PelType, Prefilter

__all__ = [
    'mlm_degrain',

    'temporal_degrain'
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


def temporal_degrain(
    clip: vs.VideoNode, /,
    tr: int = 1, grainLevel: int = 2,
    postFFT: int | VSFunction = 0, postSigma: int = 1,
    planes: int | list[int] = 4, *,
    grainLevelSetup: bool = False,
    outputStage: int = 2,
    meAlg: int = 4,
    meAlgPar: int | None = None,
    meSubpel: int | None = None,
    meBlksz: int | None = None,
    meTM: bool = False,
    ppSAD1: int | None = None,
    ppSAD2: int | None = None,
    ppSCD1: int | None = None,
    thSCD2: int = 50,
    DCT: int = 0,
    SubPelInterp: int = 2,
    SrchClipPP: int | Prefilter | vs.VideoNode | None = None,
    GlobalMotion: bool = True,
    ChromaMotion: bool = True,
    refine: bool | int = False,
    limiter: Prefilter | VSFunction | vs.VideoNode | None = None,
    limitSigma: int | None = None,
    limitBlksz: int | None = None,
    gpuId: int | None = 0,
    postTR: int = 1,
    postMix: int = 0,
    postBlkSize: int | None = None,
    extraSharp: bool | int = False,
    fftThreads: int = 1
) -> vs.VideoNode:
    width = clip.width
    height = clip.height

    neutral = get_neutral_value(clip)
    isFLOAT = get_sample_type(clip) == vs.FLOAT
    isGRAY = get_color_family(clip) == vs.GRAY

    if isinstance(planes, int):
        # Convert int-based plane selection to array-based plane selection to match the normal VS standard
        planes = [[0], [1], [2], [1, 2], [0, 1, 2]][planes]

    if isGRAY:
        ChromaMotion = False
        planes = 0

    longlat = max(width, height)
    shortlat = min(width, height)
    # Scale grainLevel from -2-3 -> 0-5
    grainLevel += 2

    if grainLevelSetup:
        outputStage = 0
        tr = 3

    if (longlat <= 1050 and shortlat <= 576):
        autoTune = 0
    elif (longlat <= 1280 and shortlat <= 720):
        autoTune = 1
    elif (longlat <= 2048 and shortlat <= 1152):
        autoTune = 2
    else:
        autoTune = 3

    postTD = postTR * 2 + 1

    if isinstance(postFFT, int):
        postBlkSize = fallback(postBlkSize, [0, 48, 32, 12, 0][postFFT])
        if postFFT <= 0:
            postTR = 0
        if postFFT == 3:
            postTR = min(postTR, 7)
        if postFFT in [1, 2]:
            postTR = min(postTR, 2)

        postDenoiser = [
            partial(removegrain, mode=1),
            partial(fft3d, sigma=postSigma, planes=planes, bt=postTD,
                    ncpu=fftThreads, bw=postBlkSize, bh=postBlkSize, func=temporal_degrain),
            partial(fft3d, sigma=postSigma, planes=planes, bt=postTD,
                    ncpu=fftThreads, bw=postBlkSize, bh=postBlkSize, func=temporal_degrain),
            partial(DFTTest.denoise, sloc=postSigma * 4, tr=postTR, planes=planes,
                    block_size=postBlkSize, overlap=postBlkSize * 9 / 12),
            partial(nl_means, strength=postSigma / 2, tr=postTR, sr=2, device_id=gpuId, planes=planes),
        ][postFFT]
    else:
        postDenoiser = postFFT  # type: ignore

    SrchClipPP = fallback(SrchClipPP, [0, 0, 0, 3, 3, 3][grainLevel])  # type: ignore

    maxTR = max(tr, postTR)

    # radius/range parameter for the motion estimation algorithms
    # AVS version uses the following, but values seemed to be based on an
    # incorrect understanding of the MVTools motion seach algorithm, mistaking
    # it for the exact x264 behavior.
    # meAlgPar = [2,2,2,2,16,24,2,2][meAlg]
    # Using Dogway's SMDegrain options here instead of the TemporalDegrain2 AVSI versions, which seem wrong.
    meAlgPar = fallback(meAlgPar, 5 if refine and meTM else 2)
    meSubpel = fallback(meSubpel, [4, 2, 2, 1][autoTune])
    meBlksz = fallback(meBlksz, [8, 8, 16, 32][autoTune])
    hpad = meBlksz
    vpad = meBlksz

    Overlap = meBlksz // 2
    Lambda = (1000 if meTM else 100) * (meBlksz ** 2) // 64
    PNew = 50 if meTM else 25
    ppSAD1 = fallback(ppSAD1, [3, 5, 7, 9, 11, 13][grainLevel])
    ppSAD2 = fallback(ppSAD2, [2, 4, 5, 6, 7, 8][grainLevel])
    ppSCD1 = fallback(ppSCD1, [3, 3, 3, 4, 5, 6][grainLevel])
    CMplanes = [0, 1, 2] if ChromaMotion else [0]

    if DCT == 5:
        # rescale threshold to match the SAD values when using SATD
        ppSAD1 = int(ppSAD1 * 1.7)
        ppSAD2 = int(ppSAD2 * 1.7)
        # ppSCD1 - this must not be scaled since scd is always based on SAD independently of the actual dct setting

    # here the per-pixel measure is converted to the per-8x8-Block (8*8 = 64) measure MVTools is using
    thSAD1 = int(ppSAD1 * 64)
    thSAD2 = int(ppSAD2 * 64)
    thSCD1 = int(ppSCD1 * 64)

    limitAT = [-1, -1, 0, 0, 0, 1][grainLevel] + autoTune + 1
    limitSigma = fallback(limitSigma, [6, 8, 12, 16, 32, 48][limitAT])
    limitBlksz = fallback(limitBlksz, [12, 16, 24, 32, 64, 96][limitAT])

    sharpenRadius = 3 if extraSharp is True else None

    # TODO: Provide DFTTest version for improved quality + performance.
    def limiterFFT3D(clip: vs.VideoNode) -> vs.VideoNode:
        assert limitSigma and limitBlksz and grainLevel is not None

        s2 = limitSigma * 0.625
        s3 = limitSigma * 0.375
        s4 = limitSigma * 0.250
        ovNum = [4, 4, 4, 3, 2, 2][grainLevel]
        ov = 2 * round(limitBlksz / ovNum * 0.5)

        return fft3d(
            clip, planes=CMplanes, sigma=limitSigma, sigma2=s2, sigma3=s3, sigma4=s4,
            bt=3, bw=limitBlksz, bh=limitBlksz, ow=ov, oh=ov, ncpu=fftThreads, func=temporal_degrain
        )

    limiter = fallback(limiter, limiterFFT3D)  # type: ignore

    assert limiter

    # Blur image and soften edges to assist in motion matching of edge blocks.
    # Blocks are matched by SAD (sum of absolute differences between blocks), but even
    # a slight change in an edge from frame to frame will give a high SAD due to the higher contrast of edges
    if isinstance(SrchClipPP, Prefilter):
        srchClip = SrchClipPP(clip)
    elif isinstance(SrchClipPP, vs.VideoNode):
        srchClip = SrchClipPP  # type: ignore
    else:
        srchClip = [
            Prefilter.NONE, Prefilter.SCALEDBLUR, Prefilter.GAUSSBLUR1, Prefilter.GAUSSBLUR2
        ][SrchClipPP](clip)  # type: ignore

    # TODO Add thSADC support, like AVS version
    preset = MVToolsPresets.CUSTOM(
        tr=tr, refine=refine, prefilter=srchClip,
        pel=meSubpel, hpad=hpad, vpad=vpad, sharp=SubPelInterp,
        block_size=meBlksz, overlap=Overlap,
        search=SearchMode(meAlg)(recalc_mode=SearchMode(meAlg), param=meAlgPar, pel=meSubpel),
        motion=MotionMode.MANUAL(truemotion=meTM, coherence=Lambda, pnew=PNew, pglobal=GlobalMotion),
        sad_mode=SADMode(DCT).same_recalc,
        super_args=dict(chroma=ChromaMotion),
        analyze_args=dict(chroma=ChromaMotion),
        recalculate_args=dict(thsad=thSAD1 // 2, lambda_=Lambda // 4),
        planes=planes
    )

    # Run motion analysis on the widest tr that we'll use for any operation,
    # whether degrain or post, and then reuse them for all following operations.
    maxMV = MVTools(clip, **preset(tr=maxTR))
    maxMV.analyze()

    # First MV-denoising stage. Usually here's some temporal-medianfiltering going on.
    # For simplicity, we just use MDegrain.
    NR1 = MVTools(clip, vectors=maxMV, **preset).degrain(thSAD=thSAD1, thSCD=(thSCD1, thSCD2))

    if tr > 0:
        spat = limiter(clip) if callable(limiter) else limiter

        spatD = core.std.MakeDiff(clip, spat)

        # Limit NR1 to not do more than what "spat" would do.
        NR1D = core.std.MakeDiff(clip, NR1)
        expr = 'x abs y abs < x y ?' if isFLOAT else f'x {neutral} - abs y {neutral} - abs < x y ?'
        DD = core.std.Expr([spatD, NR1D], [expr])
        NR1x = core.std.MakeDiff(clip, DD, [0])

        # Second MV-denoising stage. We use MDegrain.
        NR2 = MVTools(NR1x, vectors=maxMV, **preset).degrain(thSAD=thSAD2, thSCD=(thSCD1, thSCD2))
    else:
        NR2 = clip

    # Post (final stage) denoising.
    if postTR > 0:
        mvNoiseWindow = MVTools(NR2, vectors=maxMV, **preset(tr=postTR))
        dnWindow = mvNoiseWindow.compensate(postDenoiser, thSAD=thSAD2, thSCD=(thSCD1, thSCD2))
    else:
        dnWindow = postDenoiser(NR2)

    sharpened = contrasharpening(dnWindow, clip, sharpenRadius)

    if postMix > 0:
        sharpened = core.std.Expr([clip, sharpened], f"x {postMix} * y {100-postMix} * + 100 /")

    return [NR1x, NR2, sharpened][outputStage]
