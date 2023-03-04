
from dataclasses import dataclass
from functools import partial
from typing import Callable

from vsrgtools import contrasharpening, removegrain
from vstools import core, fallback, get_color_family, get_depth, get_neutral_value, get_sample_type, vs

from .dfttest import DFTTest
from .knlm import nl_means
from .mvtools import MotionMode, MVTools, MVToolsPresets, SADMode, SearchMode
from .prefilters import Prefilter

__all__ = [
    'TemporalDegrain2'
]


@dataclass
class TemporalDegrain2:
    """
    TemporalDegrain2

    Changes from previous implementations:
        1. `degrainTr` was renamed to just `tr` to better align with vsdenoise standards.
            * `postTr` still remains to differentiate between the two.
        1. `degrainPlane` was renamed to `planes` to better align with vsdenoise standards.
            * It was also updated to be an int or a list.
        2. `rec` was renamed to `refine` to better align with vsdenoise standards.
            * It was also updated to be an int or boolean, enabling multiple levels of refinement if desired.
        3. `thSCD2` is now a percentage, with range 0-100, to match vsstandards. For instance thSCD2=128 (old behavior)
            is now thSCD2=50 (new behavior).
        4. `knlDevId` was renamed to `gpuId`, since the underlying NLMeans implementation is not strictly tied to
            KNLMeansCL any more. And other filters may use this value in the future.
        5. `extraSharp` can now be a bool or an int, with an int directly specifying the sharpening radius.
        6. `SrchClipPP` now accepts custom Prefilters, or prefiltered clips directly.
        7. `limiter` is a new parameter which accepts a custom denoising function (or Prefilter, or clip) to replace
            the internal use of fft3dfilter as the limiter reference.
            * Using a custom limiter can lead to signifcant speed or quality gains (or both), as fft3dfilter is rather
            both slow and lower quality these days.
        8. `fft3dfilter` is preferred over `neo_fft3dfilter`. The latest version of fft3dfilter from AmusementClub
            (https://github.com/AmusementClub/VapourSynth-FFT3DFilter) is significantly faster.

    Example usages:
        # Default denoising
        denoised = TemporalDegrain2().denoise(clip)

        # Reusable instance with custom overrides
        td = TemporalDegrain2()
        denoised1 = td.denoised(clip, degrainTr=1, grainLevel=1)
        denoised2 = td.denoised(clip, degrainTr=2, grainLevel=2)

        # Reusable instance with custom defaults
        td = TemporalDegrain2(grainLevel=3)
        denoised1 = td.denoise()
        denoised2 = td.denoise(grainLevel=1)

        # Fully custom denoisers
        denoised = TemporalDegrain2().denoise(
                prefilter=Prefilter.NLMEANS,
                limiter=DFTTest.denoise(clip, tr=0, sigma=10))

    """
    # Main tunables
    tr: int = 1
    grainLevel: int = 2

    planes: int | list[int] = 4

    postFFT: int | Callable[[vs.VideoNode], vs.VideoNode] = 0
    postSigma: int = 1

    # Various tuning / output / knobs params
    grainLevelSetup: bool = False
    outputStage: int = 2
    extraSharp: bool | int = False
    gpuId: int | None = 0
    fftThreads: int = 1

    #  Motion params
    meAlg: int = 4
    meAlgPar: int | None = None
    meSubpel: int | None = None
    meBlksz: int | None = None
    meTM: bool = False
    ppSAD1: int | None = None
    ppSAD2: int | None = None
    ppSCD1: int | None = None
    thSCD2: int = 50
    DCT: int = 0
    SubPelInterp: int = 2
    SrchClipPP: int | Prefilter | vs.VideoNode | None = None
    GlobalMotion: bool = True
    ChromaMotion: bool = True
    refine: bool | int = False

    # denoisers
    # Function used to limit the maximum denoising effect MVDegrain can have. Defaults to custom FFT3DFilter
    limiter: Prefilter | Callable[[vs.VideoNode], vs.VideoNode] | vs.VideoNode | None = None
    limitSigma: int | None = None
    limitBlksz: int | None = None

    # post denoising
    postTR: int = 1
    postMix: int = 0
    postBlkSize: int | None = None

    def denoise(self, clip: vs.VideoNode, /,
                tr: int | None = None, grainLevel: int | None = None,
                postFFT: int | Callable[[vs.VideoNode], vs.VideoNode] | None = None, postSigma: int | None = None,
                planes: int | list[int] | None = None,
                *, grainLevelSetup: bool | None = None, outputStage: int | None = None, meAlg: int | None = None,
                meAlgPar: int | None = None, meSubpel: int | None = None, meBlksz: int | None = None,
                meTM: bool | None = None, ppSAD1: int | None = None, ppSAD2: int | None = None,
                ppSCD1: int | None = None, thSCD2: int | None = None, DCT: int | None = None,
                SubPelInterp: int | None = None, SrchClipPP: int | Prefilter | vs.VideoNode | None = None,
                GlobalMotion: bool | None = None, ChromaMotion: bool | None = None, refine: bool | int | None = None,
                limiter: Prefilter | Callable[[vs.VideoNode], vs.VideoNode] | vs.VideoNode | None = None,
                limitSigma: int | None = None, limitBlksz: int | None = None, gpuId: int | None = None,
                postTR: int | None = None, postMix: int | None = None, postBlkSize: int | None = None,
                extraSharp: bool | int | None = None, fftThreads: int | None = None) -> vs.VideoNode:
        """
        Temporal Degrain Updated by ErazorTT, ported to vsdenoise by adworacz (Adub)

        Based on function by Sagekilla, idea + original script created by Didee
        Works as a simple temporal degraining function that'll remove
        MOST or even ALL grain and noise from video sources,
        including dancing grain, like the grain found on 300.
        Also note, the parameters don't need to be tweaked much.

        Required plugins:
        FFT3DFilter: https://github.com/myrsloik/VapourSynth-FFT3DFilter
        MVtools(sf): https://github.com/dubhater/vapoursynth-mvtools
                     https://github.com/IFeelBloated/vapoursynth-mvtools-sf

        Optional plugins:
        dfttest: https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest
        KNLMeansCL: https://github.com/Khanattila/KNLMeansCL

        recommendations to be followed for each new movie:
          1. start with default settings
          2. if less denoising is needed set grainLevel to 0, if you need more degraining start over
            reading at next paragraph
          3. if you need even less denoising:
             - EITHER: set outputStage to 1 or even 0 (faster)
             - OR: use the postMix setting and increase the value from 0 to as much as 100 (slower)

        recommendations for strong degraining:
          1. start with default settings
          2. search the noisiest* patch of the entire movie, enable grainLevelSetup (=true), zoom in as much as you
            can and prepare yourself for pixel peeping. (*it really MUST be the noisiest region where you want this
            filter to be effective)
          3. compare the output on this noisy* patch of your movie with different settings of grainLevel (0 to 3) and
            use the setting where the noise level is lowest (irrespectable of whether you find this to be too
            much filtering).

            If multiple grainLevel settings yield comparable results while grainLevelSetup=true and observing at
            maximal zoom be sure to use the lowest setting! If you're unsure leave it at the default (2), your
            result might no be optimal, but it will still be great.
          4. disable grainLevelSetup (=false), or just remove this argument from the function call. Now revert the
            zoom and look from a normal distance at different scenes of the movie and decide if you like what you see.
          5. if more denoising is needed try postFFT=1 with postSigma=1, then tune postSigma (obvious blocking and
            banding of regions in the sky are indications of a value which is at least a factor 2 too high)
          6. if you would need a postSigma of more than 2, try first to increase degrainTR to 2. The goal is to
            balance the numerical values of postSigma and degrainTR, some prefer more simga and others more TR, it's
            up to you. However, do not increase degrainTR above 1/8th of the fps (at 24fps up to 3).
          7. if you cranked up postSigma higher than 3 then try postFFT=3 instead. Also if there are any issues with
            banding then try postFFT=3.
          8. if the luma is clean but you still have visible chroma noise then you can adjust postSigmaC which will
            separately clean the chroma planes (at a considerable amount of processing speed).

        use only the following knobs (all other settings should already be where they need to be):
          - degrainTR (1), temporal radius of degrain, usefull range: min=default=1, max=fps/8. Higher values do clean
            the video more, but also increase probability of wrongly identified motion vectors which leads to
            washed out regions
          - grainLevel (2), if input noise level is relatively low set this to 0, if its unusually high you might need
            to increase it to 3. The right setting must be found using grainLevelSetup=true while all other settings
            are at default. Set this setting such that the noise level is lowest.
          - grainLevelSetup (false), only to be used while finding the right setting for grainLevel. This will skip
            all your other settings!
          - postFFT (0), if you want to remove absolutely all remaining noise suggestion is to use 1 or 2 (ff3dfilter)
            or for slightly higher quality at the expense of potentially worse speed 3 (dfttest).
            4 is KNLMeansCL. 0 is simply RemoveGrain(1)
          - postSigma (1), increase it to remove all the remaining noise you want removed, but do not increase too
            much since unnecessary high values have severe negative impact on either banding and/or sharpness
          - degrainPlane (4), if you just want to denoise only the chroma use 3 (this helps with compressability
            while the clip is almost identical to the original)
          - outputStage (2), if the degraining is too strong, you can output earlier stages
          - postMix (0), if the degraining is too strong, increase the value going from 0 to 100
          - fftThreads (1), usefull if you have processor cores to spare, increasing to 2 will probably help a little
            with speed.
          - rec (false), enables use of Recalculate for refining motion analysis. Enable for higher quality motion
            estimation but lower performance.
        """
        width = clip.width
        height = clip.height

        neutral = get_neutral_value(clip)
        isFLOAT = get_sample_type(clip) == vs.FLOAT
        isGRAY = get_color_family(clip) == vs.GRAY

        planes = fallback(planes, self.planes)
        if isinstance(planes, int):
            # Convert int-based plane selection to array-based plane selection to match the normal VS standard
            planes = [[0], [1], [2], [1, 2], [0, 1, 2]][planes]

        ChromaMotion = fallback(ChromaMotion, self.ChromaMotion)

        if isGRAY:
            ChromaMotion = False
            planes = 0

        longlat = max(width, height)
        shortlat = min(width, height)
        # Scale grainLevel from -2-3 -> 0-5
        grainLevel = fallback(grainLevel, self.grainLevel) + 2

        tr = fallback(tr, self.tr)
        outputStage = fallback(outputStage, self.outputStage)
        grainLevelSetup = fallback(grainLevelSetup, self.grainLevelSetup)
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

        postTR = fallback(postTR, self.postTR)
        postTD = postTR * 2 + 1
        postSigma = fallback(postSigma, self.postSigma)
        postMix = fallback(postMix, self.postMix)
        gpuId = fallback(gpuId, self.gpuId)
        fftThreads = fallback(fftThreads, self.fftThreads)
        postFFT = fallback(postFFT, self.postFFT)
        if isinstance(postFFT, int):
            postBlkSize = fallback(postBlkSize, self.postBlkSize, [0, 48, 32, 12, 0][postFFT])
            if postFFT <= 0:
                postTR = 0
            if postFFT == 3:
                postTR = min(postTR, 7)
            if postFFT in [1, 2]:
                postTR = min(postTR, 2)

            postDenoiser = [
                partial(removegrain, mode=1),
                partial(_fft3d, sigma=postSigma, planes=planes, bt=postTD,
                        ncpu=fftThreads, bw=postBlkSize, bh=postBlkSize),
                partial(_fft3d, sigma=postSigma, planes=planes, bt=postTD,
                        ncpu=fftThreads, bw=postBlkSize, bh=postBlkSize),
                partial(DFTTest.denoise, sloc=postSigma * 4, tr=postTR, planes=planes,
                        block_size=postBlkSize, overlap=postBlkSize * 9 / 12),
                partial(nl_means, strength=postSigma / 2, tr=postTR, sr=2, device_id=gpuId, planes=planes),
            ][postFFT]
        else:
            postDenoiser = postFFT

        SrchClipPP = fallback(SrchClipPP, self.SrchClipPP, [0, 0, 0, 3, 3, 3][grainLevel])

        maxTR = max(tr, postTR)

        refine = fallback(refine, self.refine)
        refine = 1 if isinstance(refine, bool) and self.refine else 0

        meTM = fallback(meTM, self.meTM)

        # radius/range parameter for the motion estimation algorithms
        # AVS version uses the following, but values seemed to be based on an
        # incorrect understanding of the MVTools motion seach algorithm, mistaking
        # it for the exact x264 behavior.
        # meAlgPar = [2,2,2,2,16,24,2,2][meAlg]
        # Using Dogway's SMDegrain options here instead of the TemporalDegrain2 AVSI versions, which seem wrong.
        meAlgPar = fallback(meAlgPar, self.meAlgPar, 5 if refine and meTM else 2)
        meAlg = fallback(meAlg, self.meAlg)
        meSubpel = fallback(meSubpel, self.meSubpel, [4, 2, 2, 1][autoTune])
        meBlksz = fallback(meBlksz, self.meBlksz, [8, 8, 16, 32][autoTune])
        hpad = meBlksz
        vpad = meBlksz
        meSharp = fallback(SubPelInterp, self.SubPelInterp)
        Overlap = meBlksz // 2
        Lambda = (1000 if meTM else 100) * (meBlksz ** 2) // 64
        PNew = 50 if meTM else 25
        GlobalMotion = fallback(GlobalMotion, self.GlobalMotion)
        DCT = fallback(DCT, self.DCT)
        ppSAD1 = fallback(ppSAD1, self.ppSAD1, [3, 5, 7, 9, 11, 13][grainLevel])
        ppSAD2 = fallback(ppSAD2, self.ppSAD2, [2, 4, 5, 6, 7, 8][grainLevel])
        ppSCD1 = fallback(ppSCD1, self.ppSCD1, [3, 3, 3, 4, 5, 6][grainLevel])
        CMplanes = [0, 1, 2] if ChromaMotion else [0]

        if DCT == 5:
            # rescale threshold to match the SAD values when using SATD
            ppSAD1 *= 1.7
            ppSAD2 *= 1.7
            # ppSCD1 - this must not be scaled since scd is always based on SAD independently of the actual dct setting

        # here the per-pixel measure is converted to the per-8x8-Block (8*8 = 64) measure MVTools is using
        thSAD1 = int(ppSAD1 * 64)
        thSAD2 = int(ppSAD2 * 64)
        thSCD1 = int(ppSCD1 * 64)
        thSCD2 = fallback(thSCD2, self.thSCD2)

        limitAT = [-1, -1, 0, 0, 0, 1][grainLevel] + autoTune + 1
        limitSigma = fallback(limitSigma, self.limitSigma, [6, 8, 12, 16, 32, 48][limitAT])
        limitBlksz = fallback(limitBlksz, self.limitBlksz, [12, 16, 24, 32, 64, 96][limitAT])

        sharpenRadius = fallback(extraSharp, self.extraSharp)
        sharpenRadius = 3 if isinstance(sharpenRadius, bool) and sharpenRadius else None

        # TODO: Provide DFTTest version for improved quality + performance.
        def limiterFFT3D(clip: vs.VideoNode) -> vs.VideoNode:
            s2 = limitSigma * 0.625
            s3 = limitSigma * 0.375
            s4 = limitSigma * 0.250
            ovNum = [4, 4, 4, 3, 2, 2][grainLevel]
            ov = 2 * round(limitBlksz / ovNum * 0.5)

            return _fft3d(clip, planes=CMplanes, sigma=limitSigma, sigma2=s2, sigma3=s3, sigma4=s4,
                          bt=3, bw=limitBlksz, bh=limitBlksz, ow=ov, oh=ov, ncpu=fftThreads)

        limiter = fallback(limiter, self.limiter, limiterFFT3D)

        # Blur image and soften edges to assist in motion matching of edge blocks.
        # Blocks are matched by SAD (sum of absolute differences between blocks), but even
        # a slight change in an edge from frame to frame will give a high SAD due to the higher contrast of edges
        if isinstance(SrchClipPP, Prefilter):
            srchClip = SrchClipPP(clip)
        elif isinstance(SrchClipPP, vs.VideoNode):
            srchClip = SrchClipPP
        else:
            srchClip = [Prefilter.NONE, Prefilter.SCALEDBLUR,
                        Prefilter.GAUSSBLUR1, Prefilter.GAUSSBLUR2][SrchClipPP](clip)

        # TODO Add thSADC support, like AVS version
        preset = MVToolsPresets.CUSTOM(tr=tr, refine=refine, prefilter=srchClip,
                                       pel=meSubpel, hpad=hpad, vpad=vpad, sharp=meSharp,
                                       block_size=meBlksz, overlap=Overlap,
                                       search=SearchMode(meAlg)(recalc_mode=SearchMode(
                                           meAlg), param=meAlgPar, pel=meSubpel),
                                       motion=MotionMode.MANUAL(truemotion=meTM, coherence=Lambda,
                                                                pnew=PNew, pglobal=GlobalMotion),
                                       sad_mode=SADMode(DCT).same_recalc,
                                       super_args=dict(chroma=ChromaMotion),
                                       analyze_args=dict(chroma=ChromaMotion),
                                       recalculate_args=dict(thsad=thSAD1 // 2, lambda_=Lambda // 4),
                                       planes=planes)

        # Run motion analysis on the widest tr that we'll use for any operation,
        # whether degrain or post, and then reuse them for all following operations.
        maxMV = MVTools(clip, **preset(tr=maxTR))
        maxMV.analyze()

        # First MV-denoising stage. Usually here's some temporal-medianfiltering going on.
        # For simplicity, we just use MDegrain.
        NR1 = MVTools(clip, vectors=maxMV, **preset).degrain(thSAD=thSAD1, thSCD=(thSCD1, thSCD2))

        if tr > 0:
            if isinstance(limiter, vs.VideoNode):
                spat = limiter
            else:
                spat = limiter(clip)

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


def _fft3d(clip: vs.VideoNode, **kwargs):
    if hasattr(core, 'fft3dfilter'):
        # fft3dfilter from AmusementClub is significantly faster in my tests
        # https://github.com/AmusementClub/VapourSynth-FFT3DFilter

        # Only thing is that fft3dfilter produces a change in contrast/tint
        # when used as a postFFT denoiser...
        # neo_fft3d does not have the same issue.

        # I think the summary is - all fft3dfilters are trash at this rate...

        # fft3dfilter requires sigma values to be scaled to bit depth
        # https://github.com/AmusementClub/VapourSynth-FFT3DFilter/blob/mod/doc/fft3dfilter.md#scaling-parameters-according-to-bit-depth
        sigmaMultiplier = 1.0 / 256.0 if get_sample_type(clip) == vs.FLOAT else 1 << (get_depth(clip) - 8)
        for sigma in ['sigma', 'sigma2', 'sigma3', 'sigma4']:
            if sigma in kwargs:
                kwargs[sigma] *= sigmaMultiplier

        return core.fft3dfilter.FFT3DFilter(clip, **kwargs)
    elif hasattr(core, 'neo_fft3d'):
        # neo_fft3d is slower than fft3d filter for me...
        return core.neo_fft3d.FFT3D(clip, **kwargs)
    else:
        raise ImportError("TemporalDegrain2: No suitable version of fft3dfilter/neo_fft3d found, please install one.")
