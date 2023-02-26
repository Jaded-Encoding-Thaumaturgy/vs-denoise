"""
This module implements prefilters for denoisers
"""

from __future__ import annotations

from enum import EnumMeta
from math import ceil, sin
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from vsaa import Nnedi3, Znedi3
from vsexprtools import ExprOp, aka_expr_available, norm_expr
from vsmasktools import retinex
from vskernels import Bicubic, BicubicZopti, Bilinear, Scaler, ScalerT
from vsrgtools import bilateral, blur, gauss_blur, min_blur, replace_low_frequencies
from vstools import (
    ColorRange, ConvMode, CustomEnum, CustomIntEnum, CustomRuntimeError, DitherType, PlanesT, SingleOrArr,
    SingleOrArrOpt, check_variable, clamp, core, depth, disallow_variable_format, disallow_variable_resolution,
    fallback, get_depth, get_neutral_value, get_peak_value, get_y, join, normalize_planes, scale_8bit, scale_value,
    split, vs, MissingT, MISSING, normalize_seq
)

from .bm3d import BM3D as BM3DM
from .bm3d import BM3DCPU, AbstractBM3D, BM3DCuda, BM3DCudaRTC, Profile
from .dfttest import DFTTest
from .knlm import DEVICETYPE, DeviceType, nl_means

__all__ = [
    'Prefilter', 'prefilter_to_full_range',
    'MultiPrefilter',
    'PelType'
]


class PrefilterMeta(EnumMeta):
    def __instancecheck__(cls: EnumMeta, instance: Any) -> bool:
        if isinstance(instance, PrefilterPartial):
            return True
        return super().__instancecheck__(instance)  # type: ignore


class PrefilterBase(CustomIntEnum, metaclass=PrefilterMeta):
    @overload
    def __call__(  # type: ignore
        self: Prefilter, *, planes: PlanesT = None, **kwargs: Any
    ) -> PrefilterPartial:
        ...

    @overload
    def __call__(  # type: ignore
        self: Prefilter, clip: vs.VideoNode, /, planes: PlanesT = None, **kwargs: Any
    ) -> vs.VideoNode:
        ...

    def __call__(  # type: ignore
        self: Prefilter, clip: vs.VideoNode | MissingT = MISSING, /, planes: PlanesT = None, **kwargs: Any
    ) -> vs.VideoNode | PrefilterPartial:
        if clip is MISSING:
            return PrefilterPartial(self, planes, **kwargs)

        assert check_variable(clip, self)

        pref_type = Prefilter.MINBLUR3 if self == Prefilter.AUTO else self

        bits = get_depth(clip)
        peak = get_peak_value(clip)
        planes = normalize_planes(clip, planes)

        if pref_type == Prefilter.NONE:
            return clip

        if pref_type.value in {0, 1, 2}:
            return min_blur(clip, pref_type.value, planes)

        if pref_type == Prefilter.MINBLURFLUX:
            temp_thr, spat_thr = kwargs.get('temp_thr', 2), kwargs.get('spat_thr', 2)
            return min_blur(clip, 2, planes).flux.SmoothST(temp_thr, spat_thr, planes)

        if pref_type == Prefilter.DFTTEST_SMOOTH:
            sigma = kwargs.get('sigma', 128.0)
            sigma2 = kwargs.get('sigma2', sigma / 16)
            sbsize = kwargs.get('sbsize', 8 if clip.width > 1280 else 6)
            sosize = kwargs.get('sosize', 6 if clip.width > 1280 else 4)

            kwargs |= dict(
                sbsize=sbsize, sosize=sosize, slocation=[
                    0.0, sigma2, 0.05, sigma, 0.5, sigma, 0.75, sigma2, 1.0, 0.0
                ]
            )

            pref_type = Prefilter.DFTTEST

        if pref_type == Prefilter.DFTTEST:
            dftt = DFTTest(sloc={0.0: 4, 0.2: 9, 1.0: 15}, tr=0).denoise(clip, **kwargs)

            i, j = (scale_value(x, 8, bits, range_out=ColorRange.FULL) for x in (16, 75))

            pref_mask = norm_expr(
                get_y(clip),
                f'x {i} < {peak} x {j} > 0 {peak} x {i} - {peak} {j} {i} - / * - ? ?'
            )

            return dftt.std.MaskedMerge(clip, pref_mask, planes)

        if pref_type == Prefilter.NLMEANS:
            kwargs |= dict(strength=7.0, tr=1, sr=2, simr=2) | kwargs | dict(planes=planes)
            knl = nl_means(clip, **kwargs)

            return replace_low_frequencies(knl, clip, 600 * (clip.width / 1920), False, planes)

        if pref_type == Prefilter.BM3D:
            bm3d_arch: type[AbstractBM3D] = kwargs.pop('arch', None)
            gpu: bool | None = kwargs.pop('gpu', None)

            if gpu is None:
                gpu = hasattr(core, 'bm3dcuda')

            if bm3d_arch is None:
                if gpu:  # type: ignore
                    bm3d_arch = BM3DCudaRTC if hasattr(core, 'bm3dcuda_rtc') else BM3DCuda
                else:
                    bm3d_arch = BM3DCPU if hasattr(core, 'bm3dcpu') else BM3DM

            if bm3d_arch is BM3DM:
                sigma, profile = 10, Profile.FAST
            elif bm3d_arch is BM3DCPU:
                sigma, profile = 10, Profile.LOW_COMPLEXITY
            elif bm3d_arch in (BM3DCuda, BM3DCudaRTC):
                sigma, profile = 8, Profile.NORMAL
            else:
                raise ValueError

            sigmas = kwargs.pop('sigma', [sigma if 0 in planes else 0, sigma if (1 in planes or 2 in planes) else 0])

            bm3d_args = dict[str, Any](sigma=sigmas, radius=1, profile=profile) | kwargs

            return bm3d_arch(clip, **bm3d_args).final()

        if pref_type == Prefilter.SCALEDBLUR:
            scale = kwargs.pop('scale', 2)
            downscaler = Scaler.ensure_obj(kwargs.pop('downscaler', Bilinear))
            upscaler = downscaler.ensure_obj(kwargs.pop('upscaler', downscaler))

            downscale = downscaler.scale(clip, clip.width // scale, clip.height // scale)

            boxblur = blur(downscale, kwargs.pop('radius', 1), kwargs.pop('mode', ConvMode.SQUARE), planes)

            return upscaler.scale(boxblur, clip.width, clip.height)

        if pref_type == Prefilter.GAUSS:
            if 'sharp' not in kwargs and 'sigma' not in kwargs:
                kwargs |= dict(sigma=1.5)

            return gauss_blur(clip, **(kwargs | dict[str, Any](planes=planes)))

        if pref_type == Prefilter.GAUSSBLUR:
            if 'sharp' not in kwargs and 'sigma' not in kwargs:
                kwargs |= dict(sigma=1.0)

            dgd = gauss_blur(clip, **(kwargs | dict[str, Any](planes=planes)))

            return replace_low_frequencies(dgd, clip, clip.width / 2)

        if pref_type in {Prefilter.GAUSSBLUR1, Prefilter.GAUSSBLUR2}:
            boxblur = blur(clip, kwargs.pop('radius', 1), kwargs.get('mode', ConvMode.SQUARE), planes=planes)

            if 'sharp' not in kwargs and 'sigma' not in kwargs:
                kwargs |= dict(sigma=1.75)

            strg = clamp(kwargs.pop('strength', 50 if pref_type == Prefilter.GAUSSBLUR2 else 90), 0, 98) + 1

            gaussblur = gauss_blur(boxblur, **(kwargs | dict[str, Any](planes=planes)))

            if pref_type == Prefilter.GAUSSBLUR2:
                i2, i7 = (scale_8bit(clip, x) for x in (2, 7))

                merge_expr = f'x {i7} + y < x {i2} + x {i7} - y > x {i2} - x {strg} * y {100 - strg} * + 100 / ? ?'
            else:
                merge_expr = f'x {strg / 100} * y {(100 - strg) / 100} * +'

            return norm_expr([gaussblur, clip], merge_expr, planes)

        if pref_type is Prefilter.BILATERAL:
            sigmaS = cast(float | list[float] | tuple[float | list[float], ...], kwargs.pop('sigmaS', 3.0))
            sigmaR = cast(float | list[float] | tuple[float | list[float], ...], kwargs.pop('sigmaR', 0.02))

            if isinstance(sigmaS, tuple):
                baseS, *otherS = sigmaS
            else:
                baseS, otherS = sigmaS, []

            if isinstance(sigmaR, tuple):
                baseR, *otherR = sigmaR
            else:
                baseR, otherR = sigmaR, []

            base, ref = clip, None
            max_len = max(len(otherS), len(otherR))

            if max_len:
                otherS = list[float | list[float]](reversed(normalize_seq(otherS or baseS, max_len)))
                otherR = list[float | list[float]](reversed(normalize_seq(otherR or baseR, max_len)))

                for siS, siR in zip(otherS, otherR):
                    base, ref = ref or clip, bilateral(base, siS, siR, ref, **kwargs)

            return bilateral(clip, baseS, baseR, ref, **kwargs)

        if pref_type is Prefilter.BMLATERAL:
            sigma = kwargs.pop('sigma', 1.5)
            tr = kwargs.pop('tr', 2)
            radius = kwargs.pop('radius', 7)
            cuda = kwargs.pop('gpu', hasattr(core, 'bm3dcuda'))

            den = Prefilter.BM3D(
                clip, planes, sigma=[10.0, 8.0] if cuda else [8.0, 6.4], radius=tr, gpu=cuda, **kwargs
            )

            return Prefilter.BILATERAL(den, planes, sigmaS=[sigma, sigma / 3], sigmaR=radius / 255)


class Prefilter(PrefilterBase):
    """
    Enum representing available filters.\n
    These are mainly thought of as prefilters for :py:attr:`MVTools`,
    but can be used standalone as-is.
    """

    AUTO = -2
    """Automatically decide what prefilter to use."""

    NONE = -1
    """Don't do any prefiltering. Returns the clip as-is."""

    MINBLUR1 = 0
    """A gaussian/temporal median merge with a radius of 1."""

    MINBLUR2 = 1
    """A gaussian/temporal median merge with a radius of 2."""

    MINBLUR3 = 2
    """A gaussian/temporal median merge with a radius of 3."""

    MINBLURFLUX = 3
    """:py:attr:`MINBLUR2` with temporal/spatial average."""

    DFTTEST = 4
    """Denoising in frequency domain with dfttest and an adaptive mask for retaining lineart."""

    DFTTEST_SMOOTH = 12
    """Denoising like in DFTTEST but with high defaults for lower frequencies."""

    NLMEANS = 5
    """Denoising with NLMeans, then postprocessed to remove low frequencies."""

    KNLMEANSCL = NLMEANS
    """Deprecated, use NLMEANS instead."""

    BM3D = 6
    """Normal spatio-temporal denoising using BM3D."""

    SCALEDBLUR = 7
    """Perform blurring at a scaled-down resolution, then scale it back up."""

    GAUSS = 13
    """Simply Gaussian blur."""

    GAUSSBLUR = 8
    """Gaussian blurred, then postprocessed to remove low frequencies."""

    GAUSSBLUR1 = 9
    """Clamped gaussian/box blurring."""

    GAUSSBLUR2 = 10
    """Clamped gaussian/box blurring with edge preservation."""

    BILATERAL = 11
    """Classic bilateral filtering or edge-preserving bilateral multi pass filtering."""

    BMLATERAL = 14
    """BM3D + BILATERAL blurring."""

    if TYPE_CHECKING:
        from .prefilters import Prefilter

        @overload  # type: ignore
        def __call__(
            self: Literal[Prefilter.MINBLURFLUX], clip: vs.VideoNode, /, planes: PlanesT = None,
            *, temp_thr: int = 2, spat_thr: int = 2
        ) -> vs.VideoNode:
            """
            :py:attr:`MINBLUR2` with temporal/spatial average.

            :param clip:        Clip to be preprocessed.
            :param planes:      Planes to be preprocessed.
            :param temp_thr:    Temporal threshold for the temporal median function.
            :param spat_thr:    Spatial threshold for the temporal median function.

            :return:            Preprocessed clip.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.DFTTEST], clip: vs.VideoNode, /, planes: PlanesT = None,
            *,
            tbsize: int = 1, sbsize: int = 12, sosize: int = 6, swin: int = 2,
            slocation: SingleOrArr[float] = [0.0, 4.0, 0.2, 9.0, 1.0, 15.0],
            ftype: int | None = None, sigma: float | None = None, sigma2: float | None = None,
            pmin: float | None = None, pmax: float | None = None, smode: int | None = None,
            tmode: int | None = None, tosize: int | None = None, twin: int | None = None,
            sbeta: float | None = None, tbeta: float | None = None, zmean: int | None = None,
            f0beta: float | None = None, nlocation: SingleOrArrOpt[int] = None, alpha: float | None = None,
            ssx: SingleOrArrOpt[float] = None, ssy: SingleOrArrOpt[float] = None, sst: SingleOrArrOpt[float] = None,
            ssystem: int | None = None, opt: int | None = None
        ) -> vs.VideoNode:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.DFTTEST_SMOOTH], clip: vs.VideoNode, /, planes: PlanesT = None, *,
            sigma: float = 128.0, sigma2: float | None = None, sbsize: int | None = None, sosize: int | None = None,
            **kwargs: Any
        ) -> vs.VideoNode:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.NLMEANS], clip: vs.VideoNode, /, planes: PlanesT = None,
            *, strength: SingleOrArr[float] = 7.0, tr: SingleOrArr[int] = 1, sr: SingleOrArr[int] = 2,
            simr: SingleOrArr[int] = 2, device_type: DEVICETYPE | DeviceType = DeviceType.AUTO, **kwargs: Any
        ) -> vs.VideoNode:
            """
            Denoising with NLMeans, then postprocessed to remove low frequencies.

            :param clip:            Source clip.
            :param strength:        Controls the strength of the filtering.\n
                                    Larger values will remove more noise.
            :param tr:              Temporal Radius. Temporal size = `(2 * tr + 1)`.\n
                                    Sets the number of past and future frames to uses for denoising the current frame.\n
                                    tr=0 uses 1 frame, while tr=1 uses 3 frames and so on.\n
                                    Usually, larger values result in better denoising.
            :param sr:              Search Radius. Spatial size = `(2 * sr + 1)^2`.\n
                                    Sets the radius of the search window.\n
                                    sr=1 uses 9 pixel, while sr=2 uses 25 pixels and so on.\n
                                    Usually, larger values result in better denoising.
            :param simr:            Similarity Radius. Similarity neighbourhood size = `(2 * simr + 1) ** 2`.\n
                                    Sets the radius of the similarity neighbourhood window.\n
                                    The impact on performance is low, therefore it depends on the nature of the noise.
            :param planes:          Set the clip planes to be processed.
            :param device_type:     Set the device to use for processing. The fastest device will be used by default.
            :param kwargs:          Additional arguments passed to the plugin.

            :return:                Denoised clip.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.BM3D], clip: vs.VideoNode, /, planes: PlanesT = None,
            *, arch: type[AbstractBM3D] = ..., gpu: bool = False,
            sigma: SingleOrArr[float] = ..., radius: SingleOrArr[int] = 1,
            profile: Profile = ..., ref: vs.VideoNode | None = None, refine: int = 1
        ) -> vs.VideoNode:
            """
            Normal spatio-temporal denoising using BM3D.

            :param clip:        Clip to be preprocessed.
            :param sigma:       Strength of denoising, valid range is [0, +inf].
            :param radius:      Temporal radius, valid range is [1, 16].
            :param profile:     See :py:attr:`vsdenoise.bm3d.Profile`.
            :param ref:         Reference clip used in block-matching, replacing the basic estimation.
                                If not specified, the input clip is used instead.
            :param refine:      Times to refine the estimation.
                                * 0 means basic estimate only.
                                * 1 means basic estimate with one final estimate.
                                * n means basic estimate refined with final estimate for n times.

            :return:            Preprocessed clip.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.SCALEDBLUR], clip: vs.VideoNode, /, planes: PlanesT = None,
            scale: int = 2, radius: int = 1, mode: ConvMode = ConvMode.SQUARE,
            downscaler: ScalerT = Bilinear, upscaler: ScalerT | None = None
        ) -> vs.VideoNode:
            """
            Perform blurring at a scaled-down resolution, then scale it back up.

            :param clip:        Clip to be preprocessed.
            :param planes:      Planes to be preprocessed.
            :param scale:       Ratios for downscaling.
                                A ratio of 2 will divide the resolution by 2, 4 by 4, etc.
            :param radius:      :py:attr:`vsrgtools.blur` radius param.
            :param mode:        Convolution mode for blurring.
            :param downscaler:  Scaler to be used for downscaling.
            :param upscaler:    Scaler to be used for reupscaling.\n
                                If None, :py:attr:`downscaler` will be used.

            :return:            Preprocessed clip.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.GAUSSBLUR], clip: vs.VideoNode, /, planes: PlanesT = None,
            *, sigma: float | None = 1.0, sharp: float | None = None, mode: ConvMode = ConvMode.SQUARE
        ) -> vs.VideoNode:
            """
            Gaussian blurred, then postprocessed to remove low frequencies.

            :param clip:        Clip to be preprocessed.
            :param planes:      Planes to be preprocessed.
            :param sigma:       Sigma param for :py:attr:`vsrgtools.gauss_blur`.
            :param sharp:       Sharp param for :py:attr:`vsrgtools.gauss_blur`.\n
                                Either :py:attr:`sigma` or this should be specified.
            :param mode:        Convolution mode for blurring.

            :return:            Preprocessed clip.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.GAUSSBLUR1], clip: vs.VideoNode, /, planes: PlanesT = None,
            *, radius: int = 1, strength: int = 90, sigma: float | None = 1.75,
            sharp: float | None = None, mode: ConvMode = ConvMode.SQUARE
        ) -> vs.VideoNode:
            """
            Clamped gaussian/box blurring with edge preservation.

            :param clip:        Clip to be preprocessed.
            :param planes:      Planes to be preprocessed.
            :param radius:      Radius param for the blurring.
            :param strength:    Clamping strength between the two blurred clips.\n
                                Must be between 1 and 99 (inclusive).
            :param sigma:       Sigma param for :py:attr:`vsrgtools.gauss_blur`.
            :param sharp:       Sharp param for :py:attr:`vsrgtools.gauss_blur`.\n
                                Either :py:attr:`sigma` or this should be specified.
            :param mode:        Convolution mode for blurring.

            :return:            Preprocessed clip.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.GAUSSBLUR2], clip: vs.VideoNode, /, planes: PlanesT = None,
            *, radius: int = 1, strength: int = 50, sigma: float | None = 1.75,
            sharp: float | None = None, mode: ConvMode = ConvMode.SQUARE
        ) -> vs.VideoNode:
            """
            Clamped gaussian/box blurring.

            :param clip:        Clip to be preprocessed.
            :param planes:      Planes to be preprocessed.
            :param radius:      Radius param for the blurring.
            :param strength:    Edge detection strength.\n
                                Must be between 1 and 99 (inclusive).
            :param sigma:       Sigma param for :py:attr:`vsrgtools.gauss_blur`.
            :param sharp:       Sharp param for :py:attr:`vsrgtools.gauss_blur`.\n
                                Either :py:attr:`sigma` or this should be specified.
            :param mode:        Convolution mode for blurring.

            :return:            Preprocessed clip.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.BILATERAL], clip: vs.VideoNode, /, planes: PlanesT = None,
            *, sigmaS: float | list[float] | tuple[float | list[float], ...] = 3.0,
            sigmaR: float | list[float] | tuple[float | list[float], ...] = 0.02,
            gpu: bool | None = None, **kwargs: Any
        ) -> vs.VideoNode:
            """
            Classic bilateral filtering or edge-preserving bilateral multi pass filtering.
            If sigmaS or sigmaR are tuples, first values will be used as base,
            other values as a recursive reference.

            :param clip:        Clip to be preprocessed.
            :param planes:      Planes to be preprocessed.
            :param sigmaS:      Sigma of Gaussian function to calculate spatial weight.
            :param sigmaR:      Sigma of Gaussian function to calculate range weight.
            :param gpu:         Whether to use GPU processing if available or not.

            :return:            Preprocessed clip.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.BMLATERAL], clip: vs.VideoNode, /, planes: PlanesT = None,
            sigma: float = 1.5, radius: float = 7, tr: int = 2, gpu: bool | None = None, **kwargs: Any
        ) -> vs.VideoNode:
            """
            BM3D + BILATERAL smoothing.

            :param clip:        Clip to be preprocessed.
            :param planes:      Planes to be preprocessed.
            :param sigma:       ``Bilateral`` spatial weight sigma.
            :param radius:      ``Bilateral`` radius weight sigma.
            :param tr:          Temporal radius for BM3D.
            :param gpu:         Whether to process with GPU or not. None is auto.
            :param **kwargs:    Kwargs passed to BM3D.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(self, clip: vs.VideoNode, /, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
            """
            Run the selected filter.

            :param clip:        Clip to be preprocessed.
            :param planes:      Planes to be preprocessed.
            :param kwargs:      Arguments for the specified filter.

            :return:            Preprocessed clip.
            """

        @overload  # type: ignore
        def __call__(
            self: Literal[Prefilter.MINBLURFLUX], *, planes: PlanesT = None, temp_thr: int = 2, spat_thr: int = 2
        ) -> PrefilterPartial:
            """
            :py:attr:`MINBLUR2` with temporal/spatial average.

            :param planes:      Planes to be preprocessed.
            :param temp_thr:    Temporal threshold for the temporal median function.
            :param spat_thr:    Spatial threshold for the temporal median function.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.DFTTEST], *, planes: PlanesT = None,
            tbsize: int = 1, sbsize: int = 12, sosize: int = 6, swin: int = 2,
            slocation: SingleOrArr[float] = [0.0, 4.0, 0.2, 9.0, 1.0, 15.0],
            ftype: int | None = None, sigma: float | None = None, sigma2: float | None = None,
            pmin: float | None = None, pmax: float | None = None, smode: int | None = None,
            tmode: int | None = None, tosize: int | None = None, twin: int | None = None,
            sbeta: float | None = None, tbeta: float | None = None, zmean: int | None = None,
            f0beta: float | None = None, nlocation: SingleOrArrOpt[int] = None, alpha: float | None = None,
            ssx: SingleOrArrOpt[float] = None, ssy: SingleOrArrOpt[float] = None, sst: SingleOrArrOpt[float] = None,
            ssystem: int | None = None, opt: int | None = None
        ) -> PrefilterPartial:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.DFTTEST_SMOOTH], *, planes: PlanesT = None,
            sigma: float = 128.0, sigma2: float | None = None, sbsize: int | None = None, sosize: int | None = None,
            **kwargs: Any
        ) -> PrefilterPartial:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.NLMEANS], *, planes: PlanesT = None,
            strength: SingleOrArr[float] = 7.0, tr: SingleOrArr[int] = 1, sr: SingleOrArr[int] = 2,
            simr: SingleOrArr[int] = 2, device_type: DEVICETYPE | DeviceType = DeviceType.AUTO, **kwargs: Any
        ) -> PrefilterPartial:
            """
            Denoising with NLMeans, then postprocessed to remove low frequencies.

            :param planes:          Set the clip planes to be processed.
            :param strength:        Controls the strength of the filtering.\n
                                    Larger values will remove more noise.
            :param tr:              Temporal Radius. Temporal size = `(2 * tr + 1)`.\n
                                    Sets the number of past and future frames to uses for denoising the current frame.\n
                                    tr=0 uses 1 frame, while tr=1 uses 3 frames and so on.\n
                                    Usually, larger values result in better denoising.
            :param sr:              Search Radius. Spatial size = `(2 * sr + 1)^2`.\n
                                    Sets the radius of the search window.\n
                                    sr=1 uses 9 pixel, while sr=2 uses 25 pixels and so on.\n
                                    Usually, larger values result in better denoising.
            :param simr:            Similarity Radius. Similarity neighbourhood size = `(2 * simr + 1) ** 2`.\n
                                    Sets the radius of the similarity neighbourhood window.\n
                                    The impact on performance is low, therefore it depends on the nature of the noise.
            :param device_type:     Set the device to use for processing. The fastest device will be used by default.
            :param kwargs:          Additional arguments passed to the plugin.

            :return:                Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.BM3D], *, planes: PlanesT = None,
            arch: type[AbstractBM3D] = ..., gpu: bool = False,
            sigma: SingleOrArr[float] = ..., radius: SingleOrArr[int] = 1,
            profile: Profile = ..., ref: vs.VideoNode | None = None, refine: int = 1
        ) -> PrefilterPartial:
            """
            Normal spatio-temporal denoising using BM3D.

            :param sigma:       Strength of denoising, valid range is [0, +inf].
            :param radius:      Temporal radius, valid range is [1, 16].
            :param profile:     See :py:attr:`vsdenoise.bm3d.Profile`.
            :param ref:         Reference clip used in block-matching, replacing the basic estimation.
                                If not specified, the input clip is used instead.
            :param refine:      Times to refine the estimation.
                                * 0 means basic estimate only.
                                * 1 means basic estimate with one final estimate.
                                * n means basic estimate refined with final estimate for n times.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.SCALEDBLUR], *, planes: PlanesT = None,
            scale: int = 2, radius: int = 1, mode: ConvMode = ConvMode.SQUARE,
            downscaler: ScalerT = Bilinear, upscaler: ScalerT | None = None
        ) -> PrefilterPartial:
            """
            Perform blurring at a scaled-down resolution, then scale it back up.

            :param planes:      Planes to be preprocessed.
            :param scale:       Ratios for downscaling.
                                A ratio of 2 will divide the resolution by 2, 4 by 4, etc.
            :param radius:      :py:attr:`vsrgtools.blur` radius param.
            :param mode:        Convolution mode for blurring.
            :param downscaler:  Scaler to be used for downscaling.
            :param upscaler:    Scaler to be used for reupscaling.\n
                                If None, :py:attr:`downscaler` will be used.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.GAUSSBLUR], *, planes: PlanesT = None,
            sigma: float | None = 1.0, sharp: float | None = None, mode: ConvMode = ConvMode.SQUARE
        ) -> PrefilterPartial:
            """
            Gaussian blurred, then postprocessed to remove low frequencies.

            :param planes:      Planes to be preprocessed.
            :param sigma:       Sigma param for :py:attr:`vsrgtools.gauss_blur`.
            :param sharp:       Sharp param for :py:attr:`vsrgtools.gauss_blur`.\n
                                Either :py:attr:`sigma` or this should be specified.
            :param mode:        Convolution mode for blurring.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.GAUSSBLUR1], *, planes: PlanesT = None,
            radius: int = 1, strength: int = 90, sigma: float | None = 1.75,
            sharp: float | None = None, mode: ConvMode = ConvMode.SQUARE
        ) -> PrefilterPartial:
            """
            Clamped gaussian/box blurring with edge preservation.

            :param planes:      Planes to be preprocessed.
            :param radius:      Radius param for the blurring.
            :param strength:    Clamping strength between the two blurred clips.\n
                                Must be between 1 and 99 (inclusive).
            :param sigma:       Sigma param for :py:attr:`vsrgtools.gauss_blur`.
            :param sharp:       Sharp param for :py:attr:`vsrgtools.gauss_blur`.\n
                                Either :py:attr:`sigma` or this should be specified.
            :param mode:        Convolution mode for blurring.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.GAUSSBLUR2], *, planes: PlanesT = None,
            radius: int = 1, strength: int = 50, sigma: float | None = 1.75,
            sharp: float | None = None, mode: ConvMode = ConvMode.SQUARE
        ) -> PrefilterPartial:
            """
            Clamped gaussian/box blurring.

            :param planes:      Planes to be preprocessed.
            :param radius:      Radius param for the blurring.
            :param strength:    Edge detection strength.\n
                                Must be between 1 and 99 (inclusive).
            :param sigma:       Sigma param for :py:attr:`vsrgtools.gauss_blur`.
            :param sharp:       Sharp param for :py:attr:`vsrgtools.gauss_blur`.\n
                                Either :py:attr:`sigma` or this should be specified.
            :param mode:        Convolution mode for blurring.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.BILATERAL], *, planes: PlanesT = None,
            sigmaS: float | list[float] | tuple[float | list[float], ...] = 3.0,
            sigmaR: float | list[float] | tuple[float | list[float], ...] = 0.02,
            gpu: bool | None = None, **kwargs: Any
        ) -> vs.VideoNode:
            """
            Classic bilateral filtering or edge-preserving bilateral multi pass filtering.
            If sigmaS or sigmaR are tuples, first values will be used as base,
            other values as a recursive reference.

            :param planes:      Planes to be preprocessed.
            :param sigmaS:      Sigma of Gaussian function to calculate spatial weight.
            :param sigmaR:      Sigma of Gaussian function to calculate range weight.
            :param gpu:         Whether to use GPU processing if available or not.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.BMLATERAL], *, planes: PlanesT = None,
            sigma: float = 1.5, radius: float = 7, tr: int = 2, gpu: bool | None = None, **kwargs: Any
        ) -> vs.VideoNode:
            """
            BM3D + BILATERAL smoothing.

            :param planes:      Planes to be preprocessed.
            :param sigma:       ``Bilateral`` spatial weight sigma.
            :param radius:      ``Bilateral`` radius weight sigma.
            :param tr:          Temporal radius for BM3D.
            :param gpu:         Whether to process with GPU or not. None is auto.
            :param **kwargs:    Kwargs passed to BM3D.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(self, *, planes: PlanesT = None, **kwargs: Any) -> PrefilterPartial:
            """
            Run the selected filter.

            :param planes:      Planes to be preprocessed.
            :param kwargs:      Arguments for the specified filter.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self, *, planes: PlanesT = None, **kwargs: Any
        ) -> PrefilterPartial:
            ...

        @overload
        def __call__(  # type: ignore
            self, clip: vs.VideoNode, /, planes: PlanesT = None, **kwargs: Any
        ) -> vs.VideoNode:
            ...

        def __call__(  # type: ignore
            self, clip: vs.VideoNode | MissingT = MISSING, /, planes: PlanesT = None, **kwargs: Any
        ) -> vs.VideoNode | PrefilterPartial:
            ...


if TYPE_CHECKING:
    class PrefBase(Prefilter):  # type: ignore
        ...
else:
    class PrefBase:
        ...


class PrefilterPartial(PrefBase):  # type: ignore
    def __init__(self, prefilter: Prefilter, planes: PlanesT, **kwargs: Any) -> None:
        self.prefilter = prefilter
        self.planes = planes
        self.kwargs = kwargs

    def __call__(  # type: ignore
        self, clip: vs.VideoNode, /, planes: PlanesT = None, **kwargs: Any
    ) -> vs.VideoNode:
        return self.prefilter(clip, planes=fallback(planes, self.planes), **kwargs | self.kwargs)  # type: ignore


class MultiPrefilter(PrefBase):  # type: ignore
    def __init__(self, *prefilters: Prefilter) -> None:
        self.prefilters = prefilters

    def __call__(self, clip: vs.VideoNode, /, **kwargs: Any) -> vs.VideoNode:  # type: ignore
        for pref in self.prefilters:
            clip = pref(clip)

        return clip


def prefilter_to_full_range(pref: vs.VideoNode, range_conversion: float, planes: PlanesT = None) -> vs.VideoNode:
    """
    Convert a limited range clip to full range.\n
    Useful for expanding prefiltered clip's ranges to give motion estimation additional information to work with.

    :param pref:                Clip to be preprocessed.
    :param range_conversion:    Value which determines what range conversion method gets used.\n
                                 * >= 1.0 - Expansion with expr based on this coefficient.
                                 * >  0.0 - Expansion with retinex.
                                 * <= 0.0 - Simple conversion with resize plugin.
    :param planes:              Planes to be processed.

    :return:                    Full range clip.
    """

    planes = normalize_planes(pref, planes)

    work_clip, *chroma = split(pref) if planes == [0] else (pref, )

    assert (fmt := work_clip.format) and pref.format

    bits = get_depth(pref)
    is_integer = fmt.sample_type == vs.INTEGER

    # Luma expansion TV->PC (up to 16% more values for motion estimation)
    if range_conversion >= 1.0:
        neutral = get_neutral_value(work_clip, True)
        max_val = get_peak_value(work_clip)

        c = sin(0.0625)
        k = (range_conversion - 1) * c

        if is_integer:
            t = f'x {scale_8bit(pref, 16)} - {scale_8bit(pref, 219)} / {ExprOp.clamp(0, 1)}'
        else:
            t = ExprOp.clamp(0, 1, 'x').to_str()

        head = f'{k} {1 + c} {(1 + c) * c}'

        if aka_expr_available:
            head = f'{t} T! {head}'
            t = 'T@'

        luma_expr = f'{head} {t} {c} + / - * {t} 1 {k} - * +'

        if is_integer:
            luma_expr += f' {max_val} *'

        pref_full = norm_expr(work_clip, (luma_expr, f'x {neutral} - 128 * 112 / {neutral} +'), planes)
    elif range_conversion > 0.0:
        pref_full = retinex(work_clip, upper_thr=range_conversion, fast=False)
    else:
        pref_full = depth(
            work_clip, bits, range_out=ColorRange.FULL, range_in=ColorRange.LIMITED, dither_type=DitherType.NONE
        )

    if chroma:
        return join(pref_full, *chroma, family=pref.format.color_family)

    return pref_full


if TYPE_CHECKING:
    PelTypeBase = CustomEnum
else:
    class PelTypeBase(CustomEnum):
        ...

    class CUSTOM(Scaler):
        def __init__(self, scaler: str | type[Scaler] | Scaler, **kwargs: Any) -> None:
            self.scaler = Scaler.ensure_obj(scaler)
            self.kwargs = kwargs

        @disallow_variable_format
        @disallow_variable_resolution
        def __call__(
            self, clip: vs.VideoNode, pel: int, subpixel: int = 3,
            default: ScalerT | None = None, **kwargs: Any
        ) -> vs.VideoNode:
            return PelType.__call__(self.scaler, clip, pel, default, **(self.kwargs | kwargs))

        def scale(
            self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
        ) -> vs.VideoNode:
            self.scaler.scale(clip, width, height, shift, **kwargs)

    BICUBIC = CUSTOM(Bicubic)
    WIENER = CUSTOM(BicubicZopti)

    PelTypeBase.CUSTOM = CUSTOM
    PelTypeBase.BICUBIC = BICUBIC
    PelTypeBase.WIENER = WIENER


class PelType(int, PelTypeBase):
    AUTO = -1
    """Automatically decide what :py:class:`PelType` to use."""

    NONE = 0
    """Don't perform any scaling."""

    NNEDI3 = 4
    """Performs scaling with NNedi3, ZNedi3."""

    if TYPE_CHECKING:
        from .prefilters import PelType

        class CUSTOM(Scaler, PelType):  # type: ignore
            """Class for constructing your own :py:class:`PelType`."""

            def __init__(self, scaler: str | type[Scaler] | Scaler, **kwargs: Any) -> None:
                """
                Create custom :py:class`PelType` from a scaler.

                :param scaler:  Scaler to be used for scaling and create a pel clip.
                """

            def scale(  # type: ignore
                self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
            ) -> vs.VideoNode:
                ...

        BICUBIC: CUSTOM
        """Performs scaling with default bicubic values (:py:class:`vskernels.Catrom`)."""

        WIENER: CUSTOM
        """Performs scaling with the wiener filter (:py:class:`BicubicZopti`)."""

        def __new__(cls, value: int) -> PelType:
            ...

        def __init__(self, value: int) -> None:
            ...

    @disallow_variable_format
    @disallow_variable_resolution
    def __call__(
        pel_type: Scaler | PelType, clip: vs.VideoNode, pel: int,
        default: ScalerT | PelType | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        """
        Scale a clip. Useful for motion interpolation.

        :param clip:        Clip to be scaled.
        :param pel:         Rate of scaling.
        :param subpixel:    Precision used in mvtools calls.\n
                            Will be used with :py:attr:`PelType.AUTO`.
        :param default:     Specify a default :py:class:`PelType`/:py:class:`Scaler` top be used.\n
                            Will be used with :py:attr:`PelType.AUTO`.
        :param kwargs:      Keyword arguments passed to the scaler.

        :return:            Upscaled clip.
        """

        assert clip.format

        if pel_type is PelType.NONE or pel <= 1:
            return clip

        if pel_type is PelType.AUTO:
            if default:
                pel_type = default if isinstance(default, PelType) else Scaler.ensure_obj(default)
            else:
                val = 1 << 3 - ceil(clip.height / 1000)

                if val <= 1:
                    pel_type = PelType.BICUBIC
                elif val == 2:
                    pel_type = PelType.WIENER
                else:
                    pel_type = PelType.NNEDI3

        if pel_type == PelType.NNEDI3:
            nnedicl, nnedi, znedi = (hasattr(core, ns) for ns in ('nnedi3cl', 'nnedi3', 'znedi3'))
            do_nnedi = (nnedicl or nnedi) and not znedi

            if not any((nnedi, znedi, nnedicl)):
                raise CustomRuntimeError('Missing any nnedi3 implementation!', PelType.NNEDI3)

            kwargs |= {'nsize': 0, 'nns': clamp(((pel - 1) // 2) + 1, 0, 4), 'qual': clamp(pel - 1, 1, 3)} | kwargs

            pel_type = Nnedi3(**kwargs, opencl=nnedicl) if do_nnedi else Znedi3(**kwargs)

        assert isinstance(pel_type, Scaler)

        return pel_type.scale(clip, clip.width * pel, clip.height * pel, **kwargs)
