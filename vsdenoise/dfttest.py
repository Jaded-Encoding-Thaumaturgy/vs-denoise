from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Iterator, Literal, Mapping, NamedTuple, Sequence, TypeAlias, TypeVar, overload

from vstools import (
    CustomEnum, CustomIntEnum, CustomOverflowError, CustomRuntimeError, CustomValueError, DependencyNotFoundError,
    FuncExceptT, KwargsNotNone, KwargsT, PlanesT, SupportsFloatOrIndex, check_variable, core, flatten, inject_self, vs
)

__all__ = [
    'SInterMode',

    'SLocation', 'SLocationT', 'SLocT',

    'NLocation',

    'FilterType', 'FilterTypeT',

    'SynthesisType', 'SynthesisTypeT',

    'DFTTest'
]

Frequency: TypeAlias = float
Sigma: TypeAlias = float


class SInterMode(CustomEnum):
    LINEAR = 'linear'
    SPLINE = 1
    SPLINE_LINEAR = 'slinear'
    QUADRATIC = 'quadratic'
    CUBIC = 'cubic'
    NEAREST = 'nearest'
    NEAREST_UP = 'nearest-up'
    ZERO = 'zero'

    @overload
    def __call__(self, location: SLocationT, /, res: int = 20, digits: int = 3) -> SLocation:
        ...

    @overload
    def __call__(
        self, h_loc: SLocationT | None = None, v_loc: SLocationT | None = None,
        t_loc: SLocationT | None = None, /, res: int = 20, digits: int = 3
    ) -> SLocation.MultiDim:
        ...

    @overload
    def __call__(self, *location: SLocationT, res: int = 20, digits: int = 3) -> SLocation | SLocation.MultiDim:
        ...

    def __call__(  # type: ignore
        self, *locations: SLocationT, res: int = 20, digits: int = 3
    ) -> SLocation | SLocation.MultiDim:
        if len(locations) == 1:
            return SLocation.from_param(locations[0]).interpolate(self, res, digits)

        return SLocation.MultiDim(*(self(x, res, digits) for x in locations))


class SLocation:
    frequencies: tuple[float, ...]
    sigmas: tuple[float, ...]

    NoProcess: SLocation

    @classmethod
    def boundsCheck(
        cls, values: list[float], bounds: tuple[float | None, float | None], strict: bool = False
    ) -> list[float]:
        if not values:
            raise CustomValueError('"values" can\'t be empty!', cls)

        values = values.copy()

        bounds_str = iter('inf' if x is None else x for x in bounds)
        of_error = CustomOverflowError("Invalid value at index {i}, not in ({bounds})", cls, bounds=bounds_str)

        low_bound, up_bound = bounds

        for i, value in enumerate(values):
            if low_bound is not None and value < low_bound:
                if strict:
                    raise of_error(i=i)

                values[i] = low_bound

            if up_bound is not None and value > up_bound:
                if strict:
                    raise of_error(i=i)

                values[i] = up_bound

        return values

    @overload
    @classmethod
    def from_param(cls: type[SLocBoundT], location: SLocationT | Literal[False]) -> SLocBoundT:
        ...

    @overload
    @classmethod
    def from_param(cls: type[SLocBoundT], location: SLocationT | Literal[False] | None) -> SLocBoundT | None:
        ...

    @classmethod
    def from_param(cls: type[SLocBoundT], location: SLocationT | Literal[False] | None) -> SLocBoundT | None:
        if isinstance(location, SupportsFloatOrIndex):  # type: ignore
            location = float(location)  # type: ignore
            location = {0: location, 1: location}

        if location is None:
            return None

        if location is False:
            location = SLocation.NoProcess

        if isinstance(location, SLocation):
            return cls(list(location))

        return cls(location)  # type: ignore

    def __init__(
        self, locations: Sequence[Frequency | Sigma] | Sequence[tuple[Frequency, Sigma]] | Mapping[Frequency, Sigma],
        interpolate: SInterMode | None = None, strict: bool = True
    ) -> None:
        if isinstance(locations, Mapping):
            frequencies, sigmas = list(locations.keys()), list(locations.values())
        else:
            locations = list[float](flatten(locations))  # type: ignore [arg-type]

            if len(locations) % 2:
                raise CustomValueError(
                    "slocations must resolve to an even number of total items, pairing frequency and sigma respectively",
                    self.__class__
                )

            frequencies, sigmas = list(locations[0::2]), list(locations[1::2])

        frequencies = self.boundsCheck(frequencies, (0, 1), strict)
        sigmas = self.boundsCheck(sigmas, (0, None), strict)

        self.frequencies, self.sigmas = (t for t in zip(*sorted(zip(frequencies, sigmas))))

        if interpolate:
            interpolated = self.interpolate(interpolate)

            self.frequencies, self.sigmas = interpolated.frequencies, interpolated.sigmas

    def __iter__(self) -> Iterator[float]:
        return iter([v for pair in zip(self.frequencies, self.sigmas) for v in pair])

    def __reversed__(self) -> SLocation:
        return SLocation(
            dict(zip((1 - f for f in reversed(self.frequencies)), list(reversed(self.sigmas))))
        )

    def interpolate(self, method: SInterMode = SInterMode.LINEAR, res: int = 20, digits: int = 3) -> SLocation:
        try:
            from scipy.interpolate import interp1d  # type: ignore
        except ModuleNotFoundError as e:
            raise DependencyNotFoundError(
                self.__class__, e, "scipy is required for interpolation. Use `pip install scipy`"
            )

        frequencies = list({round(x / (res - 1), digits) for x in range(res)})
        sigmas = interp1d(
            list(self.frequencies), list(self.sigmas), method.value, fill_value='extrapolate'
        )(frequencies)

        return SLocation(
            dict(zip(frequencies, sigmas)) | dict(zip(self.frequencies, self.sigmas)), strict=False
        )

    @dataclass
    class MultiDim:
        horizontal: SLocationT | Literal[False] | None = None
        vertical: SLocationT | Literal[False] | None = None
        temporal: SLocationT | Literal[False] | None = None

        def __post_init__(self) -> None:
            if not (self.horizontal or self.vertical or self.temporal):
                raise CustomValueError('You must specify at least one dimension!')

            self._horizontal = SLocation.from_param(self.horizontal)
            self._vertical = SLocation.from_param(self.vertical)
            self._temporal = SLocation.from_param(self.temporal)


SLocation.NoProcess = SLocation({0: 0, 1: 0})

SLocBoundT = TypeVar('SLocBoundT', bound=SLocation)

SLocationT = float | SLocation | Sequence[Frequency
                                          | Sigma] | Sequence[tuple[Frequency, Sigma]] | Mapping[Frequency, Sigma]
SLocT = SLocationT | SLocation.MultiDim


class NLocation(NamedTuple):
    frame_number: int
    plane: int
    ypos: int
    xpos: int


class FilterTypeWithInfo(KwargsT):
    ...


class FilterType(CustomIntEnum):
    WIENER = 0
    THR = 1
    MULT = 2
    MULT_PSD = 3
    MULT_RANGE = 4

    if TYPE_CHECKING:
        from .dfttest import FilterType

        @overload
        def __call__(  # type: ignore[misc]
            self: Literal[FilterType.WIENER], *, sigma: float = 8.0, beta: float = 1.0,
            nlocation: Sequence[NLocation] | None = None
        ) -> FilterTypeWithInfo:
            ...

        @overload
        def __call__(  # type: ignore[misc]
            self: Literal[FilterType.THR] | Literal[FilterType.MULT], *, sigma: float = 8.0,
            nlocation: Sequence[NLocation] | None = None
        ) -> FilterTypeWithInfo:
            ...

        @overload
        def __call__(  # type: ignore[misc]
            self: Literal[FilterType.MULT_PSD], *,
            sigma: float = 8.0, pmin: float = 0.0,
            sigma2: float = 16.0, pmax: float = 500.0
        ) -> FilterTypeWithInfo:
            ...

        @overload
        def __call__(  # type: ignore[misc]
            self: Literal[FilterType.MULT_RANGE], *,
            sigma: float = 8.0, pmin: float = 0.0, pmax: float = 500.0
        ) -> FilterTypeWithInfo:
            ...

        def __call__(self, **kwargs: Any) -> FilterTypeWithInfo:
            ...
    else:
        def __call__(self, **kwargs: Any) -> FilterTypeWithInfo:
            if self is FilterType.WIENER:
                def_kwargs = KwargsT(sigma=8.0, beta=1.0, nlocation=None)
            elif self in {FilterType.THR, FilterType.MULT}:
                def_kwargs = KwargsT(sigma=8.0, nlocation=None)
            elif self is FilterType.MULT_PSD:
                def_kwargs = KwargsT(sigma=8.0, pmin=0.0, sigma2=16.0, pmax=500.0)
            elif self is FilterType.MULT_RANGE:
                def_kwargs = KwargsT(sigma=8.0, pmin=0.0, pmax=500.0)
            else:
                def_kwargs = KwargsT()

            kwargs = def_kwargs | kwargs

            if 'beta' in kwargs:
                kwargs['f0beta'] = kwargs.pop('beta')

            return FilterTypeWithInfo(ftype=self.value, **kwargs)


FilterTypeT = FilterType | FilterTypeWithInfo


class SynthesisTypeWithInfo(KwargsT):
    def to_dict(self, otype: str) -> KwargsT:
        value = self.copy()

        if 'beta' in value:
            value = value.copy()
            value[f'{otype}beta'] = value.pop('beta')

        value[f'{otype}win'] = value.pop('win')

        return value


class SynthesisType(CustomIntEnum):
    HANNING = 0
    HAMMING = 1
    NUTTALL = 10
    BLACKMAN = 2
    BLACKMAN_NUTTALL = 11
    BLACKMAN_HARRIS_4TERM = 3
    BLACKMAN_HARRIS_7TERM = 5
    KAISER_BESSEL = 4
    FLAT_TOP = 6
    RECTANGULAR = 7
    BARLETT = 8
    BARLETT_HANN = 9

    if TYPE_CHECKING:
        from .dfttest import SynthesisType

        @overload
        def __call__(  # type: ignore[misc]
            self: Literal[SynthesisType.KAISER_BESSEL], *, beta: float = 2.5
        ) -> SynthesisTypeWithInfo:
            ...

        @overload
        def __call__(self, **kwargs: Any) -> SynthesisTypeWithInfo:
            ...

        def __call__(self, **kwargs: Any) -> SynthesisTypeWithInfo:
            ...
    else:
        def __call__(self, **kwargs: Any) -> SynthesisTypeWithInfo:
            if self is SynthesisType.KAISER_BESSEL and 'beta' not in kwargs:
                kwargs['beta'] = 2.5

            return SynthesisTypeWithInfo(win=self.value, **kwargs)


SynthesisTypeT = SynthesisType | SynthesisTypeWithInfo


class BackendInfo(KwargsT):
    backend: DFTTest.Backend

    def __init__(self, backend: DFTTest.Backend, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.backend = backend

    def __call__(
        self, clip: vs.VideoNode, sloc: SLocT | None = None,
        ftype: FilterTypeT | None = None,
        block_size: int | None = None, overlap: int | None = None,
        tr: int | None = None, tr_overlap: int | None = None,
        swin: SynthesisTypeT | None = None,
        twin: SynthesisTypeT | None = None,
        zmean: bool | None = None, alpha: float | None = None, ssystem: int | None = None,
        blockwise: bool = True, planes: PlanesT = None, *, func: FuncExceptT | None = None,
        default_args: KwargsT | None = None, **dkwargs: Any
    ) -> vs.VideoNode:
        func = func or self.__class__

        assert check_variable(clip, func)

        Backend = DFTTest.Backend

        kwargs: KwargsT = KwargsT(
            ftype=ftype, swin=swin, sbsize=block_size, sosize=overlap, tbsize=((tr or 0) * 2) + 1,
            tosize=tr_overlap, twin=twin, zmean=zmean, alpha=alpha, ssystem=ssystem,
            planes=planes, smode=int(blockwise)
        )

        if isinstance(sloc, SLocation.MultiDim):
            kwargs |= KwargsT(ssx=sloc._horizontal, ssy=sloc._vertical, sst=sloc._temporal)
        else:
            kwargs |= KwargsT(slocation=SLocation.from_param(sloc))

        clean_dft_args = KwargsNotNone(default_args or {}) | KwargsNotNone(kwargs)

        dft_args: KwargsT = KwargsT()

        for key, value in clean_dft_args.items():
            if isinstance(value, Enum) and callable(value):
                value = value()

            if isinstance(value, SynthesisTypeWithInfo):
                value = value.to_dict(key[0])

            if isinstance(value, dict):
                dft_args |= value
                continue

            if key == 'nlocation' and value:
                value = list[float](flatten(value))

            if isinstance(value, SLocation):
                value = list(value)

            if isinstance(value, Enum):
                value = value.value

            dft_args[key] = value

        backend = self.backend

        if backend is Backend.AUTO:
            try:
                from dfttest2 import __version__  # type: ignore  # noqa: F401

                if hasattr(core, 'dfttest2_nvrtc'):
                    backend = Backend.NVRTC
                elif hasattr(core, 'dfttest2_cuda'):
                    backend = Backend.cuFFT
                elif hasattr(core, 'dfttest2_cpu'):
                    backend = Backend.CPU
                elif hasattr(core, 'dfttest2_gcc'):
                    backend = Backend.GCC
                else:
                    raise KeyError
            except (ModuleNotFoundError, KeyError):
                if hasattr(core, 'neo_dfttest'):
                    backend = Backend.NEO
                elif hasattr(core, 'dfttest'):
                    backend = Backend.OLD

        if backend in {Backend.cuFFT, Backend.NVRTC, Backend.CPU, Backend.GCC}:
            from dfttest2 import Backend as DFTBackend
            from dfttest2 import DFTTest as DFTTest2  # noqa

            dft2_backend = None

            if backend is Backend.NVRTC:
                num_streams = dkwargs.pop('num_streams', self.num_streams if hasattr(self, 'num_streams') else 1)
                dft2_backend = DFTBackend.NVRTC(**(dict(**self) | dict(num_streams=num_streams)))
            elif backend is Backend.cuFFT:
                dft2_backend = DFTBackend.cuFFT(**self)
            elif backend is Backend.CPU:
                dft2_backend = DFTBackend.CPU(**self)
            elif backend is Backend.GCC:
                dft2_backend = DFTBackend.GCC(**self)

            if (tosize := dft_args.pop('tosize', 0)):
                raise CustomValueError('{backend} doesn\'t support tosize != 0', func, tosize, backend=backend)

            if (smode := dft_args.pop('smode', 1)) != 1:
                raise CustomValueError(
                    '{backend} doesn\'t support smode != 1!', func, smode, backend=backend
                )

            if (sbsize := dft_args.pop('sbsize', 16)) != 16:
                raise CustomValueError(
                    '{backend} doesn\'t support sbsize != 16!', func, sbsize, backend=backend
                )

            if (nlocation := dft_args.pop('nlocation', None)) is not None:
                raise CustomValueError(
                    '{backend} doesn\'t support nlocation!', func, nlocation, backend=backend
                )

            if (alpha := dft_args.pop('alpha', None)) is not None:
                raise CustomValueError(
                    '{backend} doesn\'t support alpha!', func, nlocation, backend=backend
                )

            return DFTTest2(clip, **dft_args, backend=dft2_backend)  # type: ignore

        dft_args |= self

        if backend is Backend.OLD:
            return core.dfttest.DFTTest(clip, **dft_args)

        if backend is Backend.NEO:
            return core.neo_dfttest.DFTTest(clip, **dft_args)  # type: ignore

        raise CustomRuntimeError(
            'No implementation of DFTTest could be found, please install one. dfttest2 is recommended.', self.__class__
        )


class DFTTest:
    default_args: KwargsT
    default_slocation: SLocation | SLocation.MultiDim | None

    class Backend(CustomIntEnum):
        AUTO = auto()
        OLD = auto()
        NEO = auto()
        cuFFT = auto()
        NVRTC = auto()
        CPU = auto()
        GCC = auto()

        if TYPE_CHECKING:
            from .dfttest import DFTTest

            Backend: TypeAlias = DFTTest.Backend

            @overload
            def __call__(  # type: ignore [misc]
                self: Literal[Backend.OLD] | Literal[Backend.CPU], *, opt: int = ...
            ) -> BackendInfo:
                ...

            @overload
            def __call__(  # type: ignore [misc]
                self: Literal[Backend.NEO], *,
                threads: int = ..., fft_threads: int = ..., opt: int = ..., dither: int = ...
            ) -> BackendInfo:
                ...

            @overload
            def __call__(  # type: ignore [misc]
                self: Literal[Backend.cuFFT], *, device_id: int = 0, in_place: bool = True
            ) -> BackendInfo:
                ...

            @overload
            def __call__(  # type: ignore [misc]
                self: Literal[Backend.NVRTC], *, device_id: int = 0, num_streams: int = 1
            ) -> BackendInfo:
                ...

            @overload
            def __call__(self: Literal[Backend.GCC]) -> BackendInfo:  # type: ignore [misc]
                ...

            def __call__(self: Backend, **kwargs: Any) -> BackendInfo:
                ...
        else:
            def __call__(self, **kwargs: Any) -> BackendInfo:
                return BackendInfo(self, **kwargs)

    def __init__(
        self, clip: vs.VideoNode | None = None, plugin: Backend | BackendInfo = Backend.AUTO,
        sloc: SLocT | None = None, **kwargs: Any
    ) -> None:
        self.clip = clip
        self.plugin: BackendInfo = plugin() if isinstance(plugin, DFTTest.Backend) else plugin

        self.default_args = kwargs.copy()
        self.default_slocation = sloc if isinstance(sloc, SLocation.MultiDim) else SLocation.from_param(sloc)

    @overload  # type: ignore
    @classmethod
    def denoise(
        cls, ref: vs.VideoNode, sloc: SLocT | None = None,
        ftype: FilterTypeT = FilterType.WIENER,
        tr: int = 0, tr_overlap: int = 0,
        swin: SynthesisTypeT = SynthesisType.HANNING,
        twin: SynthesisTypeT = SynthesisType.RECTANGULAR,
        block_size: int = 16, overlap: int = 12,
        zmean: bool = True, alpha: float | None = None, ssystem: int = 0,
        blockwise: bool = True, planes: PlanesT = None, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        ...

    @overload
    @classmethod
    def denoise(
        cls, sloc: SLocT, ref: vs.VideoNode | None = None,
        ftype: FilterTypeT = FilterType.WIENER,
        tr: int = 0, tr_overlap: int = 0,
        swin: SynthesisTypeT = SynthesisType.HANNING,
        twin: SynthesisTypeT = SynthesisType.RECTANGULAR,
        block_size: int = 16, overlap: int = 12,
        zmean: bool = True, alpha: float | None = None, ssystem: int = 0,
        blockwise: bool = True, planes: PlanesT = None, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        ...

    if not TYPE_CHECKING:
        @inject_self
        def denoise(
            self, ref: SLocT | vs.VideoNode | None = None, sloc: SLocT | vs.VideoNode | None = None,
            ftype: FilterTypeT = FilterType.WIENER,
            tr: int = 0, tr_overlap: int = 0,
            swin: SynthesisTypeT = SynthesisType.HANNING,
            twin: SynthesisTypeT = SynthesisType.RECTANGULAR,
            block_size: int = 16, overlap: int = 12,
            zmean: bool = True, alpha: float | None = None, ssystem: int = 0,
            blockwise: bool = True, planes: PlanesT = None, func: FuncExceptT | None = None, **kwargs: Any
        ) -> vs.VideoNode:
            func = func or self.denoise

            clip = self.clip
            nsloc = self.default_slocation

            if ref is not None:
                if isinstance(ref, vs.VideoNode):
                    clip = ref
                else:
                    nsloc = ref

            if sloc is not None:
                if isinstance(sloc, vs.VideoNode):
                    clip = sloc
                else:
                    nsloc = sloc

            if clip is None:
                raise CustomValueError('You must pass a clip!', func)

            return self.plugin(
                clip, nsloc, func=func, **(self.default_args | dict(
                    ftype=ftype, block_size=block_size, overlap=overlap, tr=tr, tr_overlap=tr_overlap, swin=swin,
                    twin=twin, zmean=zmean, alpha=alpha, ssystem=ssystem, blockwise=blockwise, planes=planes
                ) | kwargs)
            )

    @inject_self
    def insert_freq(self, low: vs.VideoNode, high: vs.VideoNode, sloc: SLocT, **kwargs: Any) -> vs.VideoNode:
        return low.std.MergeDiff(high.std.MakeDiff(self.denoise(high, sloc, func=self.insert_freq, **kwargs)))

    @inject_self
    def merge_freq(self, low: vs.VideoNode, high: vs.VideoNode, sloc: SLocT, **kwargs: Any) -> vs.VideoNode:
        return self.insert_freq(
            self.denoise(sloc, low, func=self.merge_freq, **kwargs), high, sloc, func=self.merge_freq, **kwargs
        )
