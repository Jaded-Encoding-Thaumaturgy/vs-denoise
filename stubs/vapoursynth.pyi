# Stop pep8 from complaining (hopefully)
# NOQA

# Ignore Flake Warnings
# flake8: noqa

# Ignore coverage
# (No coverage)

# From https://gist.github.com/pylover/7870c235867cf22817ac5b096defb768
# noinspection PyPep8
# noinspection PyPep8Naming
# noinspection PyTypeChecker
# noinspection PyAbstractClass
# noinspection PyArgumentEqualDefault
# noinspection PyArgumentList
# noinspection PyAssignmentToLoopOrWithParameter
# noinspection PyAttributeOutsideInit
# noinspection PyAugmentAssignment
# noinspection PyBroadException
# noinspection PyByteLiteral
# noinspection PyCallByClass
# noinspection PyChainedComparsons
# noinspection PyClassHasNoInit
# noinspection PyClassicStyleClass
# noinspection PyComparisonWithNone
# noinspection PyCompatibility
# noinspection PyDecorator
# noinspection PyDefaultArgument
# noinspection PyDictCreation
# noinspection PyDictDuplicateKeys
# noinspection PyDocstringTypes
# noinspection PyExceptClausesOrder
# noinspection PyExceptionInheritance
# noinspection PyFromFutureImport
# noinspection PyGlobalUndefined
# noinspection PyIncorrectDocstring
# noinspection PyInitNewSignature
# noinspection PyInterpreter
# noinspection PyListCreation
# noinspection PyMandatoryEncoding
# noinspection PyMethodFirstArgAssignment
# noinspection PyMethodMayBeStatic
# noinspection PyMethodOverriding
# noinspection PyMethodParameters
# noinspection PyMissingConstructor
# noinspection PyMissingOrEmptyDocstring
# noinspection PyNestedDecorators
# noinspection PynonAsciiChar
# noinspection PyNoneFunctionAssignment
# noinspection PyOldStyleClasses
# noinspection PyPackageRequirements
# noinspection PyPropertyAccess
# noinspection PyPropertyDefinition
# noinspection PyProtectedMember
# noinspection PyRaisingNewStyleClass
# noinspection PyRedeclaration
# noinspection PyRedundantParentheses
# noinspection PySetFunctionToLiteral
# noinspection PySimplifyBooleanCheck
# noinspection PySingleQuotedDocstring
# noinspection PyStatementEffect
# noinspection PyStringException
# noinspection PyStringFormat
# noinspection PySuperArguments
# noinspection PyTrailingSemicolon
# noinspection PyTupleAssignmentBalance
# noinspection PyTupleItemAssignment
# noinspection PyUnboundLocalVariable
# noinspection PyUnnecessaryBackslash
# noinspection PyUnreachableCode
# noinspection PyUnresolvedReferences
# noinspection PyUnusedLocal
# noinspection ReturnValueFromInit


from abc import abstractmethod
from ctypes import Structure, c_void_p
from enum import IntEnum
from fractions import Fraction
from inspect import Parameter, Signature
from logging import NOTSET as LogLevelUnset
from logging import Handler, LogRecord
from types import MappingProxyType, TracebackType
from typing import (
    TYPE_CHECKING, Any, BinaryIO, Callable, ContextManager, Dict, Generic, Iterator, Literal, MutableMapping,
    NamedTuple, NoReturn, Optional, Protocol, Sequence, Type, TypedDict, TypeVar, Union, overload, runtime_checkable
)

__all__ = [
    # Versioning
    '__version__', '__api_version__',

    # Cython capsule
    '__pyx_capi__',

    # Enums and constants
    'MessageType',
        'MESSAGE_TYPE_DEBUG', 'MESSAGE_TYPE_INFORMATION', 'MESSAGE_TYPE_WARNING',
        'MESSAGE_TYPE_CRITICAL', 'MESSAGE_TYPE_FATAL',

    'FilterMode',
        'fmParallel', 'fmParallelRequests', 'fmUnordered', 'fmFrameState',

    'CoreCreationFlags',
        'ccfEnableGraphInspection', 'ccfDisableAutoLoading', 'ccfDisableLibraryUnloading',

    'MediaType',
        'VIDEO', 'AUDIO',

    'ColorFamily',
        'UNDEFINED', 'GRAY', 'RGB', 'YUV',

    'ColorRange',
        'RANGE_FULL', 'RANGE_LIMITED',

    'SampleType',
        'INTEGER', 'FLOAT',

    'PresetFormat',
        'GRAY',
        'GRAY8', 'GRAY9', 'GRAY10', 'GRAY12', 'GRAY14', 'GRAY16', 'GRAY32', 'GRAYH', 'GRAYS',
        'RGB',
        'RGB24', 'RGB27', 'RGB30', 'RGB36', 'RGB42', 'RGB48', 'RGBH', 'RGBS',
        'YUV',
        'YUV410P8',
        'YUV411P8',
        'YUV420P8', 'YUV420P9', 'YUV420P10', 'YUV420P12', 'YUV420P14', 'YUV420P16',
        'YUV422P8', 'YUV422P9', 'YUV422P10', 'YUV422P12', 'YUV422P14', 'YUV422P16',
        'YUV440P8',
        'YUV444P8', 'YUV444P9', 'YUV444P10', 'YUV444P12', 'YUV444P14', 'YUV444P16', 'YUV444PH', 'YUV444PS',
        'NONE',

    'AudioChannels',
        'FRONT_LEFT', 'FRONT_RIGHT', 'FRONT_CENTER',
        'BACK_LEFT', 'BACK_RIGHT', 'BACK_CENTER',
        'SIDE_LEFT', 'SIDE_RIGHT',
        'TOP_CENTER',

        'TOP_FRONT_LEFT', 'TOP_FRONT_RIGHT', 'TOP_FRONT_CENTER',
        'TOP_BACK_LEFT', 'TOP_BACK_RIGHT', 'TOP_BACK_CENTER',

        'WIDE_LEFT', 'WIDE_RIGHT',

        'SURROUND_DIRECT_LEFT', 'SURROUND_DIRECT_RIGHT',

        'FRONT_LEFT_OF_CENTER', 'FRONT_RIGHT_OF_CENTER',

        'STEREO_LEFT', 'STEREO_RIGHT',

        'LOW_FREQUENCY', 'LOW_FREQUENCY2',

    'ChromaLocation',
        'CHROMA_TOP_LEFT', 'CHROMA_TOP',
        'CHROMA_LEFT', 'CHROMA_CENTER',
        'CHROMA_BOTTOM_LEFT', 'CHROMA_BOTTOM',

    'FieldBased',
        'FIELD_PROGRESSIVE', 'FIELD_TOP', 'FIELD_BOTTOM',

    'MatrixCoefficients',
        'MATRIX_RGB', 'MATRIX_BT709', 'MATRIX_UNSPECIFIED', 'MATRIX_FCC',
        'MATRIX_BT470_BG', 'MATRIX_ST170_M', 'MATRIX_YCGCO', 'MATRIX_BT2020_NCL', 'MATRIX_BT2020_CL',
        'MATRIX_CHROMATICITY_DERIVED_NCL', 'MATRIX_CHROMATICITY_DERIVED_CL', 'MATRIX_ICTCP',

    'TransferCharacteristics',
        'TRANSFER_BT709', 'TRANSFER_UNSPECIFIED', 'TRANSFER_BT470_M', 'TRANSFER_BT470_BG', 'TRANSFER_BT601',
        'TRANSFER_ST240_M', 'TRANSFER_LINEAR', 'TRANSFER_LOG_100', 'TRANSFER_LOG_316', 'TRANSFER_IEC_61966_2_4',
        'TRANSFER_IEC_61966_2_1', 'TRANSFER_BT2020_10', 'TRANSFER_BT2020_12', 'TRANSFER_ST2084', 'TRANSFER_ARIB_B67',

    'ColorPrimaries', 'PRIMARIES_BT709', 'PRIMARIES_UNSPECIFIED',
        'PRIMARIES_BT470_M', 'PRIMARIES_BT470_BG', 'PRIMARIES_ST170_M', 'PRIMARIES_ST240_M', 'PRIMARIES_FILM',
        'PRIMARIES_BT2020', 'PRIMARIES_ST428', 'PRIMARIES_ST431_2', 'PRIMARIES_ST432_1', 'PRIMARIES_EBU3213_E',

    # Environment SubSystem
    'Environment', 'EnvironmentData',

    'EnvironmentPolicy',
        'StandaloneEnvironmentPolicy',
        'VSScriptEnvironmentPolicy',

    'EnvironmentPolicyAPI',
    'register_policy', 'has_policy',

    'get_current_environment',

    'VideoOutputTuple',
    'clear_output', 'clear_outputs', 'get_outputs', 'get_output',

    '_construct_type', '_construct_parameter', 'construct_signature',

    # Logging
    'PythonVSScriptLoggingBridge', 'LogHandle', 'Error',

    # Functions
    'FuncData', 'Func', 'VSMapValueCallbackData', 'FramePtr',
    'Plugin', 'Function',

    # Formats
    'VideoFormat',

    # Frames
    'RawFrame', 'VideoFrame', 'AudioFrame',
    'FrameProps',

    # Nodes
    'RawNode', 'VideoNode', 'AudioNode',

    'Core', '_CoreProxy', 'core',

    # Inspection API [UNSTABLE API]
    '_try_enable_introspection'
]


###
# Typing

T = TypeVar('T')
S = TypeVar('S')

SingleAndSequence = Union[T, Sequence[T]]


@runtime_checkable
class SupportsString(Protocol):
    @abstractmethod
    def __str__(self) -> str:
        ...


DataType = Union[str, bytes, bytearray, SupportsString]

_VapourSynthMapValue = Union[
    SingleAndSequence[int],
    SingleAndSequence[float],
    SingleAndSequence[DataType],
    SingleAndSequence['VideoNode'],
    SingleAndSequence['VideoFrame'],
    SingleAndSequence['AudioNode'],
    SingleAndSequence['AudioFrame'],
    SingleAndSequence['VSMapValueCallback[Any]']
]

BoundVSMapValue = TypeVar('BoundVSMapValue', bound=_VapourSynthMapValue)

VSMapValueCallback = Callable[..., BoundVSMapValue]


class _Future(Generic[T]):
    def set_result(self, value: T) -> None: ...

    def set_exception(self, exception: BaseException) -> None: ...

    def result(self) -> T: ...

    def exception(self) -> Union[NoReturn, None]: ...

###
# Typed dicts


class _VideoFormatInfo(TypedDict):
    id: int
    name: str
    color_family: 'ColorFamily'
    sample_type: 'SampleType'
    bits_per_sample: int
    bytes_per_sample: int
    subsampling_w: int
    subsampling_h: int
    num_planes: int


###
# VapourSynth Versioning


class VapourSynthVersion(NamedTuple):
    release_major: int
    release_minor: int


class VapourSynthAPIVersion(NamedTuple):
    api_major: int
    api_minor: int


__version__: VapourSynthVersion
__api_version__: VapourSynthAPIVersion

###
# VapourSynth Cython capsule


class PyCapsule(Structure):
    ...


__pyx_capi__: Dict[str, PyCapsule]

###
# VapourSynth Enums and Constants


class MessageType(IntEnum):
    MESSAGE_TYPE_DEBUG: 'MessageType'
    MESSAGE_TYPE_INFORMATION: 'MessageType'
    MESSAGE_TYPE_WARNING: 'MessageType'
    MESSAGE_TYPE_CRITICAL: 'MessageType'
    MESSAGE_TYPE_FATAL: 'MessageType'


MESSAGE_TYPE_DEBUG: Literal[MessageType.MESSAGE_TYPE_DEBUG]
MESSAGE_TYPE_INFORMATION: Literal[MessageType.MESSAGE_TYPE_INFORMATION]
MESSAGE_TYPE_WARNING: Literal[MessageType.MESSAGE_TYPE_WARNING]
MESSAGE_TYPE_CRITICAL: Literal[MessageType.MESSAGE_TYPE_CRITICAL]
MESSAGE_TYPE_FATAL: Literal[MessageType.MESSAGE_TYPE_FATAL]


class FilterMode(IntEnum):
    fmParallel: 'FilterMode'
    fmParallelRequests: 'FilterMode'
    fmUnordered: 'FilterMode'
    fmFrameState: 'FilterMode'


fmParallel: Literal[FilterMode.fmParallel]
fmParallelRequests: Literal[FilterMode.fmParallelRequests]
fmUnordered: Literal[FilterMode.fmUnordered]
fmFrameState: Literal[FilterMode.fmFrameState]


class CoreCreationFlags(IntEnum):
    ccfEnableGraphInspection: 'CoreCreationFlags'
    ccfDisableAutoLoading: 'CoreCreationFlags'
    ccfDisableLibraryUnloading: 'CoreCreationFlags'


ccfEnableGraphInspection: Literal[CoreCreationFlags.ccfEnableGraphInspection]
ccfDisableAutoLoading: Literal[CoreCreationFlags.ccfDisableAutoLoading]
ccfDisableLibraryUnloading: Literal[CoreCreationFlags.ccfDisableLibraryUnloading]


class MediaType(IntEnum):
    VIDEO: 'MediaType'
    AUDIO: 'MediaType'


VIDEO: Literal[MediaType.VIDEO]
AUDIO: Literal[MediaType.AUDIO]


class ColorFamily(IntEnum):
    UNDEFINED: 'ColorFamily'
    GRAY: 'ColorFamily'
    RGB: 'ColorFamily'
    YUV: 'ColorFamily'


UNDEFINED: Literal[ColorFamily.UNDEFINED]
GRAY: Literal[ColorFamily.GRAY]
RGB: Literal[ColorFamily.RGB]
YUV: Literal[ColorFamily.YUV]


class ColorRange(IntEnum):
    RANGE_FULL: 'ColorRange'
    RANGE_LIMITED: 'ColorRange'


RANGE_FULL: Literal[ColorRange.RANGE_FULL]
RANGE_LIMITED: Literal[ColorRange.RANGE_LIMITED]


class SampleType(IntEnum):
    INTEGER: 'SampleType'
    FLOAT: 'SampleType'


INTEGER: Literal[SampleType.INTEGER]
FLOAT: Literal[SampleType.FLOAT]


class PresetFormat(IntEnum):
    NONE: 'PresetFormat'

    GRAY8: 'PresetFormat'
    GRAY9: 'PresetFormat'
    GRAY10: 'PresetFormat'
    GRAY12: 'PresetFormat'
    GRAY14: 'PresetFormat'
    GRAY16: 'PresetFormat'
    GRAY32: 'PresetFormat'

    GRAYH: 'PresetFormat'
    GRAYS: 'PresetFormat'

    YUV420P8: 'PresetFormat'
    YUV422P8: 'PresetFormat'
    YUV444P8: 'PresetFormat'
    YUV410P8: 'PresetFormat'
    YUV411P8: 'PresetFormat'
    YUV440P8: 'PresetFormat'

    YUV420P9: 'PresetFormat'
    YUV422P9: 'PresetFormat'
    YUV444P9: 'PresetFormat'

    YUV420P10: 'PresetFormat'
    YUV422P10: 'PresetFormat'
    YUV444P10: 'PresetFormat'

    YUV420P12: 'PresetFormat'
    YUV422P12: 'PresetFormat'
    YUV444P12: 'PresetFormat'

    YUV420P14: 'PresetFormat'
    YUV422P14: 'PresetFormat'
    YUV444P14: 'PresetFormat'

    YUV420P16: 'PresetFormat'
    YUV422P16: 'PresetFormat'
    YUV444P16: 'PresetFormat'

    YUV444PH: 'PresetFormat'
    YUV444PS: 'PresetFormat'

    RGB24: 'PresetFormat'
    RGB27: 'PresetFormat'
    RGB30: 'PresetFormat'
    RGB36: 'PresetFormat'
    RGB42: 'PresetFormat'
    RGB48: 'PresetFormat'

    RGBH: 'PresetFormat'
    RGBS: 'PresetFormat'


NONE: Literal[PresetFormat.NONE]

GRAY8: Literal[PresetFormat.GRAY8]
GRAY9: Literal[PresetFormat.GRAY9]
GRAY10: Literal[PresetFormat.GRAY10]
GRAY12: Literal[PresetFormat.GRAY12]
GRAY14: Literal[PresetFormat.GRAY14]
GRAY16: Literal[PresetFormat.GRAY16]
GRAY32: Literal[PresetFormat.GRAY32]

GRAYH: Literal[PresetFormat.GRAYH]
GRAYS: Literal[PresetFormat.GRAYS]

YUV420P8: Literal[PresetFormat.YUV420P8]
YUV422P8: Literal[PresetFormat.YUV422P8]
YUV444P8: Literal[PresetFormat.YUV444P8]
YUV410P8: Literal[PresetFormat.YUV410P8]
YUV411P8: Literal[PresetFormat.YUV411P8]
YUV440P8: Literal[PresetFormat.YUV440P8]

YUV420P9: Literal[PresetFormat.YUV420P9]
YUV422P9: Literal[PresetFormat.YUV422P9]
YUV444P9: Literal[PresetFormat.YUV444P9]

YUV420P10: Literal[PresetFormat.YUV420P10]
YUV422P10: Literal[PresetFormat.YUV422P10]
YUV444P10: Literal[PresetFormat.YUV444P10]

YUV420P12: Literal[PresetFormat.YUV420P12]
YUV422P12: Literal[PresetFormat.YUV422P12]
YUV444P12: Literal[PresetFormat.YUV444P12]

YUV420P14: Literal[PresetFormat.YUV420P14]
YUV422P14: Literal[PresetFormat.YUV422P14]
YUV444P14: Literal[PresetFormat.YUV444P14]

YUV420P16: Literal[PresetFormat.YUV420P16]
YUV422P16: Literal[PresetFormat.YUV422P16]
YUV444P16: Literal[PresetFormat.YUV444P16]

YUV444PH: Literal[PresetFormat.YUV444PH]
YUV444PS: Literal[PresetFormat.YUV444PS]

RGB24: Literal[PresetFormat.RGB24]
RGB27: Literal[PresetFormat.RGB27]
RGB30: Literal[PresetFormat.RGB30]
RGB36: Literal[PresetFormat.RGB36]
RGB42: Literal[PresetFormat.RGB42]
RGB48: Literal[PresetFormat.RGB48]

RGBH: Literal[PresetFormat.RGBH]
RGBS: Literal[PresetFormat.RGBS]


class AudioChannels(IntEnum):
    FRONT_LEFT: 'AudioChannels'
    FRONT_RIGHT: 'AudioChannels'
    FRONT_CENTER: 'AudioChannels'
    LOW_FREQUENCY: 'AudioChannels'
    BACK_LEFT: 'AudioChannels'
    BACK_RIGHT: 'AudioChannels'
    FRONT_LEFT_OF_CENTER: 'AudioChannels'
    FRONT_RIGHT_OF_CENTER: 'AudioChannels'
    BACK_CENTER: 'AudioChannels'
    SIDE_LEFT: 'AudioChannels'
    SIDE_RIGHT: 'AudioChannels'
    TOP_CENTER: 'AudioChannels'
    TOP_FRONT_LEFT: 'AudioChannels'
    TOP_FRONT_CENTER: 'AudioChannels'
    TOP_FRONT_RIGHT: 'AudioChannels'
    TOP_BACK_LEFT: 'AudioChannels'
    TOP_BACK_CENTER: 'AudioChannels'
    TOP_BACK_RIGHT: 'AudioChannels'
    STEREO_LEFT: 'AudioChannels'
    STEREO_RIGHT: 'AudioChannels'
    WIDE_LEFT: 'AudioChannels'
    WIDE_RIGHT: 'AudioChannels'
    SURROUND_DIRECT_LEFT: 'AudioChannels'
    SURROUND_DIRECT_RIGHT: 'AudioChannels'
    LOW_FREQUENCY2: 'AudioChannels'


FRONT_LEFT: Literal[AudioChannels.FRONT_LEFT]
FRONT_RIGHT: Literal[AudioChannels.FRONT_RIGHT]
FRONT_CENTER: Literal[AudioChannels.FRONT_CENTER]
LOW_FREQUENCY: Literal[AudioChannels.LOW_FREQUENCY]
BACK_LEFT: Literal[AudioChannels.BACK_LEFT]
BACK_RIGHT: Literal[AudioChannels.BACK_RIGHT]
FRONT_LEFT_OF_CENTER: Literal[AudioChannels.FRONT_LEFT_OF_CENTER]
FRONT_RIGHT_OF_CENTER: Literal[AudioChannels.FRONT_RIGHT_OF_CENTER]
BACK_CENTER: Literal[AudioChannels.BACK_CENTER]
SIDE_LEFT: Literal[AudioChannels.SIDE_LEFT]
SIDE_RIGHT: Literal[AudioChannels.SIDE_RIGHT]
TOP_CENTER: Literal[AudioChannels.TOP_CENTER]
TOP_FRONT_LEFT: Literal[AudioChannels.TOP_FRONT_LEFT]
TOP_FRONT_CENTER: Literal[AudioChannels.TOP_FRONT_CENTER]
TOP_FRONT_RIGHT: Literal[AudioChannels.TOP_FRONT_RIGHT]
TOP_BACK_LEFT: Literal[AudioChannels.TOP_BACK_LEFT]
TOP_BACK_CENTER: Literal[AudioChannels.TOP_BACK_CENTER]
TOP_BACK_RIGHT: Literal[AudioChannels.TOP_BACK_RIGHT]
STEREO_LEFT: Literal[AudioChannels.STEREO_LEFT]
STEREO_RIGHT: Literal[AudioChannels.STEREO_RIGHT]
WIDE_LEFT: Literal[AudioChannels.WIDE_LEFT]
WIDE_RIGHT: Literal[AudioChannels.WIDE_RIGHT]
SURROUND_DIRECT_LEFT: Literal[AudioChannels.SURROUND_DIRECT_LEFT]
SURROUND_DIRECT_RIGHT: Literal[AudioChannels.SURROUND_DIRECT_RIGHT]
LOW_FREQUENCY2: Literal[AudioChannels.LOW_FREQUENCY2]


class ChromaLocation(IntEnum):
    CHROMA_LEFT: 'ChromaLocation'
    CHROMA_CENTER: 'ChromaLocation'
    CHROMA_TOP_LEFT: 'ChromaLocation'
    CHROMA_TOP: 'ChromaLocation'
    CHROMA_BOTTOM_LEFT: 'ChromaLocation'
    CHROMA_BOTTOM: 'ChromaLocation'


CHROMA_LEFT: Literal[ChromaLocation.CHROMA_LEFT]
CHROMA_CENTER: Literal[ChromaLocation.CHROMA_CENTER]
CHROMA_TOP_LEFT: Literal[ChromaLocation.CHROMA_TOP_LEFT]
CHROMA_TOP: Literal[ChromaLocation.CHROMA_TOP]
CHROMA_BOTTOM_LEFT: Literal[ChromaLocation.CHROMA_BOTTOM_LEFT]
CHROMA_BOTTOM: Literal[ChromaLocation.CHROMA_BOTTOM]


class FieldBased(IntEnum):
    FIELD_PROGRESSIVE: 'FieldBased'
    FIELD_TOP: 'FieldBased'
    FIELD_BOTTOM: 'FieldBased'


FIELD_PROGRESSIVE: Literal[FieldBased.FIELD_PROGRESSIVE]
FIELD_TOP: Literal[FieldBased.FIELD_TOP]
FIELD_BOTTOM: Literal[FieldBased.FIELD_BOTTOM]


class MatrixCoefficients(IntEnum):
    MATRIX_RGB: 'MatrixCoefficients'
    MATRIX_BT709: 'MatrixCoefficients'
    MATRIX_UNSPECIFIED: 'MatrixCoefficients'
    MATRIX_FCC: 'MatrixCoefficients'
    MATRIX_BT470_BG: 'MatrixCoefficients'
    MATRIX_ST170_M: 'MatrixCoefficients'
    MATRIX_YCGCO: 'MatrixCoefficients'
    MATRIX_BT2020_NCL: 'MatrixCoefficients'
    MATRIX_BT2020_CL: 'MatrixCoefficients'
    MATRIX_CHROMATICITY_DERIVED_NCL: 'MatrixCoefficients'
    MATRIX_CHROMATICITY_DERIVED_CL: 'MatrixCoefficients'
    MATRIX_ICTCP: 'MatrixCoefficients'


MATRIX_RGB: Literal[MatrixCoefficients.MATRIX_RGB]
MATRIX_BT709: Literal[MatrixCoefficients.MATRIX_BT709]
MATRIX_UNSPECIFIED: Literal[MatrixCoefficients.MATRIX_UNSPECIFIED]
MATRIX_FCC: Literal[MatrixCoefficients.MATRIX_FCC]
MATRIX_BT470_BG: Literal[MatrixCoefficients.MATRIX_BT470_BG]
MATRIX_ST170_M: Literal[MatrixCoefficients.MATRIX_ST170_M]
MATRIX_YCGCO: Literal[MatrixCoefficients.MATRIX_YCGCO]
MATRIX_BT2020_NCL: Literal[MatrixCoefficients.MATRIX_BT2020_NCL]
MATRIX_BT2020_CL: Literal[MatrixCoefficients.MATRIX_BT2020_CL]
MATRIX_CHROMATICITY_DERIVED_NCL: Literal[MatrixCoefficients.MATRIX_CHROMATICITY_DERIVED_NCL]
MATRIX_CHROMATICITY_DERIVED_CL: Literal[MatrixCoefficients.MATRIX_CHROMATICITY_DERIVED_CL]
MATRIX_ICTCP: Literal[MatrixCoefficients.MATRIX_ICTCP]


class TransferCharacteristics(IntEnum):
    TRANSFER_BT709: 'TransferCharacteristics'
    TRANSFER_UNSPECIFIED: 'TransferCharacteristics'
    TRANSFER_BT470_M: 'TransferCharacteristics'
    TRANSFER_BT470_BG: 'TransferCharacteristics'
    TRANSFER_BT601: 'TransferCharacteristics'
    TRANSFER_ST240_M: 'TransferCharacteristics'
    TRANSFER_LINEAR: 'TransferCharacteristics'
    TRANSFER_LOG_100: 'TransferCharacteristics'
    TRANSFER_LOG_316: 'TransferCharacteristics'
    TRANSFER_IEC_61966_2_4: 'TransferCharacteristics'
    TRANSFER_IEC_61966_2_1: 'TransferCharacteristics'
    TRANSFER_BT2020_10: 'TransferCharacteristics'
    TRANSFER_BT2020_12: 'TransferCharacteristics'
    TRANSFER_ST2084: 'TransferCharacteristics'
    TRANSFER_ARIB_B67: 'TransferCharacteristics'


TRANSFER_BT709: Literal[TransferCharacteristics.TRANSFER_BT709]
TRANSFER_UNSPECIFIED: Literal[TransferCharacteristics.TRANSFER_UNSPECIFIED]
TRANSFER_BT470_M: Literal[TransferCharacteristics.TRANSFER_BT470_M]
TRANSFER_BT470_BG: Literal[TransferCharacteristics.TRANSFER_BT470_BG]
TRANSFER_BT601: Literal[TransferCharacteristics.TRANSFER_BT601]
TRANSFER_ST240_M: Literal[TransferCharacteristics.TRANSFER_ST240_M]
TRANSFER_LINEAR: Literal[TransferCharacteristics.TRANSFER_LINEAR]
TRANSFER_LOG_100: Literal[TransferCharacteristics.TRANSFER_LOG_100]
TRANSFER_LOG_316: Literal[TransferCharacteristics.TRANSFER_LOG_316]
TRANSFER_IEC_61966_2_4: Literal[TransferCharacteristics.TRANSFER_IEC_61966_2_4]
TRANSFER_IEC_61966_2_1: Literal[TransferCharacteristics.TRANSFER_IEC_61966_2_1]
TRANSFER_BT2020_10: Literal[TransferCharacteristics.TRANSFER_BT2020_10]
TRANSFER_BT2020_12: Literal[TransferCharacteristics.TRANSFER_BT2020_12]
TRANSFER_ST2084: Literal[TransferCharacteristics.TRANSFER_ST2084]
TRANSFER_ARIB_B67: Literal[TransferCharacteristics.TRANSFER_ARIB_B67]


class ColorPrimaries(IntEnum):
    PRIMARIES_BT709: 'ColorPrimaries'
    PRIMARIES_UNSPECIFIED: 'ColorPrimaries'
    PRIMARIES_BT470_M: 'ColorPrimaries'
    PRIMARIES_BT470_BG: 'ColorPrimaries'
    PRIMARIES_ST170_M: 'ColorPrimaries'
    PRIMARIES_ST240_M: 'ColorPrimaries'
    PRIMARIES_FILM: 'ColorPrimaries'
    PRIMARIES_BT2020: 'ColorPrimaries'
    PRIMARIES_ST428: 'ColorPrimaries'
    PRIMARIES_ST431_2: 'ColorPrimaries'
    PRIMARIES_ST432_1: 'ColorPrimaries'
    PRIMARIES_EBU3213_E: 'ColorPrimaries'


PRIMARIES_BT709: Literal[ColorPrimaries.PRIMARIES_BT709]
PRIMARIES_UNSPECIFIED: Literal[ColorPrimaries.PRIMARIES_UNSPECIFIED]
PRIMARIES_BT470_M: Literal[ColorPrimaries.PRIMARIES_BT470_M]
PRIMARIES_BT470_BG: Literal[ColorPrimaries.PRIMARIES_BT470_BG]
PRIMARIES_ST170_M: Literal[ColorPrimaries.PRIMARIES_ST170_M]
PRIMARIES_ST240_M: Literal[ColorPrimaries.PRIMARIES_ST240_M]
PRIMARIES_FILM: Literal[ColorPrimaries.PRIMARIES_FILM]
PRIMARIES_BT2020: Literal[ColorPrimaries.PRIMARIES_BT2020]
PRIMARIES_ST428: Literal[ColorPrimaries.PRIMARIES_ST428]
PRIMARIES_ST431_2: Literal[ColorPrimaries.PRIMARIES_ST431_2]
PRIMARIES_ST432_1: Literal[ColorPrimaries.PRIMARIES_ST432_1]
PRIMARIES_EBU3213_E: Literal[ColorPrimaries.PRIMARIES_EBU3213_E]


###
# VapourSynth Environment SubSystem


class EnvironmentData:
    def __init__(self) -> NoReturn: ...  # type: ignore[misc]


class EnvironmentPolicy:
    def on_policy_registered(self, special_api: 'EnvironmentPolicyAPI') -> None: ...

    def on_policy_cleared(self) -> None: ...

    @abstractmethod
    def get_current_environment(self) -> Union[EnvironmentData, None]: ...

    @abstractmethod
    def set_environment(self, environment: Union[EnvironmentData, None]) -> Union[EnvironmentData, None]: ...

    def is_alive(self, environment: EnvironmentData) -> bool: ...


class StandaloneEnvironmentPolicy(EnvironmentPolicy):
    def __init__(self) -> NoReturn: ...  # type: ignore[misc]

    def _on_log_message(self, level: MessageType, msg: str) -> None: ...

    def on_policy_registered(self, api: 'EnvironmentPolicyAPI') -> None: ...

    def on_policy_cleared(self) -> None: ...

    def get_current_environment(self) -> EnvironmentData: ...

    def set_environment(self, environment: Union[EnvironmentData, None]) -> EnvironmentData: ...

    def is_alive(self, environment: EnvironmentData) -> bool: ...


class VSScriptEnvironmentPolicy(EnvironmentPolicy):
    def __init__(self) -> NoReturn: ...  # type: ignore[misc]

    def on_policy_registered(self, policy_api: 'EnvironmentPolicyAPI') -> None: ...

    def on_policy_cleared(self) -> None: ...

    def get_current_environment(self) -> Union[EnvironmentData, None]: ...

    def set_environment(self, environment: Union[EnvironmentData, None]) -> Union[EnvironmentData, None]: ...

    def is_alive(self, environment: EnvironmentData) -> bool: ...


class EnvironmentPolicyAPI:
    def __init__(self) -> NoReturn: ...  # type: ignore[misc]

    def wrap_environment(self, environment_data: EnvironmentData) -> 'Environment': ...

    def create_environment(self, flags: int = 0) -> EnvironmentData: ...

    def set_logger(self, env: EnvironmentData, logger: Callable[[int, str], None]) -> None: ...

    def destroy_environment(self, env: EnvironmentData) -> None: ...

    def unregister_policy(self) -> None: ...


def register_policy(policy: EnvironmentPolicy) -> None:
    ...


if not TYPE_CHECKING:
    def _try_enable_introspection(version: int = None): ...


def has_policy() -> bool:
    ...


class Environment:
    env: EnvironmentData

    def __init__(self) -> NoReturn: ...  # type: ignore[misc]

    @property
    def alive(self) -> bool: ...

    @property
    def single(self) -> bool: ...

    @classmethod
    def is_single(self) -> bool: ...

    @property
    def env_id(self) -> int: ...

    @property
    def active(self) -> bool: ...

    def copy(self) -> 'Environment': ...

    def use(self) -> ContextManager[None]: ...

    def __eq__(self, other: 'Environment') -> bool: ...  # type: ignore[override]

    def __repr__(self) -> str: ...


def get_current_environment() -> Environment:
    ...


class VideoOutputTuple(NamedTuple):
    clip: 'VideoNode'
    alpha: Union['VideoNode', None]
    alt_output: Literal[0, 1, 2]


def _construct_type(signature: str) -> type:
    ...


def _construct_parameter(signature: str) -> Parameter:
    ...


def construct_signature(signature: str, return_signature: str, injected: Union[str, None] = None) -> Signature:
    ...


class Error(Exception):
    ...


def clear_output(index: int = 0) -> None:
    ...


def clear_outputs() -> None:
    ...


def get_outputs() -> MappingProxyType[int, Union[VideoOutputTuple, 'AudioNode']]:
    ...


def get_output(index: int = 0) -> Union[VideoOutputTuple, 'AudioNode']:
    ...


class FuncData:
    def __init__(self) -> NoReturn: ...  # type: ignore[misc]

    def __call__(self, **kwargs: _VapourSynthMapValue) -> _VapourSynthMapValue: ...


class Func:
    def __init__(self) -> NoReturn: ...  # type: ignore[misc]

    def __call__(self, **kwargs: _VapourSynthMapValue) -> _VapourSynthMapValue: ...


class FramePtr:
    def __init__(self) -> NoReturn: ...  # type: ignore[misc]


class VideoFormat:
    id: int
    name: str
    color_family: ColorFamily
    sample_type: SampleType
    bits_per_sample: int
    bytes_per_sample: int
    subsampling_w: int
    subsampling_h: int
    num_planes: int

    def __init__(self) -> NoReturn: ...  # type: ignore[misc]

    def _as_dict(self) -> _VideoFormatInfo: ...

    def replace(
        self, *,
        color_family: Union[ColorFamily, None] = None,
        sample_type: Union[SampleType, None] = None,
        bits_per_sample: Union[int, None] = None,
        subsampling_w: Union[int, None] = None,
        subsampling_h: Union[int, None] = None
    ) -> 'VideoFormat': ...

    @overload
    def __eq__(self, other: 'VideoFormat') -> bool: ...  # type: ignore[misc]

    @overload
    def __eq__(self, other: Any) -> Literal[False]: ...


class FrameProps(MutableMapping[str, _VapourSynthMapValue]):
    def __init__(self) -> NoReturn: ...  # type: ignore[misc]

    def setdefault(
        self, key: str, default: _VapourSynthMapValue = 0
    ) -> _VapourSynthMapValue: ...

    def copy(self) -> 'FrameProps': ...

    def __setattr__(self, name: str, value: _VapourSynthMapValue) -> None: ...

    def __getattr__(self, name: str) -> _VapourSynthMapValue: ...

    def __delattr__(self, name: str) -> None: ...

    def __setitem__(self, name: str, value: _VapourSynthMapValue) -> None: ...

    def __getitem__(self, name: str) -> _VapourSynthMapValue: ...

    def __delitem__(self, name: str) -> None: ...

    def __iter__(self) -> Iterator[str]: ...

    def __len__(self) -> int: ...


class RawFrame:
    def __init__(self) -> NoReturn: ...  # type: ignore[misc]

    @property
    def closed(self) -> bool: ...

    def close(self) -> None: ...

    def copy(self: 'SelfFrame') -> 'SelfFrame': ...

    @property
    def props(self) -> FrameProps: ...

    @props.setter
    def props(self, new_props: MappingProxyType[str, _VapourSynthMapValue]) -> None: ...

    def get_write_ptr(self, plane: int) -> c_void_p: ...

    def get_read_ptr(self, plane: int) -> c_void_p: ...

    def get_stride(self, plane: int) -> int: ...

    @property
    def readonly(self) -> bool: ...

    def __enter__(self: 'SelfFrame') -> 'SelfFrame': ...

    def __exit__(
        self, exc_type: Union[Type[BaseException], None],
        exc_value: Union[BaseException, None],
        traceback: Union[TracebackType, None], /,
    ) -> Union[bool, None]: ...

    def __getitem__(self, index: int) -> memoryview: ...

    def __len__(self) -> int: ...


SelfFrame = TypeVar('SelfFrame', bound=RawFrame)


class VideoFrame(RawFrame):
    format: VideoFormat
    width: int
    height: int

    def _writelines(self, write: Callable[[bytes], None]) -> None: ...


class AudioFrame(RawFrame):
    sample_type: SampleType
    bits_per_sample: int
    bytes_per_sample: int
    channel_layout: int
    num_channels: int


    
# implementation: bm3d

class _Plugin_bm3d_Core_Bound(Plugin):
    """This class implements the module definitions for the "bm3d" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Basic(self, input: 'VideoNode', ref: Optional['VideoNode'] = None, profile: Optional[DataType] = None, sigma: Optional[SingleAndSequence[float]] = None, block_size: Optional[int] = None, block_step: Optional[int] = None, group_size: Optional[int] = None, bm_range: Optional[int] = None, bm_step: Optional[int] = None, th_mse: Optional[float] = None, hard_thr: Optional[float] = None, matrix: Optional[int] = None) -> 'VideoNode': ...
    def Final(self, input: 'VideoNode', ref: 'VideoNode', profile: Optional[DataType] = None, sigma: Optional[SingleAndSequence[float]] = None, block_size: Optional[int] = None, block_step: Optional[int] = None, group_size: Optional[int] = None, bm_range: Optional[int] = None, bm_step: Optional[int] = None, th_mse: Optional[float] = None, matrix: Optional[int] = None) -> 'VideoNode': ...
    def OPP2RGB(self, input: 'VideoNode', sample: Optional[int] = None) -> 'VideoNode': ...
    def RGB2OPP(self, input: 'VideoNode', sample: Optional[int] = None) -> 'VideoNode': ...
    def VAggregate(self, input: 'VideoNode', radius: Optional[int] = None, sample: Optional[int] = None) -> 'VideoNode': ...
    def VBasic(self, input: 'VideoNode', ref: Optional['VideoNode'] = None, profile: Optional[DataType] = None, sigma: Optional[SingleAndSequence[float]] = None, radius: Optional[int] = None, block_size: Optional[int] = None, block_step: Optional[int] = None, group_size: Optional[int] = None, bm_range: Optional[int] = None, bm_step: Optional[int] = None, ps_num: Optional[int] = None, ps_range: Optional[int] = None, ps_step: Optional[int] = None, th_mse: Optional[float] = None, hard_thr: Optional[float] = None, matrix: Optional[int] = None) -> 'VideoNode': ...
    def VFinal(self, input: 'VideoNode', ref: 'VideoNode', profile: Optional[DataType] = None, sigma: Optional[SingleAndSequence[float]] = None, radius: Optional[int] = None, block_size: Optional[int] = None, block_step: Optional[int] = None, group_size: Optional[int] = None, bm_range: Optional[int] = None, bm_step: Optional[int] = None, ps_num: Optional[int] = None, ps_range: Optional[int] = None, ps_step: Optional[int] = None, th_mse: Optional[float] = None, matrix: Optional[int] = None) -> 'VideoNode': ...

class _Plugin_bm3d_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "bm3d" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Basic(self, ref: Optional['VideoNode'] = None, profile: Optional[DataType] = None, sigma: Optional[SingleAndSequence[float]] = None, block_size: Optional[int] = None, block_step: Optional[int] = None, group_size: Optional[int] = None, bm_range: Optional[int] = None, bm_step: Optional[int] = None, th_mse: Optional[float] = None, hard_thr: Optional[float] = None, matrix: Optional[int] = None) -> 'VideoNode': ...
    def Final(self, ref: 'VideoNode', profile: Optional[DataType] = None, sigma: Optional[SingleAndSequence[float]] = None, block_size: Optional[int] = None, block_step: Optional[int] = None, group_size: Optional[int] = None, bm_range: Optional[int] = None, bm_step: Optional[int] = None, th_mse: Optional[float] = None, matrix: Optional[int] = None) -> 'VideoNode': ...
    def OPP2RGB(self, sample: Optional[int] = None) -> 'VideoNode': ...
    def RGB2OPP(self, sample: Optional[int] = None) -> 'VideoNode': ...
    def VAggregate(self, radius: Optional[int] = None, sample: Optional[int] = None) -> 'VideoNode': ...
    def VBasic(self, ref: Optional['VideoNode'] = None, profile: Optional[DataType] = None, sigma: Optional[SingleAndSequence[float]] = None, radius: Optional[int] = None, block_size: Optional[int] = None, block_step: Optional[int] = None, group_size: Optional[int] = None, bm_range: Optional[int] = None, bm_step: Optional[int] = None, ps_num: Optional[int] = None, ps_range: Optional[int] = None, ps_step: Optional[int] = None, th_mse: Optional[float] = None, hard_thr: Optional[float] = None, matrix: Optional[int] = None) -> 'VideoNode': ...
    def VFinal(self, ref: 'VideoNode', profile: Optional[DataType] = None, sigma: Optional[SingleAndSequence[float]] = None, radius: Optional[int] = None, block_size: Optional[int] = None, block_step: Optional[int] = None, group_size: Optional[int] = None, bm_range: Optional[int] = None, bm_step: Optional[int] = None, ps_num: Optional[int] = None, ps_range: Optional[int] = None, ps_step: Optional[int] = None, th_mse: Optional[float] = None, matrix: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: bm3dcpu

class _Plugin_bm3dcpu_Core_Bound(Plugin):
    """This class implements the module definitions for the "bm3dcpu" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(self, clip: 'VideoNode', ref: Optional['VideoNode'] = None, sigma: Optional[SingleAndSequence[float]] = None, block_step: Optional[SingleAndSequence[int]] = None, bm_range: Optional[SingleAndSequence[int]] = None, radius: Optional[int] = None, ps_num: Optional[int] = None, ps_range: Optional[int] = None, chroma: Optional[int] = None) -> 'VideoNode': ...

class _Plugin_bm3dcpu_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "bm3dcpu" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(self, ref: Optional['VideoNode'] = None, sigma: Optional[SingleAndSequence[float]] = None, block_step: Optional[SingleAndSequence[int]] = None, bm_range: Optional[SingleAndSequence[int]] = None, radius: Optional[int] = None, ps_num: Optional[int] = None, ps_range: Optional[int] = None, chroma: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: bm3dcuda

class _Plugin_bm3dcuda_Core_Bound(Plugin):
    """This class implements the module definitions for the "bm3dcuda" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(self, clip: 'VideoNode', ref: Optional['VideoNode'] = None, sigma: Optional[SingleAndSequence[float]] = None, block_step: Optional[SingleAndSequence[int]] = None, bm_range: Optional[SingleAndSequence[int]] = None, radius: Optional[int] = None, ps_num: Optional[SingleAndSequence[int]] = None, ps_range: Optional[SingleAndSequence[int]] = None, chroma: Optional[int] = None, device_id: Optional[int] = None, fast: Optional[int] = None, extractor_exp: Optional[int] = None) -> 'VideoNode': ...

class _Plugin_bm3dcuda_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "bm3dcuda" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(self, ref: Optional['VideoNode'] = None, sigma: Optional[SingleAndSequence[float]] = None, block_step: Optional[SingleAndSequence[int]] = None, bm_range: Optional[SingleAndSequence[int]] = None, radius: Optional[int] = None, ps_num: Optional[SingleAndSequence[int]] = None, ps_range: Optional[SingleAndSequence[int]] = None, chroma: Optional[int] = None, device_id: Optional[int] = None, fast: Optional[int] = None, extractor_exp: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: bm3dcuda_rtc

class _Plugin_bm3dcuda_rtc_Core_Bound(Plugin):
    """This class implements the module definitions for the "bm3dcuda_rtc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(self, clip: 'VideoNode', ref: Optional['VideoNode'] = None, sigma: Optional[SingleAndSequence[float]] = None, block_step: Optional[SingleAndSequence[int]] = None, bm_range: Optional[SingleAndSequence[int]] = None, radius: Optional[int] = None, ps_num: Optional[SingleAndSequence[int]] = None, ps_range: Optional[SingleAndSequence[int]] = None, chroma: Optional[int] = None, device_id: Optional[int] = None, fast: Optional[int] = None, extractor_exp: Optional[int] = None, bm_error_s: Optional[SingleAndSequence[DataType]] = None, transform_2d_s: Optional[SingleAndSequence[DataType]] = None, transform_1d_s: Optional[SingleAndSequence[DataType]] = None) -> 'VideoNode': ...

class _Plugin_bm3dcuda_rtc_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "bm3dcuda_rtc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(self, ref: Optional['VideoNode'] = None, sigma: Optional[SingleAndSequence[float]] = None, block_step: Optional[SingleAndSequence[int]] = None, bm_range: Optional[SingleAndSequence[int]] = None, radius: Optional[int] = None, ps_num: Optional[SingleAndSequence[int]] = None, ps_range: Optional[SingleAndSequence[int]] = None, chroma: Optional[int] = None, device_id: Optional[int] = None, fast: Optional[int] = None, extractor_exp: Optional[int] = None, bm_error_s: Optional[SingleAndSequence[DataType]] = None, transform_2d_s: Optional[SingleAndSequence[DataType]] = None, transform_1d_s: Optional[SingleAndSequence[DataType]] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: dfttest

class _Plugin_dfttest_Core_Bound(Plugin):
    """This class implements the module definitions for the "dfttest" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DFTTest(self, clip: 'VideoNode', ftype: Optional[int] = None, sigma: Optional[float] = None, sigma2: Optional[float] = None, pmin: Optional[float] = None, pmax: Optional[float] = None, sbsize: Optional[int] = None, smode: Optional[int] = None, sosize: Optional[int] = None, tbsize: Optional[int] = None, tmode: Optional[int] = None, tosize: Optional[int] = None, swin: Optional[int] = None, twin: Optional[int] = None, sbeta: Optional[float] = None, tbeta: Optional[float] = None, zmean: Optional[int] = None, f0beta: Optional[float] = None, nlocation: Optional[SingleAndSequence[int]] = None, alpha: Optional[float] = None, slocation: Optional[SingleAndSequence[float]] = None, ssx: Optional[SingleAndSequence[float]] = None, ssy: Optional[SingleAndSequence[float]] = None, sst: Optional[SingleAndSequence[float]] = None, ssystem: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None, opt: Optional[int] = None) -> 'VideoNode': ...

class _Plugin_dfttest_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "dfttest" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DFTTest(self, ftype: Optional[int] = None, sigma: Optional[float] = None, sigma2: Optional[float] = None, pmin: Optional[float] = None, pmax: Optional[float] = None, sbsize: Optional[int] = None, smode: Optional[int] = None, sosize: Optional[int] = None, tbsize: Optional[int] = None, tmode: Optional[int] = None, tosize: Optional[int] = None, swin: Optional[int] = None, twin: Optional[int] = None, sbeta: Optional[float] = None, tbeta: Optional[float] = None, zmean: Optional[int] = None, f0beta: Optional[float] = None, nlocation: Optional[SingleAndSequence[int]] = None, alpha: Optional[float] = None, slocation: Optional[SingleAndSequence[float]] = None, ssx: Optional[SingleAndSequence[float]] = None, ssy: Optional[SingleAndSequence[float]] = None, sst: Optional[SingleAndSequence[float]] = None, ssystem: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None, opt: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: flux

class _Plugin_flux_Core_Bound(Plugin):
    """This class implements the module definitions for the "flux" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def SmoothST(self, clip: 'VideoNode', temporal_threshold: Optional[int] = None, spatial_threshold: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def SmoothT(self, clip: 'VideoNode', temporal_threshold: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...

class _Plugin_flux_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "flux" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def SmoothST(self, temporal_threshold: Optional[int] = None, spatial_threshold: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def SmoothT(self, temporal_threshold: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: knlm

class _Plugin_knlm_Core_Bound(Plugin):
    """This class implements the module definitions for the "knlm" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def KNLMeansCL(self, clip: 'VideoNode', d: Optional[int] = None, a: Optional[int] = None, s: Optional[int] = None, h: Optional[float] = None, channels: Optional[DataType] = None, wmode: Optional[int] = None, wref: Optional[float] = None, rclip: Optional['VideoNode'] = None, device_type: Optional[DataType] = None, device_id: Optional[int] = None, ocl_x: Optional[int] = None, ocl_y: Optional[int] = None, ocl_r: Optional[int] = None, info: Optional[int] = None) -> 'VideoNode': ...

class _Plugin_knlm_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "knlm" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def KNLMeansCL(self, d: Optional[int] = None, a: Optional[int] = None, s: Optional[int] = None, h: Optional[float] = None, channels: Optional[DataType] = None, wmode: Optional[int] = None, wref: Optional[float] = None, rclip: Optional['VideoNode'] = None, device_type: Optional[DataType] = None, device_id: Optional[int] = None, ocl_x: Optional[int] = None, ocl_y: Optional[int] = None, ocl_r: Optional[int] = None, info: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: mv

class _Plugin_mv_Core_Bound(Plugin):
    """This class implements the module definitions for the "mv" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Analyse(self, super: 'VideoNode', blksize: Optional[int] = None, blksizev: Optional[int] = None, levels: Optional[int] = None, search: Optional[int] = None, searchparam: Optional[int] = None, pelsearch: Optional[int] = None, isb: Optional[int] = None, lambda_: Optional[int] = None, chroma: Optional[int] = None, delta: Optional[int] = None, truemotion: Optional[int] = None, lsad: Optional[int] = None, plevel: Optional[int] = None, global_: Optional[int] = None, pnew: Optional[int] = None, pzero: Optional[int] = None, pglobal: Optional[int] = None, overlap: Optional[int] = None, overlapv: Optional[int] = None, divide: Optional[int] = None, badsad: Optional[int] = None, badrange: Optional[int] = None, opt: Optional[int] = None, meander: Optional[int] = None, trymany: Optional[int] = None, fields: Optional[int] = None, tff: Optional[int] = None, search_coarse: Optional[int] = None, dct: Optional[int] = None) -> 'VideoNode': ...
    def BlockFPS(self, clip: 'VideoNode', super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', num: Optional[int] = None, den: Optional[int] = None, mode: Optional[int] = None, ml: Optional[float] = None, blend: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Compensate(self, clip: 'VideoNode', super: 'VideoNode', vectors: 'VideoNode', scbehavior: Optional[int] = None, thsad: Optional[int] = None, fields: Optional[int] = None, time: Optional[float] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None, tff: Optional[int] = None) -> 'VideoNode': ...
    def Degrain1(self, clip: 'VideoNode', super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', thsad: Optional[int] = None, thsadc: Optional[int] = None, plane: Optional[int] = None, limit: Optional[int] = None, limitc: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Degrain2(self, clip: 'VideoNode', super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', mvbw2: 'VideoNode', mvfw2: 'VideoNode', thsad: Optional[int] = None, thsadc: Optional[int] = None, plane: Optional[int] = None, limit: Optional[int] = None, limitc: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Degrain3(self, clip: 'VideoNode', super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', mvbw2: 'VideoNode', mvfw2: 'VideoNode', mvbw3: 'VideoNode', mvfw3: 'VideoNode', thsad: Optional[int] = None, thsadc: Optional[int] = None, plane: Optional[int] = None, limit: Optional[int] = None, limitc: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def DepanAnalyse(self, clip: 'VideoNode', vectors: 'VideoNode', mask: Optional['VideoNode'] = None, zoom: Optional[int] = None, rot: Optional[int] = None, pixaspect: Optional[float] = None, error: Optional[float] = None, info: Optional[int] = None, wrong: Optional[float] = None, zerow: Optional[float] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, fields: Optional[int] = None, tff: Optional[int] = None) -> 'VideoNode': ...
    def DepanCompensate(self, clip: 'VideoNode', data: 'VideoNode', offset: Optional[float] = None, subpixel: Optional[int] = None, pixaspect: Optional[float] = None, matchfields: Optional[int] = None, mirror: Optional[int] = None, blur: Optional[int] = None, info: Optional[int] = None, fields: Optional[int] = None, tff: Optional[int] = None) -> 'VideoNode': ...
    def DepanEstimate(self, clip: 'VideoNode', trust: Optional[float] = None, winx: Optional[int] = None, winy: Optional[int] = None, wleft: Optional[int] = None, wtop: Optional[int] = None, dxmax: Optional[int] = None, dymax: Optional[int] = None, zoommax: Optional[float] = None, stab: Optional[float] = None, pixaspect: Optional[float] = None, info: Optional[int] = None, show: Optional[int] = None, fields: Optional[int] = None, tff: Optional[int] = None) -> 'VideoNode': ...
    def DepanStabilise(self, clip: 'VideoNode', data: 'VideoNode', cutoff: Optional[float] = None, damping: Optional[float] = None, initzoom: Optional[float] = None, addzoom: Optional[int] = None, prev: Optional[int] = None, next: Optional[int] = None, mirror: Optional[int] = None, blur: Optional[int] = None, dxmax: Optional[float] = None, dymax: Optional[float] = None, zoommax: Optional[float] = None, rotmax: Optional[float] = None, subpixel: Optional[int] = None, pixaspect: Optional[float] = None, fitlast: Optional[int] = None, tzoom: Optional[float] = None, info: Optional[int] = None, method: Optional[int] = None, fields: Optional[int] = None) -> 'VideoNode': ...
    def Finest(self, super: 'VideoNode', opt: Optional[int] = None) -> 'VideoNode': ...
    def Flow(self, clip: 'VideoNode', super: 'VideoNode', vectors: 'VideoNode', time: Optional[float] = None, mode: Optional[int] = None, fields: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None, tff: Optional[int] = None) -> 'VideoNode': ...
    def FlowBlur(self, clip: 'VideoNode', super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', blur: Optional[float] = None, prec: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def FlowFPS(self, clip: 'VideoNode', super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', num: Optional[int] = None, den: Optional[int] = None, mask: Optional[int] = None, ml: Optional[float] = None, blend: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def FlowInter(self, clip: 'VideoNode', super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', time: Optional[float] = None, ml: Optional[float] = None, blend: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Mask(self, clip: 'VideoNode', vectors: 'VideoNode', ml: Optional[float] = None, gamma: Optional[float] = None, kind: Optional[int] = None, time: Optional[float] = None, ysc: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Recalculate(self, super: 'VideoNode', vectors: 'VideoNode', thsad: Optional[int] = None, smooth: Optional[int] = None, blksize: Optional[int] = None, blksizev: Optional[int] = None, search: Optional[int] = None, searchparam: Optional[int] = None, lambda_: Optional[int] = None, chroma: Optional[int] = None, truemotion: Optional[int] = None, pnew: Optional[int] = None, overlap: Optional[int] = None, overlapv: Optional[int] = None, divide: Optional[int] = None, opt: Optional[int] = None, meander: Optional[int] = None, fields: Optional[int] = None, tff: Optional[int] = None, dct: Optional[int] = None) -> 'VideoNode': ...
    def SCDetection(self, clip: 'VideoNode', vectors: 'VideoNode', thscd1: Optional[int] = None, thscd2: Optional[int] = None) -> 'VideoNode': ...
    def Super(self, clip: 'VideoNode', hpad: Optional[int] = None, vpad: Optional[int] = None, pel: Optional[int] = None, levels: Optional[int] = None, chroma: Optional[int] = None, sharp: Optional[int] = None, rfilter: Optional[int] = None, pelclip: Optional['VideoNode'] = None, opt: Optional[int] = None) -> 'VideoNode': ...

class _Plugin_mv_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "mv" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Analyse(self, blksize: Optional[int] = None, blksizev: Optional[int] = None, levels: Optional[int] = None, search: Optional[int] = None, searchparam: Optional[int] = None, pelsearch: Optional[int] = None, isb: Optional[int] = None, lambda_: Optional[int] = None, chroma: Optional[int] = None, delta: Optional[int] = None, truemotion: Optional[int] = None, lsad: Optional[int] = None, plevel: Optional[int] = None, global_: Optional[int] = None, pnew: Optional[int] = None, pzero: Optional[int] = None, pglobal: Optional[int] = None, overlap: Optional[int] = None, overlapv: Optional[int] = None, divide: Optional[int] = None, badsad: Optional[int] = None, badrange: Optional[int] = None, opt: Optional[int] = None, meander: Optional[int] = None, trymany: Optional[int] = None, fields: Optional[int] = None, tff: Optional[int] = None, search_coarse: Optional[int] = None, dct: Optional[int] = None) -> 'VideoNode': ...
    def BlockFPS(self, super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', num: Optional[int] = None, den: Optional[int] = None, mode: Optional[int] = None, ml: Optional[float] = None, blend: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Compensate(self, super: 'VideoNode', vectors: 'VideoNode', scbehavior: Optional[int] = None, thsad: Optional[int] = None, fields: Optional[int] = None, time: Optional[float] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None, tff: Optional[int] = None) -> 'VideoNode': ...
    def Degrain1(self, super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', thsad: Optional[int] = None, thsadc: Optional[int] = None, plane: Optional[int] = None, limit: Optional[int] = None, limitc: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Degrain2(self, super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', mvbw2: 'VideoNode', mvfw2: 'VideoNode', thsad: Optional[int] = None, thsadc: Optional[int] = None, plane: Optional[int] = None, limit: Optional[int] = None, limitc: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Degrain3(self, super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', mvbw2: 'VideoNode', mvfw2: 'VideoNode', mvbw3: 'VideoNode', mvfw3: 'VideoNode', thsad: Optional[int] = None, thsadc: Optional[int] = None, plane: Optional[int] = None, limit: Optional[int] = None, limitc: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def DepanAnalyse(self, vectors: 'VideoNode', mask: Optional['VideoNode'] = None, zoom: Optional[int] = None, rot: Optional[int] = None, pixaspect: Optional[float] = None, error: Optional[float] = None, info: Optional[int] = None, wrong: Optional[float] = None, zerow: Optional[float] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, fields: Optional[int] = None, tff: Optional[int] = None) -> 'VideoNode': ...
    def DepanCompensate(self, data: 'VideoNode', offset: Optional[float] = None, subpixel: Optional[int] = None, pixaspect: Optional[float] = None, matchfields: Optional[int] = None, mirror: Optional[int] = None, blur: Optional[int] = None, info: Optional[int] = None, fields: Optional[int] = None, tff: Optional[int] = None) -> 'VideoNode': ...
    def DepanEstimate(self, trust: Optional[float] = None, winx: Optional[int] = None, winy: Optional[int] = None, wleft: Optional[int] = None, wtop: Optional[int] = None, dxmax: Optional[int] = None, dymax: Optional[int] = None, zoommax: Optional[float] = None, stab: Optional[float] = None, pixaspect: Optional[float] = None, info: Optional[int] = None, show: Optional[int] = None, fields: Optional[int] = None, tff: Optional[int] = None) -> 'VideoNode': ...
    def DepanStabilise(self, data: 'VideoNode', cutoff: Optional[float] = None, damping: Optional[float] = None, initzoom: Optional[float] = None, addzoom: Optional[int] = None, prev: Optional[int] = None, next: Optional[int] = None, mirror: Optional[int] = None, blur: Optional[int] = None, dxmax: Optional[float] = None, dymax: Optional[float] = None, zoommax: Optional[float] = None, rotmax: Optional[float] = None, subpixel: Optional[int] = None, pixaspect: Optional[float] = None, fitlast: Optional[int] = None, tzoom: Optional[float] = None, info: Optional[int] = None, method: Optional[int] = None, fields: Optional[int] = None) -> 'VideoNode': ...
    def Finest(self, opt: Optional[int] = None) -> 'VideoNode': ...
    def Flow(self, super: 'VideoNode', vectors: 'VideoNode', time: Optional[float] = None, mode: Optional[int] = None, fields: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None, tff: Optional[int] = None) -> 'VideoNode': ...
    def FlowBlur(self, super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', blur: Optional[float] = None, prec: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def FlowFPS(self, super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', num: Optional[int] = None, den: Optional[int] = None, mask: Optional[int] = None, ml: Optional[float] = None, blend: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def FlowInter(self, super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', time: Optional[float] = None, ml: Optional[float] = None, blend: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Mask(self, vectors: 'VideoNode', ml: Optional[float] = None, gamma: Optional[float] = None, kind: Optional[int] = None, time: Optional[float] = None, ysc: Optional[int] = None, thscd1: Optional[int] = None, thscd2: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Recalculate(self, vectors: 'VideoNode', thsad: Optional[int] = None, smooth: Optional[int] = None, blksize: Optional[int] = None, blksizev: Optional[int] = None, search: Optional[int] = None, searchparam: Optional[int] = None, lambda_: Optional[int] = None, chroma: Optional[int] = None, truemotion: Optional[int] = None, pnew: Optional[int] = None, overlap: Optional[int] = None, overlapv: Optional[int] = None, divide: Optional[int] = None, opt: Optional[int] = None, meander: Optional[int] = None, fields: Optional[int] = None, tff: Optional[int] = None, dct: Optional[int] = None) -> 'VideoNode': ...
    def SCDetection(self, vectors: 'VideoNode', thscd1: Optional[int] = None, thscd2: Optional[int] = None) -> 'VideoNode': ...
    def Super(self, hpad: Optional[int] = None, vpad: Optional[int] = None, pel: Optional[int] = None, levels: Optional[int] = None, chroma: Optional[int] = None, sharp: Optional[int] = None, rfilter: Optional[int] = None, pelclip: Optional['VideoNode'] = None, opt: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: mvsf

class _Plugin_mvsf_Core_Bound(Plugin):
    """This class implements the module definitions for the "mvsf" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Analyze(self, super: 'VideoNode', radius: Optional[int] = None, blksize: Optional[int] = None, blksizev: Optional[int] = None, levels: Optional[int] = None, search: Optional[int] = None, searchparam: Optional[int] = None, pelsearch: Optional[int] = None, isb: Optional[int] = None, lambda_: Optional[float] = None, chroma: Optional[int] = None, delta: Optional[int] = None, truemotion: Optional[int] = None, lsad: Optional[float] = None, plevel: Optional[int] = None, global_: Optional[int] = None, pnew: Optional[int] = None, pzero: Optional[int] = None, pglobal: Optional[int] = None, overlap: Optional[int] = None, overlapv: Optional[int] = None, divide: Optional[int] = None, badsad: Optional[float] = None, badrange: Optional[int] = None, meander: Optional[int] = None, trymany: Optional[int] = None, fields: Optional[int] = None, tff: Optional[int] = None, search_coarse: Optional[int] = None, dct: Optional[int] = None) -> 'VideoNode': ...
    def BlockFPS(self, clip: 'VideoNode', super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', num: Optional[int] = None, den: Optional[int] = None, mode: Optional[int] = None, ml: Optional[float] = None, blend: Optional[int] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None) -> 'VideoNode': ...
    def Compensate(self, clip: 'VideoNode', super: 'VideoNode', vectors: 'VideoNode', cclip: Optional['VideoNode'] = None, scbehavior: Optional[int] = None, thsad: Optional[float] = None, thsad2: Optional[float] = None, fields: Optional[int] = None, time: Optional[float] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None, tff: Optional[int] = None) -> 'VideoNode': ...
    def Degrain(self, clip: 'VideoNode', super: 'VideoNode', mvmulti: 'VideoNode', thsad: Optional[SingleAndSequence[float]] = None, thsad2: Optional[SingleAndSequence[float]] = None, plane: Optional[int] = None, limit: Optional[SingleAndSequence[float]] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None) -> 'VideoNode': ...
    def Finest(self, super: 'VideoNode') -> 'VideoNode': ...
    def Flow(self, clip: 'VideoNode', super: 'VideoNode', vectors: 'VideoNode', cclip: Optional['VideoNode'] = None, time: Optional[float] = None, mode: Optional[int] = None, fields: Optional[int] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None, tff: Optional[int] = None) -> 'VideoNode': ...
    def FlowBlur(self, clip: 'VideoNode', super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', blur: Optional[float] = None, prec: Optional[int] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None) -> 'VideoNode': ...
    def FlowFPS(self, clip: 'VideoNode', super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', num: Optional[int] = None, den: Optional[int] = None, mask: Optional[int] = None, ml: Optional[float] = None, blend: Optional[int] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None) -> 'VideoNode': ...
    def FlowInter(self, clip: 'VideoNode', super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', time: Optional[float] = None, ml: Optional[float] = None, blend: Optional[int] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None) -> 'VideoNode': ...
    def Mask(self, clip: 'VideoNode', vectors: 'VideoNode', ml: Optional[float] = None, gamma: Optional[float] = None, kind: Optional[int] = None, time: Optional[float] = None, ysc: Optional[float] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None) -> 'VideoNode': ...
    def Recalculate(self, super: 'VideoNode', vectors: 'VideoNode', thsad: Optional[float] = None, smooth: Optional[int] = None, blksize: Optional[int] = None, blksizev: Optional[int] = None, search: Optional[int] = None, searchparam: Optional[int] = None, lambda_: Optional[float] = None, chroma: Optional[int] = None, truemotion: Optional[int] = None, pnew: Optional[int] = None, overlap: Optional[int] = None, overlapv: Optional[int] = None, divide: Optional[int] = None, meander: Optional[int] = None, fields: Optional[int] = None, tff: Optional[int] = None, dct: Optional[int] = None) -> 'VideoNode': ...
    def SCDetection(self, clip: 'VideoNode', vectors: 'VideoNode', thscd1: Optional[float] = None, thscd2: Optional[float] = None) -> 'VideoNode': ...
    def Super(self, clip: 'VideoNode', hpad: Optional[int] = None, vpad: Optional[int] = None, pel: Optional[int] = None, levels: Optional[int] = None, chroma: Optional[int] = None, sharp: Optional[int] = None, rfilter: Optional[int] = None, pelclip: Optional['VideoNode'] = None) -> 'VideoNode': ...

class _Plugin_mvsf_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "mvsf" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Analyze(self, radius: Optional[int] = None, blksize: Optional[int] = None, blksizev: Optional[int] = None, levels: Optional[int] = None, search: Optional[int] = None, searchparam: Optional[int] = None, pelsearch: Optional[int] = None, isb: Optional[int] = None, lambda_: Optional[float] = None, chroma: Optional[int] = None, delta: Optional[int] = None, truemotion: Optional[int] = None, lsad: Optional[float] = None, plevel: Optional[int] = None, global_: Optional[int] = None, pnew: Optional[int] = None, pzero: Optional[int] = None, pglobal: Optional[int] = None, overlap: Optional[int] = None, overlapv: Optional[int] = None, divide: Optional[int] = None, badsad: Optional[float] = None, badrange: Optional[int] = None, meander: Optional[int] = None, trymany: Optional[int] = None, fields: Optional[int] = None, tff: Optional[int] = None, search_coarse: Optional[int] = None, dct: Optional[int] = None) -> 'VideoNode': ...
    def BlockFPS(self, super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', num: Optional[int] = None, den: Optional[int] = None, mode: Optional[int] = None, ml: Optional[float] = None, blend: Optional[int] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None) -> 'VideoNode': ...
    def Compensate(self, super: 'VideoNode', vectors: 'VideoNode', cclip: Optional['VideoNode'] = None, scbehavior: Optional[int] = None, thsad: Optional[float] = None, thsad2: Optional[float] = None, fields: Optional[int] = None, time: Optional[float] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None, tff: Optional[int] = None) -> 'VideoNode': ...
    def Degrain(self, super: 'VideoNode', mvmulti: 'VideoNode', thsad: Optional[SingleAndSequence[float]] = None, thsad2: Optional[SingleAndSequence[float]] = None, plane: Optional[int] = None, limit: Optional[SingleAndSequence[float]] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None) -> 'VideoNode': ...
    def Finest(self) -> 'VideoNode': ...
    def Flow(self, super: 'VideoNode', vectors: 'VideoNode', cclip: Optional['VideoNode'] = None, time: Optional[float] = None, mode: Optional[int] = None, fields: Optional[int] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None, tff: Optional[int] = None) -> 'VideoNode': ...
    def FlowBlur(self, super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', blur: Optional[float] = None, prec: Optional[int] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None) -> 'VideoNode': ...
    def FlowFPS(self, super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', num: Optional[int] = None, den: Optional[int] = None, mask: Optional[int] = None, ml: Optional[float] = None, blend: Optional[int] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None) -> 'VideoNode': ...
    def FlowInter(self, super: 'VideoNode', mvbw: 'VideoNode', mvfw: 'VideoNode', time: Optional[float] = None, ml: Optional[float] = None, blend: Optional[int] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None) -> 'VideoNode': ...
    def Mask(self, vectors: 'VideoNode', ml: Optional[float] = None, gamma: Optional[float] = None, kind: Optional[int] = None, time: Optional[float] = None, ysc: Optional[float] = None, thscd1: Optional[float] = None, thscd2: Optional[float] = None) -> 'VideoNode': ...
    def Recalculate(self, vectors: 'VideoNode', thsad: Optional[float] = None, smooth: Optional[int] = None, blksize: Optional[int] = None, blksizev: Optional[int] = None, search: Optional[int] = None, searchparam: Optional[int] = None, lambda_: Optional[float] = None, chroma: Optional[int] = None, truemotion: Optional[int] = None, pnew: Optional[int] = None, overlap: Optional[int] = None, overlapv: Optional[int] = None, divide: Optional[int] = None, meander: Optional[int] = None, fields: Optional[int] = None, tff: Optional[int] = None, dct: Optional[int] = None) -> 'VideoNode': ...
    def SCDetection(self, vectors: 'VideoNode', thscd1: Optional[float] = None, thscd2: Optional[float] = None) -> 'VideoNode': ...
    def Super(self, hpad: Optional[int] = None, vpad: Optional[int] = None, pel: Optional[int] = None, levels: Optional[int] = None, chroma: Optional[int] = None, sharp: Optional[int] = None, rfilter: Optional[int] = None, pelclip: Optional['VideoNode'] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: nnedi3

class _Plugin_nnedi3_Core_Bound(Plugin):
    """This class implements the module definitions for the "nnedi3" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def nnedi3(self, clip: 'VideoNode', field: int, dh: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None, nsize: Optional[int] = None, nns: Optional[int] = None, qual: Optional[int] = None, etype: Optional[int] = None, pscrn: Optional[int] = None, opt: Optional[int] = None, int16_prescreener: Optional[int] = None, int16_predictor: Optional[int] = None, exp: Optional[int] = None, show_mask: Optional[int] = None, combed_only: Optional[int] = None) -> 'VideoNode': ...

class _Plugin_nnedi3_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "nnedi3" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def nnedi3(self, field: int, dh: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None, nsize: Optional[int] = None, nns: Optional[int] = None, qual: Optional[int] = None, etype: Optional[int] = None, pscrn: Optional[int] = None, opt: Optional[int] = None, int16_prescreener: Optional[int] = None, int16_predictor: Optional[int] = None, exp: Optional[int] = None, show_mask: Optional[int] = None, combed_only: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: nnedi3cl

class _Plugin_nnedi3cl_Core_Bound(Plugin):
    """This class implements the module definitions for the "nnedi3cl" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def NNEDI3CL(self, clip: 'VideoNode', field: int, dh: Optional[int] = None, dw: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None, nsize: Optional[int] = None, nns: Optional[int] = None, qual: Optional[int] = None, etype: Optional[int] = None, pscrn: Optional[int] = None, device: Optional[int] = None, list_device: Optional[int] = None, info: Optional[int] = None) -> 'VideoNode': ...

class _Plugin_nnedi3cl_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "nnedi3cl" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def NNEDI3CL(self, field: int, dh: Optional[int] = None, dw: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None, nsize: Optional[int] = None, nns: Optional[int] = None, qual: Optional[int] = None, etype: Optional[int] = None, pscrn: Optional[int] = None, device: Optional[int] = None, list_device: Optional[int] = None, info: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: resize

class _Plugin_resize_Core_Bound(Plugin):
    """This class implements the module definitions for the "resize" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...
    def Bilinear(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...
    def Bob(self, clip: 'VideoNode', filter: Optional[DataType] = None, tff: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...
    def Lanczos(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...
    def Point(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...
    def Spline16(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...
    def Spline36(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...
    def Spline64(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...

class _Plugin_resize_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "resize" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...
    def Bilinear(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...
    def Bob(self, filter: Optional[DataType] = None, tff: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...
    def Lanczos(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...
    def Point(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...
    def Spline16(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...
    def Spline36(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...
    def Spline64(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: retinex

class _Plugin_retinex_Core_Bound(Plugin):
    """This class implements the module definitions for the "retinex" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def MSRCP(self, input: 'VideoNode', sigma: Optional[SingleAndSequence[float]] = None, lower_thr: Optional[float] = None, upper_thr: Optional[float] = None, fulls: Optional[int] = None, fulld: Optional[int] = None, chroma_protect: Optional[float] = None) -> 'VideoNode': ...
    def MSRCR(self, input: 'VideoNode', sigma: Optional[SingleAndSequence[float]] = None, lower_thr: Optional[float] = None, upper_thr: Optional[float] = None, fulls: Optional[int] = None, fulld: Optional[int] = None, restore: Optional[float] = None) -> 'VideoNode': ...

class _Plugin_retinex_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "retinex" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def MSRCP(self, sigma: Optional[SingleAndSequence[float]] = None, lower_thr: Optional[float] = None, upper_thr: Optional[float] = None, fulls: Optional[int] = None, fulld: Optional[int] = None, chroma_protect: Optional[float] = None) -> 'VideoNode': ...
    def MSRCR(self, sigma: Optional[SingleAndSequence[float]] = None, lower_thr: Optional[float] = None, upper_thr: Optional[float] = None, fulls: Optional[int] = None, fulld: Optional[int] = None, restore: Optional[float] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: std

class _Plugin_std_Core_Bound(Plugin):
    """This class implements the module definitions for the "std" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AddBorders(self, clip: 'VideoNode', left: Optional[int] = None, right: Optional[int] = None, top: Optional[int] = None, bottom: Optional[int] = None, color: Optional[SingleAndSequence[float]] = None) -> 'VideoNode': ...
    def AssumeFPS(self, clip: 'VideoNode', src: Optional['VideoNode'] = None, fpsnum: Optional[int] = None, fpsden: Optional[int] = None) -> 'VideoNode': ...
    def AssumeSampleRate(self, clip: 'AudioNode', src: Optional['AudioNode'] = None, samplerate: Optional[int] = None) -> 'AudioNode': ...
    def AudioGain(self, clip: 'AudioNode', gain: Optional[SingleAndSequence[float]] = None) -> 'AudioNode': ...
    def AudioLoop(self, clip: 'AudioNode', times: Optional[int] = None) -> 'AudioNode': ...
    def AudioMix(self, clips: SingleAndSequence['AudioNode'], matrix: SingleAndSequence[float], channels_out: SingleAndSequence[int]) -> 'AudioNode': ...
    def AudioReverse(self, clip: 'AudioNode') -> 'AudioNode': ...
    def AudioSplice(self, clips: SingleAndSequence['AudioNode']) -> 'AudioNode': ...
    def AudioTrim(self, clip: 'AudioNode', first: Optional[int] = None, last: Optional[int] = None, length: Optional[int] = None) -> 'AudioNode': ...
    def AverageFrames(self, clips: SingleAndSequence['VideoNode'], weights: SingleAndSequence[float], scale: Optional[float] = None, scenechange: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Binarize(self, clip: 'VideoNode', threshold: Optional[SingleAndSequence[float]] = None, v0: Optional[SingleAndSequence[float]] = None, v1: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def BinarizeMask(self, clip: 'VideoNode', threshold: Optional[SingleAndSequence[float]] = None, v0: Optional[SingleAndSequence[float]] = None, v1: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def BlankAudio(self, clip: Optional['AudioNode'] = None, channels: Optional[int] = None, bits: Optional[int] = None, sampletype: Optional[int] = None, samplerate: Optional[int] = None, length: Optional[int] = None, keep: Optional[int] = None) -> 'AudioNode': ...
    def BlankClip(self, clip: Optional['VideoNode'] = None, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, length: Optional[int] = None, fpsnum: Optional[int] = None, fpsden: Optional[int] = None, color: Optional[SingleAndSequence[float]] = None, keep: Optional[int] = None) -> 'VideoNode': ...
    def BoxBlur(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, hradius: Optional[int] = None, hpasses: Optional[int] = None, vradius: Optional[int] = None, vpasses: Optional[int] = None) -> 'VideoNode': ...
    def Cache(self, clip: 'VideoNode', size: Optional[int] = None, fixed: Optional[int] = None, make_linear: Optional[int] = None) -> 'VideoNode': ...
    def ClipToProp(self, clip: 'VideoNode', mclip: 'VideoNode', prop: Optional[DataType] = None) -> 'VideoNode': ...
    def Convolution(self, clip: 'VideoNode', matrix: SingleAndSequence[float], bias: Optional[float] = None, divisor: Optional[float] = None, planes: Optional[SingleAndSequence[int]] = None, saturate: Optional[int] = None, mode: Optional[DataType] = None) -> 'VideoNode': ...
    def CopyFrameProps(self, clip: 'VideoNode', prop_src: 'VideoNode') -> 'VideoNode': ...
    def Crop(self, clip: 'VideoNode', left: Optional[int] = None, right: Optional[int] = None, top: Optional[int] = None, bottom: Optional[int] = None) -> 'VideoNode': ...
    def CropAbs(self, clip: 'VideoNode', width: int, height: int, left: Optional[int] = None, top: Optional[int] = None, x: Optional[int] = None, y: Optional[int] = None) -> 'VideoNode': ...
    def CropRel(self, clip: 'VideoNode', left: Optional[int] = None, right: Optional[int] = None, top: Optional[int] = None, bottom: Optional[int] = None) -> 'VideoNode': ...
    def Deflate(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None) -> 'VideoNode': ...
    def DeleteFrames(self, clip: 'VideoNode', frames: SingleAndSequence[int]) -> 'VideoNode': ...
    def DoubleWeave(self, clip: 'VideoNode', tff: Optional[int] = None) -> 'VideoNode': ...
    def DuplicateFrames(self, clip: 'VideoNode', frames: SingleAndSequence[int]) -> 'VideoNode': ...
    def Expr(self, clips: SingleAndSequence['VideoNode'], expr: SingleAndSequence[DataType], format: Optional[int] = None) -> 'VideoNode': ...
    def FlipHorizontal(self, clip: 'VideoNode') -> 'VideoNode': ...
    def FlipVertical(self, clip: 'VideoNode') -> 'VideoNode': ...
    def FrameEval(self, clip: 'VideoNode', eval: VSMapValueCallback[VideoNode], prop_src: Optional[SingleAndSequence[VideoNode]] = None, clip_src: Optional[SingleAndSequence[VideoNode]] = None) -> 'VideoNode': ...
    def FreezeFrames(self, clip: 'VideoNode', first: SingleAndSequence[int], last: SingleAndSequence[int], replacement: SingleAndSequence[int]) -> 'VideoNode': ...
    def Inflate(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None) -> 'VideoNode': ...
    def Interleave(self, clips: SingleAndSequence['VideoNode'], extend: Optional[int] = None, mismatch: Optional[int] = None, modify_duration: Optional[int] = None) -> 'VideoNode': ...
    def Invert(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def InvertMask(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Levels(self, clip: 'VideoNode', min_in: Optional[SingleAndSequence[float]] = None, max_in: Optional[SingleAndSequence[float]] = None, gamma: Optional[SingleAndSequence[float]] = None, min_out: Optional[SingleAndSequence[float]] = None, max_out: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Limiter(self, clip: 'VideoNode', min: Optional[SingleAndSequence[float]] = None, max: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def LoadAllPlugins(self, path: DataType) -> None: ...
    def LoadPlugin(self, path: DataType, altsearchpath: Optional[int] = None, forcens: Optional[DataType] = None, forceid: Optional[DataType] = None) -> None: ...
    def Loop(self, clip: 'VideoNode', times: Optional[int] = None) -> 'VideoNode': ...
    def Lut(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, lut: Optional[SingleAndSequence[int]] = None, lutf: Optional[SingleAndSequence[float]] = None, function: Optional[VSMapValueCallback[Union[int, float]]] = None, bits: Optional[int] = None, floatout: Optional[int] = None) -> 'VideoNode': ...
    def Lut2(self, clipa: 'VideoNode', clipb: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, lut: Optional[SingleAndSequence[int]] = None, lutf: Optional[SingleAndSequence[float]] = None, function: Optional[VSMapValueCallback[Union[int, float]]] = None, bits: Optional[int] = None, floatout: Optional[int] = None) -> 'VideoNode': ...
    def MakeDiff(self, clipa: 'VideoNode', clipb: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def MaskedMerge(self, clipa: 'VideoNode', clipb: 'VideoNode', mask: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, first_plane: Optional[int] = None, premultiplied: Optional[int] = None) -> 'VideoNode': ...
    def Maximum(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None, coordinates: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Median(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Merge(self, clipa: 'VideoNode', clipb: 'VideoNode', weight: Optional[SingleAndSequence[float]] = None) -> 'VideoNode': ...
    def MergeDiff(self, clipa: 'VideoNode', clipb: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Minimum(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None, coordinates: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def ModifyFrame(self, clip: 'VideoNode', clips: SingleAndSequence['VideoNode'], selector: VSMapValueCallback[VideoFrame]) -> 'VideoNode': ...
    def PEMVerifier(self, clip: 'VideoNode', upper: Optional[SingleAndSequence[float]] = None, lower: Optional[SingleAndSequence[float]] = None) -> 'VideoNode': ...
    def PlaneStats(self, clipa: 'VideoNode', clipb: Optional['VideoNode'] = None, plane: Optional[int] = None, prop: Optional[DataType] = None) -> 'VideoNode': ...
    def PreMultiply(self, clip: 'VideoNode', alpha: 'VideoNode') -> 'VideoNode': ...
    def Prewitt(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, scale: Optional[float] = None) -> 'VideoNode': ...
    def PropToClip(self, clip: 'VideoNode', prop: Optional[DataType] = None) -> 'VideoNode': ...
    def RemoveFrameProps(self, clip: 'VideoNode', props: Optional[SingleAndSequence[DataType]] = None) -> 'VideoNode': ...
    def Reverse(self, clip: 'VideoNode') -> 'VideoNode': ...
    def SelectEvery(self, clip: 'VideoNode', cycle: int, offsets: SingleAndSequence[int], modify_duration: Optional[int] = None) -> 'VideoNode': ...
    def SeparateFields(self, clip: 'VideoNode', tff: Optional[int] = None, modify_duration: Optional[int] = None) -> 'VideoNode': ...
    def SetAudioCache(self, clip: 'AudioNode', mode: Optional[int] = None, fixedsize: Optional[int] = None, maxsize: Optional[int] = None, maxhistory: Optional[int] = None) -> None: ...
    def SetFieldBased(self, clip: 'VideoNode', value: int) -> 'VideoNode': ...
    def SetFrameProp(self, clip: 'VideoNode', prop: DataType, intval: Optional[SingleAndSequence[int]] = None, floatval: Optional[SingleAndSequence[float]] = None, data: Optional[SingleAndSequence[DataType]] = None) -> 'VideoNode': ...
    def SetFrameProps(self, clip: 'VideoNode', **kwargs: _VapourSynthMapValue) -> 'VideoNode': ...
    def SetMaxCPU(self, cpu: DataType) -> DataType: ...
    def SetVideoCache(self, clip: 'VideoNode', mode: Optional[int] = None, fixedsize: Optional[int] = None, maxsize: Optional[int] = None, maxhistory: Optional[int] = None) -> None: ...
    def ShuffleChannels(self, clips: SingleAndSequence['AudioNode'], channels_in: SingleAndSequence[int], channels_out: SingleAndSequence[int]) -> 'AudioNode': ...
    def ShufflePlanes(self, clips: SingleAndSequence['VideoNode'], planes: SingleAndSequence[int], colorfamily: int) -> 'VideoNode': ...
    def Sobel(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, scale: Optional[float] = None) -> 'VideoNode': ...
    def Splice(self, clips: SingleAndSequence['VideoNode'], mismatch: Optional[int] = None) -> 'VideoNode': ...
    def SplitChannels(self, clip: 'AudioNode') -> SingleAndSequence['AudioNode']: ...
    def SplitPlanes(self, clip: 'VideoNode') -> SingleAndSequence['VideoNode']: ...
    def StackHorizontal(self, clips: SingleAndSequence['VideoNode']) -> 'VideoNode': ...
    def StackVertical(self, clips: SingleAndSequence['VideoNode']) -> 'VideoNode': ...
    def TestAudio(self, channels: Optional[int] = None, bits: Optional[int] = None, isfloat: Optional[int] = None, samplerate: Optional[int] = None, length: Optional[int] = None) -> 'AudioNode': ...
    def Transpose(self, clip: 'VideoNode') -> 'VideoNode': ...
    def Trim(self, clip: 'VideoNode', first: Optional[int] = None, last: Optional[int] = None, length: Optional[int] = None) -> 'VideoNode': ...
    def Turn180(self, clip: 'VideoNode') -> 'VideoNode': ...

class _Plugin_std_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "std" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AddBorders(self, left: Optional[int] = None, right: Optional[int] = None, top: Optional[int] = None, bottom: Optional[int] = None, color: Optional[SingleAndSequence[float]] = None) -> 'VideoNode': ...
    def AssumeFPS(self, src: Optional['VideoNode'] = None, fpsnum: Optional[int] = None, fpsden: Optional[int] = None) -> 'VideoNode': ...
    def AverageFrames(self, weights: SingleAndSequence[float], scale: Optional[float] = None, scenechange: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Binarize(self, threshold: Optional[SingleAndSequence[float]] = None, v0: Optional[SingleAndSequence[float]] = None, v1: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def BinarizeMask(self, threshold: Optional[SingleAndSequence[float]] = None, v0: Optional[SingleAndSequence[float]] = None, v1: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def BlankClip(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, length: Optional[int] = None, fpsnum: Optional[int] = None, fpsden: Optional[int] = None, color: Optional[SingleAndSequence[float]] = None, keep: Optional[int] = None) -> 'VideoNode': ...
    def BoxBlur(self, planes: Optional[SingleAndSequence[int]] = None, hradius: Optional[int] = None, hpasses: Optional[int] = None, vradius: Optional[int] = None, vpasses: Optional[int] = None) -> 'VideoNode': ...
    def Cache(self, size: Optional[int] = None, fixed: Optional[int] = None, make_linear: Optional[int] = None) -> 'VideoNode': ...
    def ClipToProp(self, mclip: 'VideoNode', prop: Optional[DataType] = None) -> 'VideoNode': ...
    def Convolution(self, matrix: SingleAndSequence[float], bias: Optional[float] = None, divisor: Optional[float] = None, planes: Optional[SingleAndSequence[int]] = None, saturate: Optional[int] = None, mode: Optional[DataType] = None) -> 'VideoNode': ...
    def CopyFrameProps(self, prop_src: 'VideoNode') -> 'VideoNode': ...
    def Crop(self, left: Optional[int] = None, right: Optional[int] = None, top: Optional[int] = None, bottom: Optional[int] = None) -> 'VideoNode': ...
    def CropAbs(self, width: int, height: int, left: Optional[int] = None, top: Optional[int] = None, x: Optional[int] = None, y: Optional[int] = None) -> 'VideoNode': ...
    def CropRel(self, left: Optional[int] = None, right: Optional[int] = None, top: Optional[int] = None, bottom: Optional[int] = None) -> 'VideoNode': ...
    def Deflate(self, planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None) -> 'VideoNode': ...
    def DeleteFrames(self, frames: SingleAndSequence[int]) -> 'VideoNode': ...
    def DoubleWeave(self, tff: Optional[int] = None) -> 'VideoNode': ...
    def DuplicateFrames(self, frames: SingleAndSequence[int]) -> 'VideoNode': ...
    def Expr(self, expr: SingleAndSequence[DataType], format: Optional[int] = None) -> 'VideoNode': ...
    def FlipHorizontal(self) -> 'VideoNode': ...
    def FlipVertical(self) -> 'VideoNode': ...
    def FrameEval(self, eval: VSMapValueCallback[VideoNode], prop_src: Optional[SingleAndSequence[VideoNode]] = None, clip_src: Optional[SingleAndSequence[VideoNode]] = None) -> 'VideoNode': ...
    def FreezeFrames(self, first: SingleAndSequence[int], last: SingleAndSequence[int], replacement: SingleAndSequence[int]) -> 'VideoNode': ...
    def Inflate(self, planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None) -> 'VideoNode': ...
    def Interleave(self, extend: Optional[int] = None, mismatch: Optional[int] = None, modify_duration: Optional[int] = None) -> 'VideoNode': ...
    def Invert(self, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def InvertMask(self, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Levels(self, min_in: Optional[SingleAndSequence[float]] = None, max_in: Optional[SingleAndSequence[float]] = None, gamma: Optional[SingleAndSequence[float]] = None, min_out: Optional[SingleAndSequence[float]] = None, max_out: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Limiter(self, min: Optional[SingleAndSequence[float]] = None, max: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Loop(self, times: Optional[int] = None) -> 'VideoNode': ...
    def Lut(self, planes: Optional[SingleAndSequence[int]] = None, lut: Optional[SingleAndSequence[int]] = None, lutf: Optional[SingleAndSequence[float]] = None, function: Optional[VSMapValueCallback[Union[int, float]]] = None, bits: Optional[int] = None, floatout: Optional[int] = None) -> 'VideoNode': ...
    def Lut2(self, clipb: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, lut: Optional[SingleAndSequence[int]] = None, lutf: Optional[SingleAndSequence[float]] = None, function: Optional[VSMapValueCallback[Union[int, float]]] = None, bits: Optional[int] = None, floatout: Optional[int] = None) -> 'VideoNode': ...
    def MakeDiff(self, clipb: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def MaskedMerge(self, clipb: 'VideoNode', mask: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, first_plane: Optional[int] = None, premultiplied: Optional[int] = None) -> 'VideoNode': ...
    def Maximum(self, planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None, coordinates: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Median(self, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Merge(self, clipb: 'VideoNode', weight: Optional[SingleAndSequence[float]] = None) -> 'VideoNode': ...
    def MergeDiff(self, clipb: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Minimum(self, planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None, coordinates: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def ModifyFrame(self, clips: SingleAndSequence['VideoNode'], selector: VSMapValueCallback[VideoFrame]) -> 'VideoNode': ...
    def PEMVerifier(self, upper: Optional[SingleAndSequence[float]] = None, lower: Optional[SingleAndSequence[float]] = None) -> 'VideoNode': ...
    def PlaneStats(self, clipb: Optional['VideoNode'] = None, plane: Optional[int] = None, prop: Optional[DataType] = None) -> 'VideoNode': ...
    def PreMultiply(self, alpha: 'VideoNode') -> 'VideoNode': ...
    def Prewitt(self, planes: Optional[SingleAndSequence[int]] = None, scale: Optional[float] = None) -> 'VideoNode': ...
    def PropToClip(self, prop: Optional[DataType] = None) -> 'VideoNode': ...
    def RemoveFrameProps(self, props: Optional[SingleAndSequence[DataType]] = None) -> 'VideoNode': ...
    def Reverse(self) -> 'VideoNode': ...
    def SelectEvery(self, cycle: int, offsets: SingleAndSequence[int], modify_duration: Optional[int] = None) -> 'VideoNode': ...
    def SeparateFields(self, tff: Optional[int] = None, modify_duration: Optional[int] = None) -> 'VideoNode': ...
    def SetFieldBased(self, value: int) -> 'VideoNode': ...
    def SetFrameProp(self, prop: DataType, intval: Optional[SingleAndSequence[int]] = None, floatval: Optional[SingleAndSequence[float]] = None, data: Optional[SingleAndSequence[DataType]] = None) -> 'VideoNode': ...
    def SetFrameProps(self, **kwargs: Any) -> 'VideoNode': ...
    def SetVideoCache(self, mode: Optional[int] = None, fixedsize: Optional[int] = None, maxsize: Optional[int] = None, maxhistory: Optional[int] = None) -> None: ...
    def ShufflePlanes(self, planes: SingleAndSequence[int], colorfamily: int) -> 'VideoNode': ...
    def Sobel(self, planes: Optional[SingleAndSequence[int]] = None, scale: Optional[float] = None) -> 'VideoNode': ...
    def Splice(self, mismatch: Optional[int] = None) -> 'VideoNode': ...
    def SplitPlanes(self) -> SingleAndSequence['VideoNode']: ...
    def StackHorizontal(self) -> 'VideoNode': ...
    def StackVertical(self) -> 'VideoNode': ...
    def Transpose(self) -> 'VideoNode': ...
    def Trim(self, first: Optional[int] = None, last: Optional[int] = None, length: Optional[int] = None) -> 'VideoNode': ...
    def Turn180(self) -> 'VideoNode': ...

class _Plugin_std_AudioNode_Bound(Plugin):
    """This class implements the module definitions for the "std" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AssumeSampleRate(self, src: Optional['AudioNode'] = None, samplerate: Optional[int] = None) -> 'AudioNode': ...
    def AudioGain(self, gain: Optional[SingleAndSequence[float]] = None) -> 'AudioNode': ...
    def AudioLoop(self, times: Optional[int] = None) -> 'AudioNode': ...
    def AudioMix(self, matrix: SingleAndSequence[float], channels_out: SingleAndSequence[int]) -> 'AudioNode': ...
    def AudioReverse(self) -> 'AudioNode': ...
    def AudioSplice(self) -> 'AudioNode': ...
    def AudioTrim(self, first: Optional[int] = None, last: Optional[int] = None, length: Optional[int] = None) -> 'AudioNode': ...
    def BlankAudio(self, channels: Optional[int] = None, bits: Optional[int] = None, sampletype: Optional[int] = None, samplerate: Optional[int] = None, length: Optional[int] = None, keep: Optional[int] = None) -> 'AudioNode': ...
    def SetAudioCache(self, mode: Optional[int] = None, fixedsize: Optional[int] = None, maxsize: Optional[int] = None, maxhistory: Optional[int] = None) -> None: ...
    def ShuffleChannels(self, channels_in: SingleAndSequence[int], channels_out: SingleAndSequence[int]) -> 'AudioNode': ...
    def SplitChannels(self) -> SingleAndSequence['AudioNode']: ...

# end implementation

    
# implementation: znedi3

class _Plugin_znedi3_Core_Bound(Plugin):
    """This class implements the module definitions for the "znedi3" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def nnedi3(self, clip: 'VideoNode', field: int, dh: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None, nsize: Optional[int] = None, nns: Optional[int] = None, qual: Optional[int] = None, etype: Optional[int] = None, pscrn: Optional[int] = None, opt: Optional[int] = None, int16_prescreener: Optional[int] = None, int16_predictor: Optional[int] = None, exp: Optional[int] = None, show_mask: Optional[int] = None, x_nnedi3_weights_bin: Optional[DataType] = None, x_cpu: Optional[DataType] = None) -> 'VideoNode': ...

class _Plugin_znedi3_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "znedi3" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def nnedi3(self, field: int, dh: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None, nsize: Optional[int] = None, nns: Optional[int] = None, qual: Optional[int] = None, etype: Optional[int] = None, pscrn: Optional[int] = None, opt: Optional[int] = None, int16_prescreener: Optional[int] = None, int16_predictor: Optional[int] = None, exp: Optional[int] = None, show_mask: Optional[int] = None, x_nnedi3_weights_bin: Optional[DataType] = None, x_cpu: Optional[DataType] = None) -> 'VideoNode': ...

# end implementation



class RawNode:
    def __init__(self) -> NoReturn: ...  # type: ignore[misc]

    def get_frame(self, n: int) -> RawFrame: ...

    @overload
    def get_frame_async(self, n: int, cb: None = None) -> _Future[RawFrame]: ...

    @overload
    def get_frame_async(self, n: int, cb: Callable[[Union[RawFrame, None], Union[Exception, None]], None]) -> None: ...

    def frames(
        self, prefetch: Union[int, None] = None, backlog: Union[int, None] = None, close: bool = False
    ) -> Iterator[RawFrame]: ...

    def set_output(self, index: int = 0) -> None: ...

    def is_inspectable(self, version: Union[int, None] = None) -> bool: ...

    if not TYPE_CHECKING:
        @property
        def _node_name(self) -> str: ...

        @property
        def _name(self) -> str: ...

        @property
        def _inputs(self) -> Dict[str, _VapourSynthMapValue]: ...

        @property
        def _timings(self) -> int: ...

        @property
        def _mode(self) -> FilterMode: ...

        @property
        def _dependencies(self): ...

    @overload
    def __eq__(self: 'SelfRawNode', other: 'SelfRawNode', /) -> bool: ...  # type: ignore[misc]

    @overload
    def __eq__(self: 'SelfRawNode', other: Any, /) -> Literal[False]: ...

    def __add__(self: 'SelfRawNode', other: 'SelfRawNode', /) -> 'SelfRawNode': ...

    def __radd__(self: 'SelfRawNode', other: 'SelfRawNode', /) -> 'SelfRawNode': ...

    def __mul__(self: 'SelfRawNode', other: int) -> 'SelfRawNode': ...

    def __rmul__(self: 'SelfRawNode', other: int) -> 'SelfRawNode': ...

    def __getitem__(self: 'SelfRawNode', index: Union[int, slice], /) -> 'SelfRawNode': ...

    def __getattr__(self, name: str) -> Plugin: ...

    def __len__(self) -> int: ...


SelfRawNode = TypeVar('SelfRawNode', bound=RawNode)


class VideoNode(RawNode):
    format: Union[VideoFormat, None]

    width: int
    height: int

    fps_num: int
    fps_den: int

    fps: Fraction

    num_frames: int

    def set_output(
        self, index: int = 0, alpha: Union['VideoNode', None] = None, alt_output: Literal[0, 1, 2] = 0
    ) -> None: ...

    def output(
        self, fileobj: BinaryIO, y4m: bool = False, progress_update: object = None, prefetch: int = 0, backlog: int = -1
    ) -> None: ...

    def get_frame(self, n: int) -> VideoFrame: ...

    @overload  # type: ignore[override]
    def get_frame_async(self, n: int, cb: None = None) -> _Future[VideoFrame]: ...

    @overload
    def get_frame_async(self, n: int, cb: Callable[[Union[VideoFrame, None], Union[Exception, None]], None]) -> None: ...

    def frames(
        self, prefetch: Union[int, None] = None, backlog: Union[int, None] = None, close: bool = False
    ) -> Iterator[VideoFrame]: ...

    # instance_bound_VideoNode: bm3d
    @property
    def bm3d(self) -> _Plugin_bm3d_VideoNode_Bound:
        """Implementation of BM3D denoising filter for VapourSynth."""
    # end instance
    # instance_bound_VideoNode: bm3dcpu
    @property
    def bm3dcpu(self) -> _Plugin_bm3dcpu_VideoNode_Bound:
        """BM3D algorithm implemented in AVX and AVX2 intrinsics"""
    # end instance
    # instance_bound_VideoNode: bm3dcuda
    @property
    def bm3dcuda(self) -> _Plugin_bm3dcuda_VideoNode_Bound:
        """BM3D algorithm implemented in CUDA"""
    # end instance
    # instance_bound_VideoNode: bm3dcuda_rtc
    @property
    def bm3dcuda_rtc(self) -> _Plugin_bm3dcuda_rtc_VideoNode_Bound:
        """BM3D algorithm implemented in CUDA (NVRTC)"""
    # end instance
    # instance_bound_VideoNode: dfttest
    @property
    def dfttest(self) -> _Plugin_dfttest_VideoNode_Bound:
        """2D/3D frequency domain denoiser"""
    # end instance
    # instance_bound_VideoNode: flux
    @property
    def flux(self) -> _Plugin_flux_VideoNode_Bound:
        """FluxSmooth plugin for VapourSynth"""
    # end instance
    # instance_bound_VideoNode: knlm
    @property
    def knlm(self) -> _Plugin_knlm_VideoNode_Bound:
        """KNLMeansCL for VapourSynth"""
    # end instance
    # instance_bound_VideoNode: mv
    @property
    def mv(self) -> _Plugin_mv_VideoNode_Bound:
        """MVTools v23"""
    # end instance
    # instance_bound_VideoNode: mvsf
    @property
    def mvsf(self) -> _Plugin_mvsf_VideoNode_Bound:
        """MVTools Single Precision"""
    # end instance
    # instance_bound_VideoNode: nnedi3
    @property
    def nnedi3(self) -> _Plugin_nnedi3_VideoNode_Bound:
        """Neural network edge directed interpolation (3rd gen.), v12"""
    # end instance
    # instance_bound_VideoNode: nnedi3cl
    @property
    def nnedi3cl(self) -> _Plugin_nnedi3cl_VideoNode_Bound:
        """An intra-field only deinterlacer"""
    # end instance
    # instance_bound_VideoNode: resize
    @property
    def resize(self) -> _Plugin_resize_VideoNode_Bound:
        """VapourSynth Resize"""
    # end instance
    # instance_bound_VideoNode: retinex
    @property
    def retinex(self) -> _Plugin_retinex_VideoNode_Bound:
        """Implementation of Retinex algorithm for VapourSynth."""
    # end instance
    # instance_bound_VideoNode: std
    @property
    def std(self) -> _Plugin_std_VideoNode_Bound:
        """VapourSynth Core Functions"""
    # end instance
    # instance_bound_VideoNode: znedi3
    @property
    def znedi3(self) -> _Plugin_znedi3_VideoNode_Bound:
        """Neural network edge directed interpolation (3rd gen.)"""
    # end instance


class AudioNode(RawNode):
    sample_type: object
    bits_per_sample: int
    bytes_per_sample: int

    channel_layout: int
    num_channels: int

    sample_rate: int
    num_samples: int

    num_frames: int

    def get_frame(self, n: int) -> AudioFrame: ...

    @overload  # type: ignore[override]
    def get_frame_async(self, n: int, cb: None = None) -> _Future[AudioFrame]: ...

    @overload
    def get_frame_async(self, n: int, cb: Callable[[Union[AudioFrame, None], Union[Exception, None]], None]) -> None: ...

    def frames(
        self, prefetch: Union[int, None] = None, backlog: Union[int, None] = None, close: bool = False
    ) -> Iterator[AudioFrame]: ...

    # instance_bound_AudioNode: std
    @property
    def std(self) -> _Plugin_std_AudioNode_Bound:
        """VapourSynth Core Functions"""
    # end instance


class VSMapValueCallbackData:
    def __init__(
        self, node: RawNode, env: EnvironmentData,
        callback: Union[Callable[[Union[RawFrame, None], Union[Exception, None]], None], None] = None
    ) -> None: ...

    def receive(self, n: int, result: Union[RawFrame, Exception]) -> None: ...


class LogHandle:
    def __init__(self) -> NoReturn: ...  # type: ignore[misc]


class PythonVSScriptLoggingBridge(Handler):

    def __init__(self, parent: Handler, level: int = LogLevelUnset) -> None: ...

    def emit(self, record: LogRecord) -> None: ...


class Function:
    plugin: 'Plugin'
    name: str
    signature: str
    return_signature: str

    def __init__(self) -> NoReturn: ...  # type: ignore[misc]

    def __call__(self, *args: _VapourSynthMapValue, **kwargs: _VapourSynthMapValue) -> _VapourSynthMapValue: ...

    @property
    def __signature__(self) -> Signature: ...


class Plugin:
    identifier: str
    namespace: str
    name: str

    def __init__(self) -> NoReturn: ...  # type: ignore[misc]

    def __getattr__(self, name: str) -> Function: ...

    def functions(self) -> Iterator[Function]: ...


class Core:
    def __init__(self) -> NoReturn: ...  # type: ignore[misc]

    @property
    def num_threads(self) -> int: ...

    @num_threads.setter
    def num_threads(self) -> None: ...

    @property
    def max_cache_size(self) -> int: ...

    @max_cache_size.setter
    def max_cache_size(self) -> None: ...

    @property
    def flags(self) -> int: ...

    def plugins(self) -> Iterator[Plugin]: ...

    def query_video_format(
        self, color_family: ColorFamily, sample_type: SampleType, bits_per_sample: int, subsampling_w: int = 0,
        subsampling_h: int = 0
    ) -> VideoFormat: ...

    def get_video_format(self, id: Union[VideoFormat, int, PresetFormat]) -> VideoFormat: ...

    def create_video_frame(self, format: VideoFormat, width: int, height: int) -> VideoFrame: ...

    def log_message(self, message_type: MessageType, message: str) -> None: ...

    def add_log_handler(self, handler_func: Callable[[MessageType, str], None]) -> LogHandle: ...

    def remove_log_handler(self, handle: LogHandle) -> None: ...

    def version(self) -> str: ...

    def version_number(self) -> int: ...

    def __getattr__(self, name: str) -> Plugin: ...

    # instance_bound_Core: bm3d
    @property
    def bm3d(self) -> _Plugin_bm3d_Core_Bound:
        """Implementation of BM3D denoising filter for VapourSynth."""
    # end instance
    # instance_bound_Core: bm3dcpu
    @property
    def bm3dcpu(self) -> _Plugin_bm3dcpu_Core_Bound:
        """BM3D algorithm implemented in AVX and AVX2 intrinsics"""
    # end instance
    # instance_bound_Core: bm3dcuda
    @property
    def bm3dcuda(self) -> _Plugin_bm3dcuda_Core_Bound:
        """BM3D algorithm implemented in CUDA"""
    # end instance
    # instance_bound_Core: bm3dcuda_rtc
    @property
    def bm3dcuda_rtc(self) -> _Plugin_bm3dcuda_rtc_Core_Bound:
        """BM3D algorithm implemented in CUDA (NVRTC)"""
    # end instance
    # instance_bound_Core: dfttest
    @property
    def dfttest(self) -> _Plugin_dfttest_Core_Bound:
        """2D/3D frequency domain denoiser"""
    # end instance
    # instance_bound_Core: flux
    @property
    def flux(self) -> _Plugin_flux_Core_Bound:
        """FluxSmooth plugin for VapourSynth"""
    # end instance
    # instance_bound_Core: knlm
    @property
    def knlm(self) -> _Plugin_knlm_Core_Bound:
        """KNLMeansCL for VapourSynth"""
    # end instance
    # instance_bound_Core: mv
    @property
    def mv(self) -> _Plugin_mv_Core_Bound:
        """MVTools v23"""
    # end instance
    # instance_bound_Core: mvsf
    @property
    def mvsf(self) -> _Plugin_mvsf_Core_Bound:
        """MVTools Single Precision"""
    # end instance
    # instance_bound_Core: nnedi3
    @property
    def nnedi3(self) -> _Plugin_nnedi3_Core_Bound:
        """Neural network edge directed interpolation (3rd gen.), v12"""
    # end instance
    # instance_bound_Core: nnedi3cl
    @property
    def nnedi3cl(self) -> _Plugin_nnedi3cl_Core_Bound:
        """An intra-field only deinterlacer"""
    # end instance
    # instance_bound_Core: resize
    @property
    def resize(self) -> _Plugin_resize_Core_Bound:
        """VapourSynth Resize"""
    # end instance
    # instance_bound_Core: retinex
    @property
    def retinex(self) -> _Plugin_retinex_Core_Bound:
        """Implementation of Retinex algorithm for VapourSynth."""
    # end instance
    # instance_bound_Core: std
    @property
    def std(self) -> _Plugin_std_Core_Bound:
        """VapourSynth Core Functions"""
    # end instance
    # instance_bound_Core: znedi3
    @property
    def znedi3(self) -> _Plugin_znedi3_Core_Bound:
        """Neural network edge directed interpolation (3rd gen.)"""
    # end instance


class _CoreProxy(Core):
    @property
    def core(self) -> Core: ...


core: _CoreProxy
