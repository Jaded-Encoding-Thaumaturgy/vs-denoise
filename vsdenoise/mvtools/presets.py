from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from _collections_abc import dict_items, dict_keys, dict_values
from vstools import CustomEnum, FieldBasedT, KwargsNotNone, KwargsT, inject_self, vs

from ..prefilters import PelType, Prefilter
from .enums import MotionMode, SADMode, SearchMode
from .motion import MotionVectors

__all__ = [
    'MVToolsPreset', 'MVToolsPresets'
]

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
    source_type: property | FieldBasedT | None = None
    high_precision: property | bool = False
    hpad: property | int | None = None
    vpad: property | int | None = None
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
    vectors: property | MotionVectors | None = None
    super_args: property | KwargsT | None = None
    analyze_args: property | KwargsT | None = None
    recalculate_args: property | KwargsT | None = None

    if TYPE_CHECKING:
        def __call__(
            self, *, tr: int | None = None, refine: int | None = None, pel: int | None = None,
            planes: int | Sequence[int] | None = None, source_type: FieldBasedT | None = None,
            high_precision: bool = False, hpad: int | None = None,
            vpad: int | None = None, block_size: int | None = None,
            overlap: int | None = None, thSAD: int | None = None, range_conversion: float | None = None,
            search: SearchMode | SearchMode.Config | None = None, motion: MotionMode.Config | None = None,
            sad_mode: SADMode | tuple[SADMode, SADMode] | None = None, rfilter: int | None = None,
            sharp: int | None = None, prefilter: Prefilter | vs.VideoNode | None = None,
            pel_type: PelType | tuple[PelType, PelType] | None = None, vectors: MotionVectors | None = None,
            super_args: KwargsT | None = None, analyze_args: KwargsT | None = None,
            recalculate_args: KwargsT | None = None
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
        pel=2, prefilter=Prefilter.NONE, sharp=2, rfilter=4,
        block_size=8, overlap=2, thSAD=300, sad_mode=SADMode.SPATIAL.same_recalc,
        motion=MotionMode.VECT_COHERENCE, search=SearchMode.HEXAGON.defaults,
        hpad=property(fget=lambda x: x.block_size), vpad=property(fget=lambda x: x.block_size),
        range_conversion=1.0
    )
    """SMDegrain by Caroliano & DogWay"""

    CMDE = MVToolsPreset(
        pel=1, prefilter=Prefilter.NONE, sharp=2, rfilter=4,
        block_size=32, overlap=16, thSAD=200, sad_mode=SADMode.SPATIAL.same_recalc,
        motion=MotionMode.HIGH_SAD, search=SearchMode.HEXAGON.defaults,
        hpad=property(fget=lambda x: x.block_size), vpad=property(fget=lambda x: x.block_size),
        range_conversion=1.0
    )
    """CMDegrain from EoE."""

    FAST = MVToolsPreset(
        pel=1, prefilter=Prefilter.MINBLUR3, thSAD=60, block_size=32,
        overlap=property(fget=lambda x: x.block_size // 2),
        sad_mode=SADMode.SPATIAL.same_recalc, search=SearchMode.DIAMOND,
        motion=MotionMode.HIGH_SAD, pel_type=PelType.BICUBIC, rfilter=2, sharp=2
    )
    """Fast preset"""

    NOISY = MVToolsPreset(
        pel=2, thSAD=100, block_size=16, overlap=property(fget=lambda x: x.block_size // 2),
        motion=MotionMode.HIGH_SAD, prefilter=Prefilter.DFTTEST,
        sad_mode=(SADMode.ADAPTIVE_SPATIAL_MIXED, SADMode.ADAPTIVE_SATD_MIXED)
    )
    """Preset for accurate estimation"""
