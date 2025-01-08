from __future__ import annotations

from typing import TypeAlias, Union

import numpy as np
from astropy.coordinates import (
    BaseCoordinateFrame,
    SkyCoord,
    SpectralCoord,
    StokesCoord,
)
from astropy.modeling import Model
from astropy.modeling.bounding_box import CompoundBoundingBox, ModelBoundingBox
from astropy.time import Time
from astropy.units import Quantity

try:
    from astropy.units.typing import Real as _Real
except ImportError:
    from fractions import Fraction

    _Real: TypeAlias = int | float | Fraction | np.integer | np.floating

__all__ = [
    "AxisPhysicalTypes",
    "BoundingBox",
    "BoundingBoxTuple",
    "Bounds",
    "HighLevelObject",
    "HighLevelObjects",
    "Interval",
    "LowLevelArrays",
    "LowLevelUnitArrays",
    "LowLevelUnitValue",
    "LowLevelValue",
    "LowOrHigh",
    "Mdl",
    "OutputLowLevelArray",
    "Real",
    "WorldAxisClass",
    "WorldAxisComponent",
    "WorldAxisComponents",
]

# Avoid pylance warning
Real = _Real

Interval: TypeAlias = tuple[Real, Real]
Bounds: TypeAlias = tuple[Interval, ...] | None

BoundingBox: TypeAlias = ModelBoundingBox | CompoundBoundingBox | None
BoundingBoxTuple: TypeAlias = tuple[Real | Interval, ...] | None

# This is to represent a single  value from a low-level function.
LowLevelValue: TypeAlias = Real | np.ndarray
# Handle when units are a possibility. Not all functions allow units in/out
LowLevelUnitValue: TypeAlias = LowLevelValue | Quantity

# This is to represent all the values together for a single low-level function.
LowLevelArrays: TypeAlias = tuple[LowLevelValue, ...]
LowLevelUnitArrays: TypeAlias = tuple[LowLevelUnitValue, ...]

# This is to represent a general array output from a low-level function.
# Due to the fact 1D outputs are returned as a single value, rather than a tuple.
OutputLowLevelArray: TypeAlias = LowLevelValue | LowLevelArrays

HighLevelObject: TypeAlias = Time | SkyCoord | SpectralCoord | StokesCoord | Quantity
HighLevelObjects: TypeAlias = tuple[HighLevelObject, ...]

LowOrHigh: TypeAlias = tuple[LowLevelValue | HighLevelObject, ...]

WorldAxisComponent: TypeAlias = tuple[str, str | int, str]
WorldAxisClass: TypeAlias = tuple[
    type | str, tuple[int | None, ...], dict[str, HighLevelObject]
]

WorldAxisComponents: TypeAlias = list[WorldAxisComponent]
WorldAxisClasses: TypeAlias = dict[str, WorldAxisClass]

Mdl: TypeAlias = Union[Model, None]  # noqa: UP007

AxisPhysicalTypes: TypeAlias = tuple[str | BaseCoordinateFrame, ...]
