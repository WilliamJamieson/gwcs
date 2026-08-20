"""Type aliases and classes for public GWCS API.

This module serves as a single source of truth for GWCS type definitions,
making them available for users and improving Sphinx documentation resolution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias, TypeVar, Union

from astropy.coordinates import BaseCoordinateFrame
from astropy.modeling import Model, projections
from astropy.modeling.bounding_box import CompoundBoundingBox, ModelBoundingBox
from astropy.time import Time
from astropy.units import Quantity, Unit
from numpy import dtype, generic, integer, ndarray, number

from .coordinate_frames import (
    AxisType,
    CoordinateFrameProtocol,
    WorldAxisObjectClass,
    WorldAxisObjectClassConverter,
)
from .wcs import Step
from .wcs._pipeline import _BasePipeline

__all__ = [
    "AstropyBuiltInFrame",
    "AxesOrder",
    "AxesType",
    "AxisNames",
    "AxisPhysicalTypes",
    "AxisTypeValue",
    "AxisTypes",
    "AxisUnits",
    "BoundingBoxInput",
    "BoundingBoxLike",
    "Degree",
    "FITSProjection",
    "ForwardTransform",
    "FrameLike",
    "HighLevelCoordinate",
    "HighLevelCoordinates",
    "LowLevelArray",
    "LowLevelArrayOutputs",
    "LowLevelArrayValue",
    "LowLevelIndexArray",
    "LowLevelIndexArrayOutputs",
    "LowLevelInput",
    "LowLevelOutputs",
    "LowLevelValue",
    "Mdl",
    "Numeric",
    "OptionalFrameLike",
    "PixelBounds",
    "PixelShape",
    "ReferencePixel",
    "Sampling",
    "StepSpec",
    "WorldAxisObjectClasses",
]

_DtypeGeneric = TypeVar("_DtypeGeneric", bound=generic)

# Coordinate-frame types
AstropyBuiltInFrame: TypeAlias = Time | BaseCoordinateFrame
FrameLike: TypeAlias = str | CoordinateFrameProtocol
OptionalFrameLike: TypeAlias = FrameLike | None
HighLevelCoordinate: TypeAlias = object
HighLevelCoordinates: TypeAlias = tuple[HighLevelCoordinate, ...] | HighLevelCoordinate
AxisTypeValue: TypeAlias = AxisType | str
AxisTypes: TypeAlias = tuple[AxisTypeValue, ...]
AxesType: TypeAlias = AxisTypes | AxisTypeValue
AxisUnits: TypeAlias = tuple[Unit | None, ...]
AxisNames: TypeAlias = tuple[str, ...]
AxesOrder: TypeAlias = tuple[int, ...]
AxisPhysicalTypes: TypeAlias = tuple[str | None, ...]
WorldAxisObjectClasses: TypeAlias = (
    dict[str, WorldAxisObjectClass]
    | dict[str, WorldAxisObjectClassConverter]
    | dict[str, WorldAxisObjectClass | WorldAxisObjectClassConverter]
)

# Low-level numerical values
LowLevelArray: TypeAlias = ndarray[tuple[int, ...], dtype[_DtypeGeneric]]
LowLevelInput: TypeAlias = LowLevelArray | Quantity
Numeric: TypeAlias = float | number
LowLevelValue: TypeAlias = LowLevelInput | Numeric
LowLevelArrayValue: TypeAlias = LowLevelArray | Numeric
LowLevelArrayOutputs: TypeAlias = tuple[LowLevelArrayValue, ...] | LowLevelArrayValue
LowLevelOutputs: TypeAlias = tuple[LowLevelInput, ...] | LowLevelInput
LowLevelIndexArray: TypeAlias = LowLevelArray[integer[Any]]
LowLevelIndexArrayOutputs: TypeAlias = (
    tuple[LowLevelIndexArray, ...] | LowLevelIndexArray
)

# FITS and approximation configuration
BoundingBoxLike: TypeAlias = ModelBoundingBox | CompoundBoundingBox
BoundingBoxInput: TypeAlias = BoundingBoxLike | LowLevelInput | Sequence[LowLevelInput]
Degree: TypeAlias = int | Sequence[int] | None
Sampling: TypeAlias = float | Sequence[float]
PixelShape: TypeAlias = tuple[int, ...] | None
PixelBounds: TypeAlias = tuple[tuple[float, float], ...] | None
ReferencePixel: TypeAlias = Sequence[float] | None
FITSProjection: TypeAlias = str | projections.Sky2PixProjection

# Models and pipeline types
Mdl: TypeAlias = Union[Model, None]  # noqa: UP007
StepSpec: TypeAlias = Step | tuple[CoordinateFrameProtocol, Mdl]
ForwardTransform: TypeAlias = Union[Model, Sequence[StepSpec] | _BasePipeline]  # noqa: UP007
