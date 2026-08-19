"""Type aliases and classes for public GWCS API.

This module serves as a single source of truth for GWCS type definitions,
making them available for users and improving Sphinx documentation resolution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, TypeVar, Union

from astropy.coordinates import BaseCoordinateFrame
from astropy.modeling import Model
from astropy.modeling.bounding_box import CompoundBoundingBox, ModelBoundingBox
from astropy.time import Time
from astropy.units import Quantity
from numpy import dtype, generic, ndarray, number

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
    "AxesType",
    "BoundingBoxInput",
    "Degree",
    "ForwardTransform",
    "FrameLike",
    "LowLevelArray",
    "LowLevelArrayValue",
    "LowLevelInput",
    "LowLevelOutputs",
    "LowLevelValue",
    "Mdl",
    "Numeric",
    "OptionalFrameLike",
    "Sampling",
    "StepSpec",
    "WorldAxisObjectClasses",
]

_DtypeGeneric = TypeVar("_DtypeGeneric", bound=generic)

# Coordinate-frame types
AstropyBuiltInFrame: TypeAlias = Time | BaseCoordinateFrame
FrameLike: TypeAlias = str | CoordinateFrameProtocol
OptionalFrameLike: TypeAlias = FrameLike | None
AxesType: TypeAlias = tuple[AxisType | str, ...] | AxisType | str
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
LowLevelOutputs: TypeAlias = tuple[LowLevelInput, ...] | LowLevelInput

# FITS and approximation configuration
BoundingBoxInput: TypeAlias = (
    ModelBoundingBox | CompoundBoundingBox | LowLevelInput | Sequence[LowLevelInput]
)
Degree: TypeAlias = int | Sequence[int] | None
Sampling: TypeAlias = float | Sequence[float]

# Models and pipeline types
Mdl: TypeAlias = Union[Model, None]  # noqa: UP007
StepSpec: TypeAlias = Step | tuple[CoordinateFrameProtocol, Mdl]
ForwardTransform: TypeAlias = Union[Model, Sequence[StepSpec] | _BasePipeline]  # noqa: UP007
