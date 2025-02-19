"""
Define type aliases specific to the GWCS WCS-API. (APE 14)

This submodule is included in the api module specifically so that the
type aliases relative to the GWCS WCS-API (APE 14) are clearly delineated
as part of the public API.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from astropy.coordinates import (
    BaseCoordinateFrame,
    SkyCoord,
    SpectralCoord,
    StokesCoord,
)
from astropy.time import Time
from astropy.units import Quantity

from gwcs._typing import Real

__all__ = [
    "AxisPhysicalType",
    "AxisPhysicalTypes",
    "GWCSArray",
    "GWCSArrays",
    "GWCSLowLevelArray",
    "GWCSLowLevelArrays",
    "GWCSLowLevelValue",
    "GWCSValue",
    "HighLevelObject",
    "HighLevelObjects",
    "LowLevelArray",
    "LowLevelArrays",
    "LowLevelValue",
]
# This is to represent a single value from a low-level function per APE 14.
LowLevelValue: TypeAlias = Real | npt.NDArray[np.number]
# This is to represent the fact that GWCS can handle units within its APE 14
# API.
GWCSLowLevelValue: TypeAlias = LowLevelValue | Quantity

# This is to represent all the values together for a single low-level function.
LowLevelArray: TypeAlias = tuple[LowLevelValue, ...]
LowLevelArrays: TypeAlias = LowLevelArray | LowLevelValue

GWCSLowLevelArray: TypeAlias = tuple[GWCSLowLevelValue, ...]
GWCSLowLevelArrays: TypeAlias = GWCSLowLevelArray | GWCSLowLevelValue

HighLevelObject: TypeAlias = Time | SkyCoord | SpectralCoord | StokesCoord | Quantity
HighLevelObjects: TypeAlias = tuple[HighLevelObject, ...] | HighLevelObject

GWCSValue: TypeAlias = HighLevelObject | LowLevelValue
GWCSArray: TypeAlias = tuple[GWCSValue, ...]
GWCSArrays: TypeAlias = GWCSArray | GWCSValue

AxisPhysicalType: TypeAlias = str | BaseCoordinateFrame
AxisPhysicalTypes: TypeAlias = tuple[str | BaseCoordinateFrame, ...]
