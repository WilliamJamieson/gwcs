"""
Define some base-line useful type aliases for GWCS.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TypeAlias, Union

import numpy as np
from astropy.modeling import Model
from astropy.modeling.bounding_box import CompoundBoundingBox, ModelBoundingBox

__all__ = [
    "Bbox",
    "BoundingBox",
    "Bounds",
    "Cbbox",
    "Interval",
    "Mdl",
    "Real",
]

Real: TypeAlias = int | float | Fraction | np.integer | np.floating

Interval: TypeAlias = tuple[Real, Real]
Bounds: TypeAlias = tuple[Interval, ...] | None
Bbox: TypeAlias = tuple[Interval, ...] | Interval
Cbbox: TypeAlias = dict[tuple[str, ...], Bbox]

BoundingBox: TypeAlias = ModelBoundingBox | CompoundBoundingBox | None

Mdl: TypeAlias = Union[Model, None]  # noqa: UP007
