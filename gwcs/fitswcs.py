# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Models for generating FITS WCS standard transforms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from astropy.modeling import projections
from astropy.modeling.core import Model
from astropy.modeling.models import (
    AffineTransformation2D,
    RotateNative2Celestial,
    Scale,
    Shift,
)
from astropy.modeling.parameters import Parameter

from .utils import _compute_lon_pole

if TYPE_CHECKING:
    from .typing import LowLevelArray, LowLevelArrayOutputs, LowLevelArrayValue

__all__ = [
    "FITSImagingWCSTransform",
]


class FITSImagingWCSTransform(Model):
    """
    FITS WCS standard imaging transform.

    Parameters
    ----------
    projection : `~astropy.modeling.projections.Projection`
      Celestial projection.
    crpix : tuple or iterable
      Pixel coordinate of the reference point. Default is [0, 0]
    crval : tuple or iterable
      Celestial longitude and latitude of the fiducial point in deg. Default is [0, 0]
    cdelt : tuple or iterable
      Coordinate scale factors. Default is [1, 1]
    pc : ndarray
      Linear transformation matrix of shape ``(2, 2)``. Default is [[1,0][0, 1]]


    Returns
    -------
    transform : `~astropy.modeling.core.CompoundModel`
      The transform between a detector and a celestial coordinate frames.

    Examples
    --------
    >>> from gwcs import FITSImagingWCSTransform
    >>> from astropy.modeling.models import Pix2Sky_Gnomonic
    >>> tan = Pix2Sky_Gnomonic()
    >>> fwcs = FITSImagingWCSTransform(tan, crpix=[10,10], crval=[5.36, -72.5])
    >>> print(fwcs)
    Model: FITSImagingWCSTransform
    Inputs: ('x', 'y')
    Outputs: ('lon', 'lat')
    Model set size: 1
    Parameters:
       crpix         crval       cdelt        pc
    ------------ ------------- ---------- ----------
    10.0 .. 10.0 5.36 .. -72.5 1.0 .. 1.0 1.0 .. 1.0

    """

    _separable = False

    standard_broadcasting = False
    fittable = False
    input_units = None

    n_inputs = 2
    n_outputs = 2

    crpix = Parameter(default=[0.0, 0.0], description="crpix")
    crval = Parameter(default=[0.0, 0.0], description="crval")
    cdelt = Parameter(default=[1.0, 1.0], description="cdelt")
    pc = Parameter(default=[[1.0, 0.0], [0.0, 1.0]], description="pc")

    def __init__(
        self,
        projection: projections.Projection,
        crpix: Parameter | Sequence[float] = crpix,
        crval: Parameter | Sequence[float] = crval,
        cdelt: Parameter | Sequence[float] = cdelt,
        pc: Parameter | LowLevelArray | Sequence[Sequence[float]] = pc,
        **kwargs: Any,
    ) -> None:
        super().__init__(crpix=crpix, crval=crval, cdelt=cdelt, pc=pc, **kwargs)
        self.projection = projection
        self.inputs = ("x", "y")
        self.outputs = ("lon", "lat")
        self.lon_pole = _compute_lon_pole(self.crval, projection)

        self.forward = (
            Shift(-self.crpix[0]) & Shift(-self.crpix[1])
            | AffineTransformation2D(matrix=self.pc)
            | Scale(self.cdelt[0]) & Scale(self.cdelt[1])
            | self.projection
            | RotateNative2Celestial(self.crval[0], self.crval[1], self.lon_pole)
        )

    def evaluate(  # noqa: PLR0917
        self,
        x: LowLevelArrayValue,
        y: LowLevelArrayValue,
        crpix: LowLevelArray,
        crval: LowLevelArray,
        cdelt: LowLevelArray,
        pc: LowLevelArray,
    ) -> LowLevelArrayOutputs:
        return cast("LowLevelArrayOutputs", self.forward(x, y))

    def inverse(self) -> Model:
        return cast(Model, self.forward.inverse)
