from astropy import units as u

from gwcs._typing import AxisPhysicalTypes

from ._axis import AxesType, AxisType
from ._coordinate_frame import CoordinateFrame

__all__ = ["Frame2D"]


class Frame2D(CoordinateFrame):
    """
    A 2D coordinate frame.

    Parameters
    ----------
    axes_order
        A dimension in the input data that corresponds to this axis.
    unit
        Unit for each axis.
    axes_names
        Names of the axes in this frame.
    name
        Name of this frame.
    axes_type
        One of ["SPATIAL", "SPECTRAL", "TIME"]
    axis_physical_types
        The physical types of the axes in this frame.
    """

    def __init__(
        self,
        axes_order: tuple[int, ...] = (0, 1),
        unit: tuple[u.Unit, ...] = (u.pix, u.pix),
        axes_names: tuple[str, ...] = ("x", "y"),
        name: str | None = None,
        axes_type: AxesType | None = None,
        axis_physical_types: AxisPhysicalTypes | None = None,
    ) -> None:
        if axes_type is None:
            axes_type = (AxisType.SPATIAL, AxisType.SPATIAL)
        pht = axis_physical_types or self._default_axis_physical_types(
            axes_names, axes_type
        )

        super().__init__(
            naxes=2,
            axes_type=axes_type,
            axes_order=axes_order,
            name=name,
            axes_names=axes_names,
            unit=unit,
            axis_physical_types=pht,
        )

    def _default_axis_physical_types(
        self, axes_names: tuple[str, ...], axes_type: AxesType
    ) -> AxisPhysicalTypes:
        if axes_names is not None and all(axes_names):
            ph_type = axes_names
        else:
            ph_type = axes_type

        return tuple(f"custom:{t}" for t in ph_type)
