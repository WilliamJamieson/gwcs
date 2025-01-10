import logging

from astropy import coordinates as coord
from astropy import units as u
from astropy.utils.misc import isiterable

from gwcs._typing import AxisPhysicalTypes, WorldAxisClasses, WorldAxisComponents

from ._axis import AxisType
from ._coordinate_frame import CoordinateFrame

__all__ = ["SpectralFrame"]


class SpectralFrame(CoordinateFrame):
    """
    Represents Spectral Frame

    Parameters
    ----------
    axes_order
        A dimension in the input data that corresponds to this axis.
    reference_frame
        Reference frame (usually used with output_frame to convert to world
        coordinate objects).
    unit
        Spectral unit.
    axes_names
        Spectral axis name.
    name
        Name for this frame.
    axis_physical_types
        The physical types of the axes in this frame.
    """

    def __init__(
        self,
        axes_order: tuple[int, ...] = (0,),
        reference_frame: coord.BaseCoordinateFrame | None = None,
        unit: list[u.Unit | str] | None = None,
        axes_names: list[str] | None = None,
        name: str | None = None,
        axis_physical_types: AxisPhysicalTypes | None = None,
    ) -> None:
        if not isiterable(unit):
            unit = (unit,)
        unit = [u.Unit(un) for un in unit]
        pht = axis_physical_types or self._default_axis_physical_types(unit)

        super().__init__(
            naxes=1,
            axes_type=AxisType.SPECTRAL,
            axes_order=axes_order,
            axes_names=axes_names,
            reference_frame=reference_frame,
            unit=unit,
            name=name,
            axis_physical_types=pht,
        )

    def _default_axis_physical_types(self, unit: u.Unit) -> AxisPhysicalTypes:
        if unit[0].physical_type == "frequency":
            return ("em.freq",)
        if unit[0].physical_type == "length":
            return ("em.wl",)
        if unit[0].physical_type == "energy":
            return ("em.energy",)
        if unit[0].physical_type == "speed":
            return ("spect.dopplerVeloc",)
            logging.warning(
                "Physical type may be ambiguous. Consider "
                "setting the physical type explicitly as "
                "either 'spect.dopplerVeloc.optical' or "
                "'spect.dopplerVeloc.radio'."
            )
        return (f"custom:{unit[0].physical_type}",)

    @property
    def world_axis_object_classes(self) -> WorldAxisClasses:
        """
        Object classes for this frame.
        """
        return {"spectral": (coord.SpectralCoord, (), {"unit": self.unit[0]})}

    @property
    def _native_world_axis_object_components(self) -> WorldAxisComponents:
        return [("spectral", 0, lambda sc: sc.to_value(self.unit[0]))]
