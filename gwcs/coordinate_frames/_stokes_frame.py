from astropy import units as u
from astropy.coordinates import StokesCoord

from gwcs._typing import AxisPhysicalTypes, WorldAxisClasses, WorldAxisComponents

from ._axis import AxisType
from ._coordinate_frame import CoordinateFrame

__all__ = ["StokesFrame"]


class StokesFrame(CoordinateFrame):
    """
    A coordinate frame for representing Stokes polarisation states.

    Parameters
    ----------
    axes_order
        A dimension in the input data that corresponds to this axis.
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
        axes_names: tuple[str, ...] = ("stokes",),
        name: str | None = None,
        axis_physical_types: AxisPhysicalTypes | None = None,
    ) -> None:
        pht = axis_physical_types or self._default_axis_physical_types()

        super().__init__(
            1,
            (AxisType.STOKES,),
            axes_order,
            name=name,
            axes_names=axes_names,
            unit=u.one,
            axis_physical_types=pht,
        )

    def _default_axis_physical_types(self) -> AxisPhysicalTypes:
        return ("phys.polarization.stokes",)

    @property
    def world_axis_object_classes(self) -> WorldAxisClasses:
        """
        Object classes for this frame.
        """
        return {
            "stokes": (
                StokesCoord,
                (),
                {},
            )
        }

    @property
    def _native_world_axis_object_components(self) -> WorldAxisComponents:
        return [("stokes", 0, "value")]
