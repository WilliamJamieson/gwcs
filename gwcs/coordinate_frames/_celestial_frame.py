from astropy import coordinates as coord
from astropy import units as u

from gwcs._typing import AxisPhysicalTypes, WorldAxisClasses, WorldAxisComponents

from ._axis import AxisType
from ._coordinate_frame import CoordinateFrame

__all__ = ["CelestialFrame"]

STANDARD_REFERENCE_FRAMES = [frame.upper() for frame in coord.builtin_frames.__all__]


class CelestialFrame(CoordinateFrame):
    """
    Representation of a Celesital coordinate system.

    This class has a native order of longitude then latitude, meaning
    ``axes_names``, ``unit`` and ``axis_physical_types`` should be lon, lat
    ordered. If your transform is in a different order this should be specified
    with ``axes_order``.

    Parameters
    ----------
    axes_order
        A dimension in the input data that corresponds to this axis.
    reference_frame
        A reference frame.
    unit
        Units on axes.
    axes_name
        Names of the axes in this frame.
    name
        Name of this frame.
    axis_physical_types
        The UCD 1+ physical types for the axes, in frame order (lon, lat).
    """

    def __init__(
        self,
        axes_order: tuple[int, ...] | None = None,
        reference_frame: coord.BaseCoordinateFrame = None,
        unit: list[u.Unit] | None = None,
        axes_names: list[str] | None = None,
        name: str | None = None,
        axis_physical_types: AxisPhysicalTypes | None = None,
    ) -> None:
        naxes = 2
        if (
            reference_frame is not None
            and not isinstance(reference_frame, str)
            and reference_frame.name.upper() in STANDARD_REFERENCE_FRAMES
        ):
            _axes_names = list(reference_frame.representation_component_names.values())
            if "distance" in _axes_names:
                _axes_names.remove("distance")
            if axes_names is None:
                axes_names = _axes_names
            naxes = len(_axes_names)

        self.native_axes_order = tuple(range(naxes))
        if axes_order is None:
            axes_order = self.native_axes_order
        if unit is None:
            unit = tuple([u.degree] * naxes)
        axes_type = (AxisType.SPATIAL,) * naxes

        pht = axis_physical_types or self._default_axis_physical_types(
            reference_frame, axes_names
        )
        super().__init__(
            naxes=naxes,
            axes_type=axes_type,
            axes_order=axes_order,
            reference_frame=reference_frame,
            unit=unit,
            axes_names=axes_names,
            name=name,
            axis_physical_types=pht,
        )

    def _default_axis_physical_types(
        self, reference_frame: coord.BaseCoordinateFrame, axes_names: list[str]
    ) -> AxisPhysicalTypes:
        if isinstance(reference_frame, coord.Galactic):
            return "pos.galactic.lon", "pos.galactic.lat"
        if isinstance(
            reference_frame,
            coord.GeocentricTrueEcliptic | coord.GCRS | coord.PrecessedGeocentric,
        ):
            return "pos.bodyrc.lon", "pos.bodyrc.lat"
        if isinstance(reference_frame, coord.builtin_frames.BaseRADecFrame):
            return "pos.eq.ra", "pos.eq.dec"
        if isinstance(reference_frame, coord.builtin_frames.BaseEclipticFrame):
            return "pos.ecliptic.lon", "pos.ecliptic.lat"
        return tuple(f"custom:{t}" for t in axes_names)

    @property
    def world_axis_object_classes(self) -> WorldAxisClasses:
        """
        Object classes for this frame.
        """
        return {
            "celestial": (
                coord.SkyCoord,
                (),
                {"frame": self.reference_frame, "unit": self._prop.unit},
            )
        }

    @property
    def _native_world_axis_object_components(self) -> WorldAxisComponents:
        return [
            ("celestial", 0, lambda sc: sc.spherical.lon.to_value(self._prop.unit[0])),
            ("celestial", 1, lambda sc: sc.spherical.lat.to_value(self._prop.unit[1])),
        ]
