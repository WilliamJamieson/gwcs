# Licensed under a 3-clause BSD style license - see LICENSE.rst

import abc

import numpy as np
from astropy import units as u
from astropy.coordinates import BaseCoordinateFrame as _BaseCoordinateFrame

from gwcs._typing import AxisPhysicalTypes, WorldAxisClasses, WorldAxisComponents

from ._axis import AxesType
from ._frame_properties import FrameProperties


class BaseCoordinateFrame(abc.ABC):
    """
    API Definition for a Coordinate frame
    """

    _prop: FrameProperties
    """
    The FrameProperties object holding properties in native frame order.
    """

    @property
    @abc.abstractmethod
    def naxes(self) -> int:
        """
        The number of axes described by this frame.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        The name of the coordinate frame.
        """

    @property
    @abc.abstractmethod
    def unit(self) -> tuple[u.Unit, ...]:
        """
        The units of the axes in this frame.
        """

    @property
    @abc.abstractmethod
    def axes_names(self) -> tuple[str, ...]:
        """
        Names describing the axes of the frame.
        """

    @property
    @abc.abstractmethod
    def axes_order(self) -> tuple[int, ...]:
        """
        The position of the axes in the frame in the transform.
        """

    @property
    @abc.abstractmethod
    def reference_frame(self) -> _BaseCoordinateFrame:
        """
        The reference frame of the coordinates described by this frame.

        This is usually an Astropy object such as ``SkyCoord`` or ``Time``.
        """

    @property
    @abc.abstractmethod
    def axes_type(self) -> AxesType:
        """
        An upcase string describing the type of the axis.
        """

    @property
    @abc.abstractmethod
    def axis_physical_types(self) -> AxisPhysicalTypes:
        """
        The UCD 1+ physical types for the axes, in frame order.
        """

    @property
    @abc.abstractmethod
    def world_axis_object_classes(self) -> WorldAxisClasses:
        """
        The APE 14 object classes for this frame.

        See Also
        --------
        astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_object_classes
        """

    @property
    def world_axis_object_components(self) -> WorldAxisComponents:
        """
        The APE 14 object components for this frame.

        See Also
        --------
        astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_object_components
        """
        if self.naxes == 1:
            return self._native_world_axis_object_components

        # If we have more than one axis then we should sort the native
        # components by the axes_order.
        ordered = np.array(self._native_world_axis_object_components, dtype=object)[
            np.argsort(self.axes_order)
        ]
        return list(map(tuple, ordered))

    @property
    @abc.abstractmethod
    def _native_world_axis_object_components(self) -> WorldAxisComponents:
        """
        This property holds the "native" frame order of the components.

        The native order of the components is the order the frame assumes the
        axes are in when creating the high level objects, for example
        ``CelestialFrame`` creates ``SkyCoord`` objects which are in lon, lat
        order (in their positional args).

        This property is used both to construct the ordered
        ``world_axis_object_components`` property as well as by `CompositeFrame`
        to be able to get the components in their native order.
        """
