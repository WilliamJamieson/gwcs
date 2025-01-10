from ._base_coordinate_frame import BaseCoordinateFrame


class EmptyFrame(BaseCoordinateFrame):
    """
    Represents a "default" detector frame. This is for use as the default value
    for input frame by the WCS object.

    Parameters
    ----------
    name
        Name for this frame.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = "detector" if name is None else name

    def __repr__(self) -> str:
        return f'<{type(self).__name__}(name="{self.name}")>'

    def __str__(self) -> str:
        if self._name is not None:
            return self._name
        return type(self).__name__

    @property
    def name(self) -> str:
        """A custom name of this frame."""
        return self._name

    @name.setter
    def name(self, val: str) -> None:
        """A custom name of this frame."""
        self._name = val

    def _raise_error(self) -> None:
        msg = "EmptyFrame does not have any information"
        raise NotImplementedError(msg)

    @property
    def naxes(self):
        self._raise_error()

    @property
    def unit(self):
        self._raise_error()

    @property
    def axes_names(self):
        self._raise_error()

    @property
    def axes_order(self):
        self._raise_error()

    @property
    def reference_frame(self):
        self._raise_error()

    @property
    def axes_type(self):
        self._raise_error()

    @property
    def axis_physical_types(self):
        self._raise_error()

    @property
    def world_axis_object_classes(self):
        self._raise_error()

    @property
    def _native_world_axis_object_components(self):
        self._raise_error()
