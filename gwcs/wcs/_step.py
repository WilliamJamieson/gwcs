from __future__ import annotations

import warnings
from copy import copy
from typing import TYPE_CHECKING, NamedTuple, Self

from astropy.modeling.core import Model

from gwcs.coordinate_frames import (
    CoordinateFrameProtocol,
    EmptyFrame,
)
from gwcs.coordinate_frames._base import _is_high_level, _LegacyCoordinateFrameProtocol

if TYPE_CHECKING:
    from gwcs.typing import Mdl

__all__ = ["IndexedStep", "Step"]


class Step:
    """
    Represents a ``step`` in the WCS pipeline.

    Parameters
    ----------
    frame
        A gwcs coordinate frame object.
    transform
        A transform from this step's frame to next step's frame.
        The transform of the last step should be `None`.
    """

    def __init__(
        self, frame: str | CoordinateFrameProtocol, transform: Mdl = None
    ) -> None:
        # Allow for a string to be passed in for the frame but be turned into a
        # frame object
        self.frame = (
            frame
            if isinstance(
                frame, (CoordinateFrameProtocol, _LegacyCoordinateFrameProtocol)
            )
            else EmptyFrame.from_transform(frame, transform)
        )
        self.transform = transform

    @property
    def frame(self) -> CoordinateFrameProtocol:
        return self._frame

    @frame.setter
    def frame(self, val: CoordinateFrameProtocol) -> None:
        if not isinstance(val, CoordinateFrameProtocol):
            if not isinstance(val, _LegacyCoordinateFrameProtocol):
                msg = '"frame" should be an instance of CoordinateFrameProtocol.'
                raise TypeError(msg)

            msg = (
                "Coordinate frames that do not implement `is_high_level` are "
                "deprecated. Please update your coordinate frame to add "
                "`is_high_level`."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            # Copy the value to avoid mutating the original object.
            val = copy(val)
            val.is_high_level = lambda *args: _is_high_level(val, *args)  # type: ignore[method-assign]

        self._frame = val

    @property
    def transform(self) -> Mdl:
        return self._transform

    @transform.setter
    def transform(self, val: Mdl) -> None:
        if val is not None and not isinstance(val, Model):
            msg = '"transform" should be an instance of astropy.modeling.Model.'
            raise TypeError(msg)
        self._transform = val

    @property
    def frame_name(self) -> str:
        return self.frame.name

    @property
    def inverse(self) -> Mdl:
        if self.transform is None:
            return None

        try:
            return self.transform.inverse
        except NotImplementedError:
            return None

    def __str__(self) -> str:
        return (
            f"{self.frame_name}\t "
            f"{getattr(self.transform, 'name', 'None') or type(self.transform).__name__}"  # noqa: E501
        )

    def __repr__(self) -> str:
        return (
            f"Step(frame={self.frame_name}, "
            f"transform={getattr(self.transform, 'name', 'None') or type(self.transform).__name__})"  # noqa: E501
        )

    def copy(self) -> Self:
        return type(self)(self.frame, self.transform)

    def __getitem__(self, ind):
        warnings.warn(
            "Indexing a WCS.pipeline step is deprecated. "
            "Use the `frame` and `transform` attributes instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if ind not in (0, 1):
            msg = "Allowed inices are 0 (frame) and 1 (transform)."
            raise IndexError(msg)
        if ind == 0:
            return self.frame
        return self.transform


class IndexedStep(NamedTuple):
    """
    Class to handle a step and its index in the pipeline.
    """

    idx: int
    step: Step
