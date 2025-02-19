from __future__ import annotations

import warnings
from typing import NamedTuple, TypeAlias, cast

from astropy.modeling.core import Model

from gwcs._typing import Mdl
from gwcs.coordinate_frames import BaseCoordinateFrame, EmptyFrame

__all__ = [
    "IndexedStep",
    "Step",
    "StepTuple",
]


StepTuple: TypeAlias = tuple[BaseCoordinateFrame, Mdl]


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
        self, frame: str | BaseCoordinateFrame | None, transform: Mdl = None
    ) -> None:
        # Allow for a string to be passed in for the frame but be turned into a
        # frame object
        self._frame = self._process_frame(frame)
        self.transform = transform

    @staticmethod
    def _process_frame(frame: str | BaseCoordinateFrame | None) -> BaseCoordinateFrame:
        return (
            frame if isinstance(frame, BaseCoordinateFrame) else EmptyFrame(name=frame)
        )

    @property
    def frame(self) -> BaseCoordinateFrame:
        return self._frame

    @frame.setter
    def frame(self, frame: str | BaseCoordinateFrame) -> None:
        if not isinstance(frame, str | BaseCoordinateFrame):
            # This is a safety check, but if the hint is followed it will never
            # be reached
            msg = '"frame" should be an instance of CoordinateFrame or a string.'  # type: ignore[unreachable]
            raise TypeError(msg)

        self._frame = self._process_frame(frame)

    @property
    def transform(self) -> Mdl:
        return self._transform

    @transform.setter
    def transform(self, val: Mdl) -> None:
        if val is not None and not isinstance(val, Model):
            # This is a safety check, but if the hint is followed it will never
            # be reached
            msg = '"transform" should be an instance of astropy.modeling.Model.'  # type: ignore[unreachable]
            raise TypeError(msg)
        self._transform = val

    @property
    def inverse(self) -> Mdl:
        if self.transform is None:
            return None
        return cast(Model, self.transform.inverse)

    @property
    def frame_name(self) -> str:
        return self.frame.name

    def __getitem__(self, index: int) -> BaseCoordinateFrame | Mdl:
        warnings.warn(
            "Indexing a WCS.pipeline step is deprecated. "
            "Use the `frame` and `transform` attributes instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if index not in (0, 1):
            msg = "Allowed inices are 0 (frame) and 1 (transform)."
            raise IndexError(msg)
        if index == 0:
            return self.frame
        return self.transform

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

    def copy(self) -> Step:
        return Step(self.frame, self.transform)


class IndexedStep(NamedTuple):
    """
    Class to handle a step and its index in the pipeline.
    """

    idx: int
    step: Step
