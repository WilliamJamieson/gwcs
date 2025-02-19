from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from astropy.modeling import Model

from gwcs._typing import BoundingBox, Mdl, Real

from ._typing import (
    GWCSArrays,
    GWCSLowLevelArrays,
    GWCSLowLevelValue,
    GWCSValue,
    HighLevelObject,
    HighLevelObjects,
)

if TYPE_CHECKING:
    from gwcs.coordinate_frames import BaseCoordinateFrame

__all__ = ["BaseGwcs"]


class BaseGwcs(abc.ABC):
    """
    Base class for a GWCS object.
    """

    @property
    @abc.abstractmethod
    def forward_transform(self) -> Model:
        """
        The forward transform of the WCS.
        """

    @property
    @abc.abstractmethod
    def input_frame(self) -> BaseCoordinateFrame | None:
        """
        The input coordinate frame of the WCS.
        """

    @property
    @abc.abstractmethod
    def output_frame(self) -> BaseCoordinateFrame | None:
        """
        The output coordinate frame of the WCS.
        """

    @property
    @abc.abstractmethod
    def available_frames(self) -> list[str]:
        """
        List of all the frame names in this WCS in their order in the pipeline
        """

    @property
    @abc.abstractmethod
    def bounding_box(self) -> BoundingBox | None:
        """
        The bounding box of the WCS.
        """

    @abc.abstractmethod
    def get_transform(
        self, from_frame: str | BaseCoordinateFrame, to_frame: str | BaseCoordinateFrame
    ) -> Mdl:
        """
        Return a transform between two coordinate frames.

        Parameters
        ----------
        from_frame
            Initial coordinate frame name of object.
        to_frame
            End coordinate frame name or object.

        Returns
        -------
            Transform between two frames.
        """

    @abc.abstractmethod
    def _call_forward(
        self,
        *args: GWCSLowLevelValue,
        from_frame: BaseCoordinateFrame | None = None,
        to_frame: BaseCoordinateFrame | None = None,
        with_bounding_box: bool = True,
        fill_value: Real = np.nan,
        **kwargs: Any,
    ) -> GWCSLowLevelArrays:
        """
        Executes the forward transform, but values only.

        Notes
        -----
        Can accept pure quantities as inputs and my return quantities
        as outputs depending on the nature of the transform.
        """

    @abc.abstractmethod
    def _call_backward(
        self,
        *args: GWCSLowLevelValue,
        with_bounding_box: bool = True,
        fill_value: Real = np.nan,
        **kwargs: Any,
    ) -> GWCSLowLevelArrays:
        """
        Executes the backward transform, but values only.

        Notes
        -----
        Can accept pure quantities as inputs and my return quantities
        as outputs depending on the nature of the transform.
        """

    @overload
    def __call__(
        self,
        *args: GWCSLowLevelValue,
        with_bounding_box: bool = True,
        fill_value: Real = np.nan,
        with_units: bool = False,
        **kwargs: Any,
    ) -> GWCSLowLevelArrays: ...

    @overload
    def __call__(
        self,
        *args: HighLevelObject,
        with_bounding_box: bool = True,
        fill_value: Real = np.nan,
        with_units: bool = False,
        **kwargs: Any,
    ) -> HighLevelObjects: ...

    @abc.abstractmethod
    def __call__(
        self,
        *args: GWCSValue,
        with_bounding_box: bool = True,
        fill_value: Real = np.nan,
        with_units: bool = False,
        **kwargs: Any,
    ) -> GWCSArrays:
        """
        Executes the forward transform.

        args
            Inputs in the input coordinate system, separate inputs
            for each dimension.
        with_bounding_box
            If True(default) values in the result which correspond to
            any of the inputs being outside the bounding_box are set
            to ``fill_value``.
        fill_value
            Output value for inputs outside the bounding_box
            (default is np.nan).
        with_units
            If ``True`` then high level Astropy objects will be returned.
            Optional, default=False.
        """
