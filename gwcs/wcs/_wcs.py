# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import annotations

import itertools
from copy import copy
from typing import Self, overload

import astropy.units as u
import numpy as np
from astropy.modeling import Model, fix_inputs
from astropy.modeling.parameters import _tofloat
from astropy.wcs.wcsapi.high_level_api import (
    high_level_objects_to_values,
    values_to_high_level_objects,
)

from gwcs.api import LowLevelArray, WCSAPIMixin
from gwcs.coordinate_frames import (
    AxisType,
    CoordinateFrame,
)
from gwcs.utils import is_high_level, to_index

from ._fits import FitsMixin
from ._inverse import NumericalInverse2D, NumericalInverseProtocol
from ._pipeline import ForwardTransform, Pipeline
from ._step import Step, StepTuple

__all__ = ["WCS"]

_ITER_INV_KWARGS = ["tolerance", "maxiter", "adaptive", "detect_divergence", "quiet"]


class WCS(Pipeline, WCSAPIMixin, FitsMixin):
    """
    Basic WCS class.

    Parameters
    ----------
    forward_transform : `~astropy.modeling.Model` or a list
        The transform between ``input_frame`` and ``output_frame``.
        A list of (frame, transform) tuples where ``frame`` is the starting frame and
        ``transform`` is the transform from this frame to the next one or
        ``output_frame``.  The last tuple is (transform, None), where None indicates
        the end of the pipeline.
    input_frame : `~gwcs.coordinate_frames.CoordinateFrame`
        A coordinates object
    output_frame : `~gwcs.coordinate_frames.CoordinateFrame`
        A coordinates object
    name : str
        a name for this WCS

    """

    @overload
    def __init__(
        self,
        forward_transform: Model,
        input_frame: CoordinateFrame,
        output_frame: CoordinateFrame,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        forward_transform: list[Step | StepTuple],
        input_frame: None = None,
        output_frame: None = None,
        name: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        forward_transform: ForwardTransform,
        input_frame: CoordinateFrame | None = None,
        output_frame: CoordinateFrame | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            forward_transform=forward_transform,
            # mypy for some reason isn't able to infer the correct overload here
            input_frame=input_frame,  # type: ignore[arg-type]
            output_frame=output_frame,  # type: ignore[arg-type]
        )

        self._name = "" if name is None else name
        self._pixel_shape = None

        # Inverse handling attributes
        self.numerical_inverse_method: NumericalInverseProtocol = NumericalInverse2D(
            self
        )

    @overload
    def evaluate(
        self,
        *args: LowLevelArray,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> tuple[LowLevelArray, ...] | LowLevelArray: ...

    # MyPy thinks that Quantity falls under the overload with LowLevelArray, but
    #   we are trying to explicitly separate the case where the input is a Quantity,
    #   vs when the input is not a Quantity.
    # This could be done with a TypeVar bound to each, but that is less informative
    #   for readers.
    # This only applies when pre-commit MyPy is running because it cannot follow
    #   the import of u.Quantity properly.
    @overload
    def evaluate(  # type: ignore[overload-cannot-match]
        self,
        *args: u.Quantity,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> tuple[u.Quantity, ...] | u.Quantity: ...

    def evaluate(
        self,
        *args: LowLevelArray | u.Quantity,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> (
        tuple[LowLevelArray, ...] | tuple[u.Quantity, ...] | LowLevelArray | u.Quantity
    ):
        # Call into variable as this is a property computed each time it is called
        transform = self.forward_transform

        input_is_quantity, transform_uses_quantity = self._units_are_present(
            args, transform
        )
        args = self._make_input_units_consistent(
            transform,
            *args,
            frame=self.input_frame,
            input_is_quantity=input_is_quantity,
            transform_uses_quantity=transform_uses_quantity,
        )

        result = transform(
            *args, with_bounding_box=with_bounding_box, fill_value=fill_value, **kwargs
        )
        if self.output_frame.naxes == 1:
            result = (result,)

        result = self._make_output_units_consistent(
            transform,
            *result,
            frame=self.output_frame,
            input_is_quantity=input_is_quantity,
            transform_uses_quantity=transform_uses_quantity,
        )

        if self.output_frame.naxes == 1:
            return result[0]
        return result

    @overload
    def __call__(
        self,
        *args: LowLevelArray,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> tuple[LowLevelArray, ...] | LowLevelArray: ...

    # MyPy thinks that Quantity falls under the overload with LowLevelArray, but
    #   we are trying to explicitly separate the case where the input is a Quantity,
    #   vs when the input is not a Quantity.
    # This could be done with a TypeVar bound to each, but that is less informative
    #   for readers.
    # This only applies when pre-commit MyPy is running because it cannot follow
    #   the import of u.Quantity properly.
    @overload
    def __call__(  # type: ignore[overload-cannot-match]
        self,
        *args: u.Quantity,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> tuple[u.Quantity, ...] | u.Quantity: ...

    def __call__(
        self,
        *args: LowLevelArray | u.Quantity,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> (
        tuple[LowLevelArray, ...] | tuple[u.Quantity, ...] | LowLevelArray | u.Quantity
    ):
        """Call the :py:meth:`evaluate` method to perform the forward transformation."""
        return self.evaluate(
            *args, with_bounding_box=with_bounding_box, fill_value=fill_value, **kwargs
        )

    def _units_are_present(self, args, transform: Model) -> tuple[bool, bool]:
        """
        Determine if the inputs to a transform are quantities and the transform
        supports units.

        Parameters
        ----------
        args : a tuple of scalars or ndarray-like objects
            Inputs to a transform.
        transform : `~astropy.modeling.Model`
            Transform to be evaluated.

        Returns
        -------
        input_is_quantity, transform_uses_quantity : bool

        """
        # Validate that the input type matches what the transform expects
        input_is_quantity = any(isinstance(a, u.Quantity) for a in args)
        transform_uses_quantity = not (transform is None or not transform.uses_quantity)
        return input_is_quantity, transform_uses_quantity

    def _make_input_units_consistent(
        self,
        transform: Model,
        *args,
        frame: CoordinateFrame,
        input_is_quantity: bool = False,
        transform_uses_quantity: bool = False,
    ):
        """
        Adds or removes units from the arguments as needed so that the transform
        can be successfully evaluated.
        """
        # Validate that the input type matches what the transform expects
        if (not input_is_quantity and not transform_uses_quantity) or (
            input_is_quantity and transform_uses_quantity
        ):
            return args
        if not input_is_quantity and (
            transform_uses_quantity or transform.parameters.size
        ):
            return frame.add_units(args)
        if not transform_uses_quantity and input_is_quantity:
            return frame.remove_units(args)
        return args

    def _make_output_units_consistent(
        self,
        transform: Model,
        *args,
        frame: CoordinateFrame,
        input_is_quantity: bool = False,
        transform_uses_quantity: bool = False,
    ):
        """
        Adds or removes units from the arguments as needed so that
        the type of the output matches the input.
        """
        if not input_is_quantity and not transform_uses_quantity:
            return args

        if input_is_quantity and transform_uses_quantity:
            # make sure the output is returned in the units of the output frame
            return frame.add_units(args)
        if not input_is_quantity and (
            transform_uses_quantity or transform.parameters.size
        ):
            return frame.remove_units(args)
        if not transform_uses_quantity and input_is_quantity:
            return frame.add_units(args)
        return args

    def in_image(
        self,
        *args: LowLevelArray | u.Quantity,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> bool | np.ndarray:
        """
        This method tests if one or more of the input world coordinates are
        contained within forward transformation's image and that it maps to
        the domain of definition of the forward transformation.
        In practical terms, this function tests
        that input world coordinate(s) can be converted to input frame and that
        it is within the forward transformation's ``bounding_box`` when
        defined.

        Parameters
        ----------
        args : float, array like, `~astropy.coordinates.SkyCoord` or
            `~astropy.units.Unit` coordinates to be inverted

        kwargs : dict
            keyword arguments to be passed either to ``backward_transform``
            (when defined) or to the iterative invert method.

        Returns
        -------
        result : bool, numpy.ndarray
            A single boolean value or an array of boolean values with `True`
            indicating that the WCS footprint contains the coordinate
            and `False` if input is outside the footprint.

        """
        coords = self.invert(
            *args, with_bounding_box=with_bounding_box, fill_value=fill_value, **kwargs
        )

        result = np.isfinite(coords)
        if self.input_frame.naxes > 1:
            result = np.all(result, axis=0)

        return result  # type: ignore[no-any-return]

    @overload
    def invert(
        self,
        *args: LowLevelArray,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> tuple[LowLevelArray, ...] | LowLevelArray: ...

    # MyPy thinks that Quantity falls under the overload with LowLevelArray, but
    #   we are trying to explicitly separate the case where the input is a Quantity,
    #   vs when the input is not a Quantity.
    # This could be done with a TypeVar bound to each, but that is less informative
    #   for readers.
    # This only applies when pre-commit MyPy is running because it cannot follow
    #   the import of u.Quantity properly.
    @overload
    def invert(  # type: ignore[overload-cannot-match]
        self,
        *args: u.Quantity,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> tuple[u.Quantity, ...] | u.Quantity: ...

    def invert(
        self,
        *args: LowLevelArray | u.Quantity,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> (
        tuple[LowLevelArray, ...] | tuple[u.Quantity, ...] | LowLevelArray | u.Quantity
    ):
        try:
            transform = self.backward_transform
        except NotImplementedError:
            transform = None

        if is_high_level(*args, low_level_wcs=self):
            message = (
                "High Level objects are not supported with the native API. "
                "Please use the `world_to_pixel` method."
            )
            raise TypeError(message)

        if with_bounding_box and self.bounding_box is not None:
            args = self.outside_footprint(args)

        input_is_quantity, transform_uses_quantity = self._units_are_present(
            args, transform
        )

        args = self._make_input_units_consistent(
            transform,
            *args,
            frame=self.output_frame,
            input_is_quantity=input_is_quantity,
            transform_uses_quantity=transform_uses_quantity,
        )
        if transform is not None:
            akwargs = {k: v for k, v in kwargs.items() if k not in _ITER_INV_KWARGS}
            result = transform(
                *args,
                with_bounding_box=with_bounding_box,
                fill_value=fill_value,
                **akwargs,
            )
        else:
            result = self.numerical_inverse(
                *args,
                with_bounding_box=with_bounding_box,
                fill_value=fill_value,
                **kwargs,
            )

        if with_bounding_box and self.bounding_box is not None:
            result = self.out_of_bounds(result, fill_value=fill_value)

        if self.input_frame.naxes == 1:
            result = (result,)
        result = self._make_output_units_consistent(
            transform,
            *result,
            frame=self.input_frame,
            input_is_quantity=input_is_quantity,
            transform_uses_quantity=transform_uses_quantity,
        )
        if self.input_frame.naxes == 1:
            return result[0]
        return result

    def outside_footprint(
        self,
        world_arrays: LowLevelArray
        | tuple[LowLevelArray, ...]
        | u.Quantity
        | tuple[u.Quantity, ...],
    ) -> tuple[LowLevelArray, ...] | tuple[u.Quantity, ...]:
        world_arrays = [copy(array) for array in world_arrays]

        axes_types = set(self.output_frame.axes_type)
        axes_phys_types = self.world_axis_physical_types
        footprint = self.footprint()
        not_numerical = False
        if is_high_level(world_arrays[0], low_level_wcs=self):
            not_numerical = True
            world_arrays = high_level_objects_to_values(
                *world_arrays, low_level_wcs=self
            )
        for axtyp in axes_types:
            ind = np.asarray(np.asarray(self.output_frame.axes_type) == axtyp)

            for idim, (coordinate, phys) in enumerate(
                zip(world_arrays, axes_phys_types, strict=False)
            ):
                coord = _tofloat(coordinate)
                if np.asarray(ind).sum() > 1:
                    axis_range = footprint[:, idim]
                else:
                    axis_range = footprint
                min_ax = axis_range.min()
                max_ax = axis_range.max()

                if (
                    axtyp == "SPATIAL"
                    and str(phys).endswith((".ra", ".lon"))
                    and (max_ax - min_ax) > 180
                ):
                    # most likely this coordinate is wrapped at 360
                    d = 0.5 * (min_ax + max_ax)
                    m = axis_range <= d
                    min_ax = axis_range[m].max()
                    max_ax = axis_range[~m].min()
                    outside = (coord > min_ax) & (coord < max_ax)
                else:
                    coord_ = self._remove_quantity_frame(
                        world_arrays, self.output_frame
                    )[idim]
                    outside = (coord_ < min_ax) | (coord_ > max_ax)
                if np.any(outside):
                    if np.isscalar(coord):
                        coord = np.nan
                    else:
                        coord[outside] = np.nan
                    world_arrays[idim] = coord
        if not_numerical:
            world_arrays = values_to_high_level_objects(
                *world_arrays, low_level_wcs=self
            )
        #  Astropy does not have proper type annotations for
        #      values_to_high_level_objects
        #  so we ignore the return-value type check here.
        return world_arrays  # type: ignore[return-value]

    def out_of_bounds(
        self,
        pixel_arrays: LowLevelArray
        | tuple[LowLevelArray, ...]
        | u.Quantity
        | tuple[u.Quantity, ...],
        fill_value: float | np.number = np.nan,
    ) -> (
        tuple[LowLevelArray, ...] | tuple[u.Quantity, ...] | LowLevelArray | u.Quantity
    ):
        if np.isscalar(pixel_arrays) or self.input_frame.naxes == 1:
            pixel_arrays = [pixel_arrays]

        pixel_arrays = list(pixel_arrays)
        bbox = self.bounding_box
        for idim, pix in enumerate(pixel_arrays):
            outside = (pix < bbox[idim][0]) | (pix > bbox[idim][1])  # type: ignore[index]
            if np.any(outside):
                if np.isscalar(pix):
                    pixel_arrays[idim] = fill_value
                else:
                    pix_ = pixel_arrays[idim].astype(float, copy=True)
                    pix_[outside] = fill_value
                    pixel_arrays[idim] = pix_
        if self.input_frame.naxes == 1:
            pixel_arrays = pixel_arrays[0]
        return pixel_arrays

    def numerical_inverse(
        self,
        *args: LowLevelArray,
        tolerance: float | np.number = 1e-5,
        maxiter: int = 30,
        adaptive: bool = True,
        detect_divergence: bool = True,
        quiet: bool = True,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> tuple[LowLevelArray, ...] | LowLevelArray:
        return self.numerical_inverse_method(
            *args,
            tolerance=tolerance,
            maxiter=maxiter,
            adaptive=adaptive,
            detect_divergence=detect_divergence,
            quiet=quiet,
            with_bounding_box=with_bounding_box,
            fill_value=fill_value,
            **kwargs,
        )

    def transform(
        self,
        from_frame: str | CoordinateFrame,
        to_frame: str | CoordinateFrame,
        *args: LowLevelArray | u.Quantity,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ) -> tuple[LowLevelArray | u.Quantity, ...] | LowLevelArray | u.Quantity:
        """
        Transform positions between two frames.

        Parameters
        ----------
        from_frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
            Initial coordinate frame.
        to_frame : str, or instance of `~gwcs.coordinate_frames.CoordinateFrame`
            Coordinate frame into which to transform.
        args : float or array-like
            Inputs in ``from_frame``, separate inputs for each dimension.
        with_bounding_box : bool, optional
            If True(default) values in the result which correspond to any of
            the inputs being outside the bounding_box are set to ``fill_value``.
        fill_value : float, optional
            Output value for inputs outside the bounding_box
            (default is np.nan).
        """
        # Pull the steps and their indices from the pipeline
        # -> this also turns the frame name strings into frame objects
        from_step = self._get_step(from_frame)
        to_step = self._get_step(to_frame)
        transform = self.get_transform(from_step.step.frame, to_step.step.frame)

        if transform is None:
            msg = f"No transformation found from {from_frame} to {to_frame}."
            raise ValueError(msg)

        # Get the frame objects from the wcs pipeline
        from_frame_obj = self.get_frame(from_frame)
        to_frame_obj = self.get_frame(to_frame)

        input_is_quantity, transform_uses_quantity = self._units_are_present(
            args, transform
        )
        args = self._make_input_units_consistent(
            transform,
            *args,
            frame=from_frame_obj,
            input_is_quantity=input_is_quantity,
            transform_uses_quantity=transform_uses_quantity,
        )

        result = transform(
            *args, with_bounding_box=with_bounding_box, fill_value=fill_value, **kwargs
        )
        if to_frame_obj is not None:
            if to_frame_obj.naxes == 1:
                result = (result,)
            result = self._make_output_units_consistent(
                transform,
                *result,
                frame=to_frame_obj,
                input_is_quantity=input_is_quantity,
                transform_uses_quantity=transform_uses_quantity,
            )
        if to_frame_obj is not None and to_frame_obj.naxes == 1:
            return result[0]
        return result

    @property
    def name(self) -> str:
        """Return the name for this WCS."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the name for the WCS."""
        self._name = value

    def __str__(self) -> str:
        from astropy.table import Table

        col1 = [step.frame for step in self._pipeline]
        col2: list[str | None] = []
        for item in self._pipeline[:-1]:
            model = item.transform
            if model is None:
                col2.append(None)
            elif model.name is not None:
                col2.append(model.name)
            else:
                col2.append(model.__class__.__name__)
        col2.append(None)
        t = Table([col1, col2], names=["From", "Transform"])
        return str(t)

    def __repr__(self) -> str:
        return (
            f"<WCS(output_frame={self.output_frame}, input_frame={self.input_frame}, "
            f"forward_transform={self.forward_transform})>"
        )

    def footprint(
        self,
        bounding_box: tuple[tuple[float, float], ...]
        | tuple[float, float]
        | None = None,
        center: bool = False,
        axis_type: AxisType | str | None = None,
    ) -> LowLevelArray:
        """
        Return the footprint in world coordinates.

        Parameters
        ----------
        bounding_box : tuple of floats: (start, stop)
            ``prop: bounding_box``
        center : bool
            If `True` use the center of the pixel, otherwise use the corner.
        axis_type : AxisType
            A supported ``output_frame.axes_type`` or ``"all"`` (default).
            One of [``'spatial'``, ``'spectral'``, ``'temporal'``] or a custom type.

        Returns
        -------
        coord : ndarray
            Array of coordinates in the output_frame mapping
            corners to the output frame. For spatial coordinates the order
            is clockwise, starting from the bottom left corner.

        """
        axis_type = AxisType.from_input("all" if axis_type is None else axis_type)

        def _order_clockwise(v):
            return np.asarray(
                [
                    [v[0][0], v[1][0]],
                    [v[0][0], v[1][1]],
                    [v[0][1], v[1][1]],
                    [v[0][1], v[1][0]],
                ]
            ).T

        if bounding_box is None:
            if self.bounding_box is None:
                msg = "Need a valid bounding_box to compute the footprint."
                raise TypeError(msg)
            bb = self.bounding_box.bounding_box(order="F")
        else:
            bb = bounding_box

        if self.output_frame is None:
            msg = "Footprint requires a defined output_frame."
            raise ValueError(msg)

        all_spatial = all(t.lower() == "spatial" for t in self.output_frame.axes_type)
        if self.output_frame.naxes == 1:
            if isinstance(bb[0], u.Quantity):
                bb = np.asarray([b.value for b in bb]) * bb[0].unit
            vertices = (bb,)
        elif all_spatial:
            vertices = _order_clockwise([self.input_frame.remove_units(b) for b in bb])
        else:
            vertices = np.array(list(itertools.product(*bb))).T  # type: ignore[assignment]

        # workaround an issue with bbox with quantity, interval needs to be a cquantity,
        # not a list of quantities strip units
        if center:
            vertices = to_index(vertices)

        result = np.asarray(self.__call__(*vertices, with_bounding_box=False))

        if axis_type is AxisType.SPATIAL and all_spatial:
            return result.T

        if axis_type != "all":
            axtyp_ind = (
                np.array([AxisType.from_input(t) for t in self.output_frame.axes_type])
                == axis_type
            )
            if not axtyp_ind.any():
                msg = f'This WCS does not have axis of type "{axis_type}".'
                raise ValueError(msg)
            if len(axtyp_ind) > 1:
                result = np.asarray([(r.min(), r.max()) for r in result[axtyp_ind]])

            if axis_type is AxisType.SPATIAL:
                result = _order_clockwise(result)
            else:
                result.sort()
                result = np.squeeze(result)

        if self.output_frame.naxes == 1:
            return np.array([result]).T

        return result.T

    def fix_inputs(
        self, fixed: dict[str | int, LowLevelArray | u.Quantity | float | np.number]
    ) -> Self:
        """
        Return a new unique WCS by fixing inputs to constant values.

        Parameters
        ----------
        fixed : dict
            Keyword arguments with fixed values corresponding to ``self.selector``.

        Returns
        -------
        new_wcs : `WCS`
            A new unique WCS corresponding to the values in ``fixed``.

        Examples
        --------
        >>> w = WCS(pipeline, selector={"spectral_order": [1, 2]}) # doctest: +SKIP
        >>> new_wcs = w.set_inputs(spectral_order=2) # doctest: +SKIP
        >>> new_wcs.inputs # doctest: +SKIP
            ("x", "y")

        """
        return type(self)(
            [
                (self.pipeline[0].frame, fix_inputs(self.pipeline[0].transform, fixed)),
                *self.pipeline[1:],
            ]
        )
