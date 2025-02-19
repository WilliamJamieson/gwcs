# Licensed under a 3-clause BSD style license - see LICENSE.rst
import itertools

import astropy.units as u
import numpy as np
from astropy.modeling import fix_inputs
from astropy.modeling.parameters import _tofloat
from astropy.wcs.wcsapi.high_level_api import (
    high_level_objects_to_values,
    values_to_high_level_objects,
)

from gwcs.api import GWCSAPIMixin
from gwcs.coordinate_frames import (
    CoordinateFrame,
)
from gwcs.utils import _toindex, is_high_level

from ._fits import FitsMixin
from ._numerical import NumericalMixin
from ._pipeline import ForwardTransform, Pipeline

__all__ = ["WCS"]

_ITER_INV_KWARGS = ["tolerance", "maxiter", "adaptive", "detect_divergence", "quiet"]


class WCS(Pipeline, GWCSAPIMixin, NumericalMixin, FitsMixin):
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
    input_frame : str, `~gwcs.coordinate_frames.CoordinateFrame`
        A coordinates object or a string name.
    output_frame : str, `~gwcs.coordinate_frames.CoordinateFrame`
        A coordinates object or a string name.
    name : str
        a name for this WCS

    """

    def __init__(
        self,
        forward_transform: ForwardTransform = None,
        input_frame: CoordinateFrame | None = None,
        output_frame: CoordinateFrame | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(forward_transform, input_frame, output_frame)
        super(GWCSAPIMixin, self).__init__()

        self._approx_inverse = None
        self._name = "" if name is None else name
        self._pixel_shape = None

    def _add_units_input(
        self, arrays: np.ndarray | float, frame: CoordinateFrame | None
    ) -> tuple[u.Quantity, ...]:
        if frame is not None:
            return frame.add_units(arrays)

        return arrays

    def _remove_units_input(
        self, arrays: list[u.Quantity], frame: CoordinateFrame | None
    ) -> tuple[np.ndarray, ...]:
        if frame is not None:
            return frame.remove_units(arrays)

        return arrays

    def __call__(
        self,
        *args,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        with_units: bool = False,
        **kwargs,
    ):
        """
        Executes the forward transform.

        args : float or array-like
            Inputs in the input coordinate system, separate inputs
            for each dimension.
        with_bounding_box : bool, optional
             If True(default) values in the result which correspond to
             any of the inputs being outside the bounding_box are set
             to ``fill_value``.
        fill_value : float, optional
            Output value for inputs outside the bounding_box
            (default is np.nan).
        with_units : bool, optional
            If ``True`` then high level Astropy objects will be returned.
            Optional, default=False.
        """
        results = self._call_forward(
            *args, with_bounding_box=with_bounding_box, fill_value=fill_value, **kwargs
        )
        if with_units:
            # values are always expected to be arrays or scalars not quantities
            results = self._remove_units_input(results, self.output_frame)
            high_level = values_to_high_level_objects(*results, low_level_wcs=self)
            if len(high_level) == 1:
                high_level = high_level[0]
            return high_level
        return results

    def _evaluate_transform(
        self,
        transform,
        from_frame,
        to_frame,
        *args,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ):
        """
        Introduces or removes units from the arguments as need so that the transform
        can be successfully evaluated.

        Notes
        -----
        Much of the logic in this method is due to the unfortunate fact that the
        `uses_quantity` property for models is not reliable for determining if one
        must pass quantities or not. It instead tells you:
            1. If it has any parameter that is a quantity
            2. It defaults to true for parameterless models.

        This is problematic because its entirely possible to construct a model with
        a parameter that is a quantity but the model itself either doesn't require
        them or in fact cannot use them. This is a very rare case but it could happen.
        Currently, this case is not handled, but it is worth noting in case it comes up

        The more problematic case is for parameterless models. `uses_quantity` assumes
        that if there are no parameters, then the model is agnostic to quantity inputs.
        This is an incorrect assumption, even with in `astropy.modeling`'s built in
        models.  The `Tabular1D` model for example has no "parameters" but it can
        require quantities if its "points" construction input is a quantity. This
        is the main case for the try/except block in this method.

        Properly dealing with this will require upstream work in `astropy.modeling`
        which is outside the scope of what GWCS can control.

        to_frame is included as we really ought to be stripping the result of units
        but we currently are not. API refactor should include addressing this.
        """

        # Validate that the input type matches what the transform expects
        input_is_quantity = any(isinstance(a, u.Quantity) for a in args)

        def _transform(*args):
            """Wrap the transform evaluation"""

            return transform(
                *args,
                with_bounding_box=with_bounding_box,
                fill_value=fill_value,
                **kwargs,
            )

        # Models with no parameters claim they use quantities but this may incorrectly
        # introduce units so we don't at first
        if (
            not input_is_quantity
            and transform.uses_quantity
            and transform.parameters.size
        ):
            args = self._add_units_input(args, from_frame)
        if not transform.uses_quantity and input_is_quantity:
            args = self._remove_units_input(args, from_frame)

        try:
            return _transform(*args)
        except u.UnitsError:
            # In this case we are handling parameterless models that require units
            # to function correctly.
            if (
                not input_is_quantity
                and transform.uses_quantity
                and not transform.parameters.size
            ):
                return _transform(*self._add_units_input(args, from_frame))

            raise

    def _call_forward(
        self,
        *args,
        from_frame: CoordinateFrame | None = None,
        to_frame: CoordinateFrame | None = None,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ):
        """
        Executes the forward transform, but values only.
        """
        if from_frame is None and to_frame is None:
            transform = self.forward_transform
        else:
            transform = self.get_transform(from_frame, to_frame)
        if from_frame is None:
            from_frame = self.input_frame
        if to_frame is None:
            to_frame = self.output_frame

        if transform is None:
            msg = "WCS.forward_transform is not implemented."
            raise NotImplementedError(msg)

        return self._evaluate_transform(
            transform,
            from_frame,
            to_frame,
            *args,
            with_bounding_box=with_bounding_box,
            fill_value=fill_value,
            **kwargs,
        )

    def in_image(self, *args, **kwargs):
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
        coords = self.invert(*args, **kwargs)

        result = np.isfinite(coords)
        if self.input_frame.naxes > 1:
            result = np.all(result, axis=0)

        return result

    def invert(
        self,
        *args,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        with_units: bool = False,
        **kwargs,
    ):
        """
        Invert coordinates from output frame to input frame using analytical or
        user-supplied inverse. When neither analytical nor user-supplied
        inverses are defined, a numerical solution will be attempted using
        :py:meth:`numerical_inverse`.

        .. note::
            Currently numerical inverse is implemented only for 2D imaging WCS.

        Parameters
        ----------
        args : float, array like, `~astropy.coordinates.SkyCoord` or `~astropy.units.Unit`
            Coordinates to be inverted. The number of arguments must be equal
            to the number of world coordinates given by ``world_n_dim``.

        with_bounding_box : bool, optional
             If `True` (default) values in the result which correspond to any
             of the inputs being outside the bounding_box are set to
             ``fill_value``.

        fill_value : float, optional
            Output value for inputs outside the bounding_box (default is ``np.nan``).

        with_units : bool, optional
            If ``True`` then high level astropy object (i.e. ``Quantity``) will
            be returned.  Optional, default=False.

        Other Parameters
        ----------------
        kwargs : dict
            Keyword arguments to be passed to :py:meth:`numerical_inverse`
            (when defined) or to the iterative invert method.

        Returns
        -------
        result : tuple or value
            Returns a tuple of scalar or array values for each axis. Unless
            ``input_frame.naxes == 1`` when it shall return the value.
            The return type will be `~astropy.units.Quantity` objects if the
            transform returns ``Quantity`` objects, else values.

        """  # noqa: E501
        if is_high_level(*args, low_level_wcs=self):
            args = high_level_objects_to_values(*args, low_level_wcs=self)

        results = self._call_backward(
            *args, with_bounding_box=with_bounding_box, fill_value=fill_value, **kwargs
        )

        if with_units:
            # values are always expected to be arrays or scalars not quantities
            results = self._remove_units_input(results, self.input_frame)
            high_level = values_to_high_level_objects(
                *results, low_level_wcs=self.input_frame
            )
            if len(high_level) == 1:
                high_level = high_level[0]
            return high_level

        return results

    def _call_backward(
        self,
        *args,
        with_bounding_box: bool = True,
        fill_value: float | np.number = np.nan,
        **kwargs,
    ):
        try:
            transform = self.backward_transform
        except NotImplementedError:
            transform = None

        if with_bounding_box and self.bounding_box is not None:
            args = self.outside_footprint(args)

        if transform is not None:
            # remove iterative inverse-specific keyword arguments:
            akwargs = {k: v for k, v in kwargs.items() if k not in _ITER_INV_KWARGS}
            result = self._evaluate_transform(
                transform,
                self.output_frame,
                self.input_frame,
                *args,
                with_bounding_box=with_bounding_box,
                fill_value=fill_value,
                **akwargs,
            )
        else:
            # Always strip units for numerical inverse
            args = self._remove_units_input(args, self.output_frame)
            result = self._numerical_inverse(
                *args,
                with_bounding_box=with_bounding_box,
                fill_value=fill_value,
                **kwargs,
            )

        # deal with values outside the bounding box
        if with_bounding_box and self.bounding_box is not None:
            result = self.out_of_bounds(result, fill_value=fill_value)

        return result

    def outside_footprint(self, world_arrays):
        world_arrays = list(world_arrays)

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
                    coord_ = self._remove_quantity_output(
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
        return world_arrays

    def out_of_bounds(self, pixel_arrays, fill_value=np.nan):
        if np.isscalar(pixel_arrays) or self.input_frame.naxes == 1:
            pixel_arrays = [pixel_arrays]

        pixel_arrays = list(pixel_arrays)
        bbox = self.bounding_box
        for idim, pix in enumerate(pixel_arrays):
            outside = (pix < bbox[idim][0]) | (pix > bbox[idim][1])
            if np.any(outside):
                if np.isscalar(pix):
                    pixel_arrays[idim] = np.nan
                else:
                    pix_ = pixel_arrays[idim].astype(float, copy=True)
                    pix_[outside] = np.nan
                    pixel_arrays[idim] = pix_
        if self.input_frame.naxes == 1:
            pixel_arrays = pixel_arrays[0]
        return pixel_arrays

    def transform(
        self,
        from_frame: str | CoordinateFrame,
        to_frame: str | CoordinateFrame,
        *args,
        with_units: bool = False,
        **kwargs,
    ):
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
        """
        # Pull the steps and their indices from the pipeline
        # -> this also turns the frame name strings into frame objects
        from_step = self._get_step(from_frame)
        to_step = self._get_step(to_frame)

        # Determine if the transform is actually an inverse
        backward = to_step.idx < from_step.idx

        if backward and is_high_level(*args, low_level_wcs=from_step.step.frame):
            args = high_level_objects_to_values(
                *args, low_level_wcs=from_step.step.frame
            )

        results = self._call_forward(
            *args,
            from_frame=from_step.step.frame,
            to_frame=to_step.step.frame,
            **kwargs,
        )

        if with_units:
            # values are always expected to be arrays or scalars not quantities
            results = self._remove_units_input(results, to_step.step.frame)

            high_level = values_to_high_level_objects(
                *results, low_level_wcs=to_step.step.frame
            )
            if len(high_level) == 1:
                high_level = high_level[0]
            return high_level

        return results

    @property
    def name(self) -> str:
        """Return the name for this WCS."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the name for the WCS."""
        self._name = value

    def _get_axes_indices(self):
        try:
            axes_ind = np.argsort(self.input_frame.axes_order)
        except AttributeError:
            # the case of a frame being a string
            axes_ind = np.arange(self.forward_transform.n_inputs)
        return axes_ind

    def __str__(self):
        from astropy.table import Table

        col1 = [step.frame for step in self._pipeline]
        col2 = []
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

    def __repr__(self):
        return (
            f"<WCS(output_frame={self.output_frame}, input_frame={self.input_frame}, "
            f"forward_transform={self.forward_transform})>"
        )

    def footprint(self, bounding_box=None, center=False, axis_type="all"):
        """
        Return the footprint in world coordinates.

        Parameters
        ----------
        bounding_box : tuple of floats: (start, stop)
            ``prop: bounding_box``
        center : bool
            If `True` use the center of the pixel, otherwise use the corner.
        axis_type : str
            A supported ``output_frame.axes_type`` or ``"all"`` (default).
            One of [``'spatial'``, ``'spectral'``, ``'temporal'``] or a custom type.

        Returns
        -------
        coord : ndarray
            Array of coordinates in the output_frame mapping
            corners to the output frame. For spatial coordinates the order
            is clockwise, starting from the bottom left corner.

        """

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

        all_spatial = all(t.lower() == "spatial" for t in self.output_frame.axes_type)
        if self.output_frame.naxes == 1:
            if isinstance(bb[0], u.Quantity):
                bb = np.asarray([b.value for b in bb]) * bb[0].unit
            vertices = (bb,)
        elif all_spatial:
            vertices = _order_clockwise(
                [self._remove_units_input(b, self.input_frame) for b in bb]
            )
        else:
            vertices = np.array(list(itertools.product(*bb))).T

        # workaround an issue with bbox with quantity, interval needs to be a cquantity,
        # not a list of quantities strip units
        if center:
            vertices = _toindex(vertices)

        result = np.asarray(self.__call__(*vertices, with_bounding_box=False))

        axis_type = axis_type.lower()
        if axis_type == "spatial" and all_spatial:
            return result.T

        if axis_type != "all":
            axtyp_ind = (
                np.array([t.lower() for t in self.output_frame.axes_type]) == axis_type
            )
            if not axtyp_ind.any():
                msg = f'This WCS does not have axis of type "{axis_type}".'
                raise ValueError(msg)
            if len(axtyp_ind) > 1:
                result = np.asarray([(r.min(), r.max()) for r in result[axtyp_ind]])

            if axis_type == "spatial":
                result = _order_clockwise(result)
            else:
                result.sort()
                result = np.squeeze(result)
        if self.output_frame.naxes == 1:
            return np.array([result]).T
        return result.T

    def fix_inputs(self, fixed):
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
        new_pipeline = []
        step0 = self.pipeline[0]
        new_transform = fix_inputs(step0[1], fixed)
        new_pipeline.append((step0[0], new_transform))
        new_pipeline.extend(self.pipeline[1:])
        return self.__class__(new_pipeline)
