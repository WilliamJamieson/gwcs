# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import annotations

import functools
import itertools

import astropy.units as u
import numpy as np
from astropy import utils as astutil
from astropy.modeling import fix_inputs
from astropy.modeling.models import (
    Mapping,
    RotateCelestial2Native,
    Shift,
    Sky2Pix_TAN,
)
from astropy.modeling.parameters import _tofloat
from astropy.wcs.wcsapi.high_level_api import (
    high_level_objects_to_values,
    values_to_high_level_objects,
)
from scipy import optimize

from gwcs._typing import BoundingBoxTuple, LowLevelArrays, LowOrHigh, Real
from gwcs.api import GWCSAPIMixin
from gwcs.coordinate_frames import (
    BaseCoordinateFrame,
    CelestialFrame,
)
from gwcs.utils import is_high_level, to_index

from ._exception import NoConvergence
from ._fits import FitsMixin
from ._pipeline import ForwardTransform, Pipeline
from ._utils import (
    fit_2D_poly,
    make_sampling_grid,
)

__all__ = ["WCS"]

_ITER_INV_KWARGS = ["tolerance", "maxiter", "adaptive", "detect_divergence", "quiet"]


class WCS(Pipeline, GWCSAPIMixin, FitsMixin):
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
    input_frame : str, `~gwcs.coordinate_frames.BaseCoordinateFrame`
        A coordinates object or a string name.
    output_frame : str, `~gwcs.coordinate_frames.BaseCoordinateFrame`
        A coordinates object or a string name.
    name : str
        a name for this WCS

    """

    def __init__(
        self,
        forward_transform: ForwardTransform = None,
        input_frame: BaseCoordinateFrame | None = None,
        output_frame: BaseCoordinateFrame | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(forward_transform, input_frame, output_frame)

        self._approx_inverse = None
        self._name = "" if name is None else name
        self._pixel_shape = None

    def _add_units_input(
        self, arrays: list[np.ndarray], frame: BaseCoordinateFrame | None
    ) -> tuple[u.Quantity, ...]:
        if frame is not None:
            return tuple(
                u.Quantity(array, unit)
                for array, unit in zip(arrays, frame.unit, strict=False)
            )

        return arrays

    def _remove_units_input(
        self, arrays: list[u.Quantity], frame: BaseCoordinateFrame | None
    ) -> tuple[np.ndarray, ...]:
        if frame is not None:
            return tuple(
                array.to_value(unit) if isinstance(array, u.Quantity) else array
                for array, unit in zip(arrays, frame.unit, strict=False)
            )

        return arrays

    def __call__(
        self,
        *args: LowLevelArrays,
        with_bounding_box: bool = True,
        fill_value: Real = np.nan,
        with_units: bool = False,
        **kwargs,
    ) -> LowLevelArrays:
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
            if not astutil.isiterable(results):
                results = (results,)
            # values are always expected to be arrays or scalars not quantities
            results = self._remove_units_input(results, self.output_frame)
            high_level = values_to_high_level_objects(*results, low_level_wcs=self)
            if len(high_level) == 1:
                high_level = high_level[0]
            return high_level
        return results

    def _call_forward(
        self,
        *args: LowLevelArrays,
        from_frame: BaseCoordinateFrame | None = None,
        to_frame: BaseCoordinateFrame | None = None,
        with_bounding_box: bool = True,
        fill_value: Real = np.nan,
        **kwargs,
    ) -> LowLevelArrays:
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

        # Validate that the input type matches what the transform expects
        input_is_quantity = any(isinstance(a, u.Quantity) for a in args)
        if not input_is_quantity and transform.uses_quantity:
            args = self._add_units_input(args, from_frame)
        if not transform.uses_quantity and input_is_quantity:
            args = self._remove_units_input(args, from_frame)

        return transform(
            *args, with_bounding_box=with_bounding_box, fill_value=fill_value, **kwargs
        )

    def in_image(self, *args: LowOrHigh, **kwargs) -> bool | np.ndarray:
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
        *args: LowOrHigh,
        with_bounding_box: bool = True,
        fill_value: Real = np.nan,
        with_units: bool = False,
        **kwargs,
    ) -> LowLevelArrays:
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
        *args: LowLevelArrays,
        with_bounding_box: bool = True,
        fill_value: Real = np.nan,
        **kwargs,
    ) -> LowLevelArrays:
        try:
            transform = self.backward_transform
        except NotImplementedError:
            transform = None

        if with_bounding_box and self.bounding_box is not None:
            args = self.outside_footprint(args)

        if transform is not None:
            # Validate that the input type matches what the transform expects
            input_is_quantity = any(isinstance(a, u.Quantity) for a in args)
            if not input_is_quantity and transform.uses_quantity:
                args = self._add_units_input(args, self.output_frame)
            if not transform.uses_quantity and input_is_quantity:
                args = self._remove_units_input(args, self.output_frame)

            # remove iterative inverse-specific keyword arguments:
            akwargs = {k: v for k, v in kwargs.items() if k not in _ITER_INV_KWARGS}
            result = transform(
                *args,
                with_bounding_box=with_bounding_box,
                fill_value=fill_value,
                **akwargs,
            )
        else:
            # Always strip units for numerical inverse
            args = self._remove_units_input(args, self.output_frame)
            result = self.numerical_inverse(
                *args,
                with_bounding_box=with_bounding_box,
                fill_value=fill_value,
                **kwargs,
            )

        # deal with values outside the bounding box
        if with_bounding_box and self.bounding_box is not None:
            result = self.out_of_bounds(result, fill_value=fill_value)

        return result

    def outside_footprint(self, world_arrays: LowOrHigh) -> LowOrHigh:
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
                    outside = (coord < min_ax) | (coord > max_ax)
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

    def out_of_bounds(
        self, pixel_arrays: LowLevelArrays, fill_value=np.nan
    ) -> LowLevelArrays:
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

    def numerical_inverse(
        self,
        *args: LowOrHigh,
        tolerance: Real = 1e-5,
        maxiter: int = 30,
        adaptive: bool = True,
        detect_divergence: bool = True,
        quiet: bool = True,
        with_bounding_box: bool = True,
        fill_value: Real = np.nan,
        **kwargs,
    ) -> LowLevelArrays:
        """
        Invert coordinates from output frame to input frame using numerical
        inverse.

        .. note::
            Currently numerical inverse is implemented only for 2D imaging WCS.

        .. note::
            This method uses a combination of vectorized fixed-point
            iterations algorithm and `scipy.optimize.root`. The later is used
            for input coordinates for which vectorized algorithm diverges.

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

        tolerance : float, optional
            *Absolute tolerance* of solution. Iteration terminates when the
            iterative solver estimates that the "true solution" is
            within this many pixels current estimate, more
            specifically, when the correction to the solution found
            during the previous iteration is smaller
            (in the sense of the L2 norm) than ``tolerance``.
            Default ``tolerance`` is 1.0e-5.

        maxiter : int, optional
            Maximum number of iterations allowed to reach a solution.
            Default is 50.

        quiet : bool, optional
            Do not throw :py:class:`NoConvergence` exceptions when
            the method does not converge to a solution with the
            required accuracy within a specified number of maximum
            iterations set by ``maxiter`` parameter. Instead,
            simply return the found solution. Default is `True`.

        adaptive : bool, optional
            Specifies whether to adaptively select only points that
            did not converge to a solution within the required
            accuracy for the next iteration. Default (`True`) is recommended.

            .. note::
               The :py:meth:`numerical_inverse` uses a vectorized
               implementation of the method of consecutive
               approximations (see ``Notes`` section below) in which it
               iterates over *all* input points *regardless* until
               the required accuracy has been reached for *all* input
               points. In some cases it may be possible that
               *almost all* points have reached the required accuracy
               but there are only a few of input data points for
               which additional iterations may be needed (this
               depends mostly on the characteristics of the geometric
               distortions for a given instrument). In this situation
               it may be advantageous to set ``adaptive`` = `True` in
               which case :py:meth:`numerical_inverse` will continue
               iterating *only* over the points that have not yet
               converged to the required accuracy.

            .. note::
               When ``detect_divergence`` is `True`,
               :py:meth:`numerical_inverse` will automatically switch
               to the adaptive algorithm once divergence has been
               detected.

        detect_divergence : bool, optional
            Specifies whether to perform a more detailed analysis
            of the convergence to a solution. Normally
            :py:meth:`numerical_inverse` may not achieve the required
            accuracy if either the ``tolerance`` or ``maxiter`` arguments
            are too low. However, it may happen that for some
            geometric distortions the conditions of convergence for
            the the method of consecutive approximations used by
            :py:meth:`numerical_inverse` may not be satisfied, in which
            case consecutive approximations to the solution will
            diverge regardless of the ``tolerance`` or ``maxiter``
            settings.

            When ``detect_divergence`` is `False`, these divergent
            points will be detected as not having achieved the
            required accuracy (without further details). In addition,
            if ``adaptive`` is `False` then the algorithm will not
            know that the solution (for specific points) is diverging
            and will continue iterating and trying to "improve"
            diverging solutions. This may result in ``NaN`` or
            ``Inf`` values in the return results (in addition to a
            performance penalties). Even when ``detect_divergence``
            is `False`, :py:meth:`numerical_inverse`, at the end of the
            iterative process, will identify invalid results
            (``NaN`` or ``Inf``) as "diverging" solutions and will
            raise :py:class:`NoConvergence` unless the ``quiet``
            parameter is set to `True`.

            When ``detect_divergence`` is `True` (default),
            :py:meth:`numerical_inverse` will detect points for which
            current correction to the coordinates is larger than
            the correction applied during the previous iteration
            **if** the requested accuracy **has not yet been
            achieved**. In this case, if ``adaptive`` is `True`,
            these points will be excluded from further iterations and
            if ``adaptive`` is `False`, :py:meth:`numerical_inverse` will
            automatically switch to the adaptive algorithm. Thus, the
            reported divergent solution will be the latest converging
            solution computed immediately *before* divergence
            has been detected.

            .. note::
               When accuracy has been achieved, small increases in
               current corrections may be possible due to rounding
               errors (when ``adaptive`` is `False`) and such
               increases will be ignored.

            .. note::
               Based on our testing using JWST NIRCAM images, setting
               ``detect_divergence`` to `True` will incur about 5-10%
               performance penalty with the larger penalty
               corresponding to ``adaptive`` set to `True`.
               Because the benefits of enabling this
               feature outweigh the small performance penalty,
               especially when ``adaptive`` = `False`, it is
               recommended to set ``detect_divergence`` to `True`,
               unless extensive testing of the distortion models for
               images from specific instruments show a good stability
               of the numerical method for a wide range of
               coordinates (even outside the image itself).

            .. note::
               Indices of the diverging inverse solutions will be
               reported in the ``divergent`` attribute of the
               raised :py:class:`NoConvergence` exception object.

        Returns
        -------
        result : tuple
            Returns a tuple of scalar or array values for each axis.

        Raises
        ------
        NoConvergence
            The iterative method did not converge to a
            solution to the required accuracy within a specified
            number of maximum iterations set by the ``maxiter``
            parameter. To turn off this exception, set ``quiet`` to
            `True`. Indices of the points for which the requested
            accuracy was not achieved (if any) will be listed in the
            ``slow_conv`` attribute of the
            raised :py:class:`NoConvergence` exception object.

            See :py:class:`NoConvergence` documentation for
            more details.

        NotImplementedError
            Numerical inverse has not been implemented for this WCS.

        ValueError
            Invalid argument values.

        Examples
        --------
        >>> from astropy.utils.data import get_pkg_data_filename
        >>> from gwcs import NoConvergence
        >>> import asdf
        >>> import numpy as np

        >>> filename = get_pkg_data_filename('data/nircamwcs.asdf', package='gwcs.tests')
        >>> with asdf.open(filename, lazy_load=False, ignore_missing_extensions=True) as af:
        ...    w = af.tree['wcs']

        >>> ra, dec = w([1,2,3], [1,1,1])
        >>> assert np.allclose(ra, [5.927628, 5.92757069, 5.92751337]);
        >>> assert np.allclose(dec, [-72.01341247, -72.01341273, -72.013413])

        >>> x, y = w.numerical_inverse(ra, dec)
        >>> assert np.allclose(x, [1.00000005, 2.00000005, 3.00000006]);
        >>> assert np.allclose(y, [1.00000004, 0.99999979, 1.00000015]);

        >>> x, y = w.numerical_inverse(ra, dec, maxiter=3, tolerance=1.0e-10, quiet=False)
        Traceback (most recent call last):
        ...
        gwcs.wcs._exception.NoConvergence: 'WCS.numerical_inverse' failed to converge to the
        requested accuracy after 3 iterations.

        >>> w.numerical_inverse(
        ...     *w([1, 300000, 3], [2, 1000000, 5], with_bounding_box=False),
        ...     adaptive=False,
        ...     detect_divergence=True,
        ...     quiet=False,
        ...     with_bounding_box=False
        ... )
        Traceback (most recent call last):
        ...
        gwcs.wcs._exception.NoConvergence: 'WCS.numerical_inverse' failed to converge to the
        requested accuracy. After 4 iterations, the solution is diverging at
        least for one input point.

        >>> # Now try to use some diverging data:
        >>> divra, divdec = w([1, 300000, 3], [2, 1000000, 5], with_bounding_box=False)
        >>> assert np.allclose(divra, [5.92762673, 148.21600848, 5.92750827])
        >>> assert np.allclose(divdec, [-72.01339464, -7.80968079, -72.01334172])
        >>> try:  # doctest: +SKIP
        ...     x, y = w.numerical_inverse(divra, divdec, maxiter=20,
        ...                                tolerance=1.0e-4, adaptive=True,
        ...                                detect_divergence=True,
        ...                                quiet=False)
        ... except NoConvergence as e:
        ...     print(f"Indices of diverging points: {e.divergent}")
        ...     print(f"Indices of poorly converging points: {e.slow_conv}")
        ...     print(f"Best solution:\\n{e.best_solution}")
        ...     print(f"Achieved accuracy:\\n{e.accuracy}")
        Indices of diverging points: None
        Indices of poorly converging points: [1]
        Best solution:
        [[1.00000040e+00 1.99999841e+00]
         [6.33507833e+17 3.40118820e+17]
         [3.00000038e+00 4.99999841e+00]]
        Achieved accuracy:
        [[2.75925982e-05 1.18471543e-05]
         [3.65405005e+04 1.31364188e+04]
         [2.76552923e-05 1.14789013e-05]]

        """  # noqa: E501
        if kwargs.pop("with_units", False):
            msg = (
                "Support for with_units in numerical_inverse has been removed, "
                "use inverse"
            )
            raise ValueError(msg)

        args_shape = np.shape(args)
        nargs = args_shape[0]
        arg_dim = len(args_shape) - 1

        if nargs != self.world_n_dim:
            msg = (
                "Number of input coordinates is different from "
                "the number of defined world coordinates in the "
                f"WCS ({self.world_n_dim:d})"
            )
            raise ValueError(msg)

        if self.world_n_dim != self.pixel_n_dim:
            msg = (
                "Support for iterative inverse for transformations with "
                "different number of inputs and outputs was not implemented."
            )
            raise NotImplementedError(msg)

        # initial guess:
        if nargs == 2 and self._approx_inverse is None:
            self._calc_approx_inv(max_inv_pix_error=5, inv_degree=None)

        if self._approx_inverse is None:
            if self.bounding_box is None:
                x0 = np.ones(self.pixel_n_dim)
            else:
                x0 = np.mean(self.bounding_box, axis=-1)

        if arg_dim == 0:
            argsi = args

            if nargs == 2 and self._approx_inverse is not None:
                x0 = self._approx_inverse(*argsi)
                if not np.all(np.isfinite(x0)):
                    return [np.array(np.nan) for _ in range(nargs)]

            result = tuple(
                self._vectorized_fixed_point(
                    x0,
                    argsi,
                    tolerance=tolerance,
                    maxiter=maxiter,
                    adaptive=adaptive,
                    detect_divergence=detect_divergence,
                    quiet=quiet,
                    with_bounding_box=with_bounding_box,
                    fill_value=fill_value,
                )
                .T.ravel()
                .tolist()
            )

        else:
            arg_shape = args_shape[1:]
            nelem = np.prod(arg_shape)

            args = np.reshape(args, (nargs, nelem))

            if self._approx_inverse is None:
                x0 = np.full((nelem, nargs), x0)
            else:
                x0 = np.array(self._approx_inverse(*args)).T

            result = self._vectorized_fixed_point(
                x0,
                args.T,
                tolerance=tolerance,
                maxiter=maxiter,
                adaptive=adaptive,
                detect_divergence=detect_divergence,
                quiet=quiet,
                with_bounding_box=with_bounding_box,
                fill_value=fill_value,
            ).T

            result = tuple(np.reshape(result, args_shape))

        return result

    def _vectorized_fixed_point(
        self,
        pix0,
        world,
        tolerance,
        maxiter,
        adaptive,
        detect_divergence,
        quiet,
        with_bounding_box,
        fill_value,
    ):
        # ############################################################
        # #            INITIALIZE ITERATIVE PROCESS:                ##
        # ############################################################

        # make a copy of the initial approximation
        pix0 = np.atleast_2d(np.array(pix0))  # 0-order solution
        pix = np.array(pix0)

        world0 = np.atleast_2d(np.array(world))
        world = np.array(world0)

        # estimate pixel scale using approximate algorithm
        # from https://trs.jpl.nasa.gov/handle/2014/40409
        if self.bounding_box is None:
            crpix = np.ones(self.pixel_n_dim)
        else:
            crpix = np.mean(self.bounding_box, axis=-1)

        l1, phi1 = np.deg2rad(self.__call__(*(crpix - 0.5)))
        l2, phi2 = np.deg2rad(self.__call__(*(crpix + [-0.5, 0.5])))  # noqa: RUF005
        l3, phi3 = np.deg2rad(self.__call__(*(crpix + 0.5)))
        l4, phi4 = np.deg2rad(self.__call__(*(crpix + [0.5, -0.5])))  # noqa: RUF005
        area = np.abs(
            0.5
            * (
                (l4 - l2) * (np.sin(phi1) - np.sin(phi3))
                + (l1 - l3) * (np.sin(phi2) - np.sin(phi4))
            )
        )
        inv_pscale = 1 / np.rad2deg(np.sqrt(area))

        # form equation:
        def f(x):
            w = np.array(self.__call__(*(x.T), with_bounding_box=False)).T
            dw = np.mod(np.subtract(w, world) - 180.0, 360.0) - 180.0
            return np.add(inv_pscale * dw, x)

        def froot(x):
            return (
                np.mod(
                    np.subtract(self.__call__(*x, with_bounding_box=False), worldi)
                    - 180.0,
                    360.0,
                )
                - 180.0
            )

        # compute correction:
        def correction(pix):
            p1 = f(pix)
            p2 = f(p1)
            d = p2 - 2.0 * p1 + pix
            idx = np.where(d != 0)
            corr = pix - p2
            corr[idx] = np.square(p1[idx] - pix[idx]) / d[idx]
            return corr

        # initial iteration:
        dpix = correction(pix)

        # Update initial solution:
        pix -= dpix

        # Norm (L2) squared of the correction:
        dn = np.sum(dpix * dpix, axis=1)
        dnprev = dn.copy()  # if adaptive else dn
        tol2 = tolerance**2

        # Prepare for iterative process
        k = 1
        ind = None
        inddiv = None

        # Turn off numpy runtime warnings for 'invalid' and 'over':
        old_invalid = np.geterr()["invalid"]
        old_over = np.geterr()["over"]
        np.seterr(invalid="ignore", over="ignore")

        # ############################################################
        # #                NON-ADAPTIVE ITERATIONS:                 ##
        # ############################################################
        if not adaptive:
            # Fixed-point iterations:
            while np.nanmax(dn) >= tol2 and k < maxiter:
                # Find correction to the previous solution:
                dpix = correction(pix)

                # Compute norm (L2) squared of the correction:
                dn = np.sum(dpix * dpix, axis=1)

                # Check for divergence (we do this in two stages
                # to optimize performance for the most common
                # scenario when successive approximations converge):

                if detect_divergence:
                    divergent = dn >= dnprev
                    if np.any(divergent):
                        # Find solutions that have not yet converged:
                        slowconv = dn >= tol2
                        (inddiv,) = np.where(divergent & slowconv)

                        if inddiv.shape[0] > 0:
                            # Update indices of elements that
                            # still need correction:
                            conv = dn < dnprev
                            iconv = np.where(conv)

                            # Apply correction:
                            dpixgood = dpix[iconv]
                            pix[iconv] -= dpixgood
                            dpix[iconv] = dpixgood

                            # For the next iteration choose
                            # non-divergent points that have not yet
                            # converged to the requested accuracy:
                            (ind,) = np.where(slowconv & conv)
                            world = world[ind]
                            dnprev[ind] = dn[ind]
                            k += 1

                            # Switch to adaptive iterations:
                            adaptive = True
                            break

                    # Save current correction magnitudes for later:
                    dnprev = dn

                # Apply correction:
                pix -= dpix
                k += 1

        # ############################################################
        # #                  ADAPTIVE ITERATIONS:                   ##
        # ############################################################
        if adaptive:
            if ind is None:
                (ind,) = np.where(np.isfinite(pix).all(axis=1))
                world = world[ind]

            # "Adaptive" fixed-point iterations:
            while ind.shape[0] > 0 and k < maxiter:
                # Find correction to the previous solution:
                dpixnew = correction(pix[ind])

                # Compute norm (L2) of the correction:
                dnnew = np.sum(np.square(dpixnew), axis=1)

                # Bookkeeping of corrections:
                dnprev[ind] = dn[ind].copy()
                dn[ind] = dnnew

                if detect_divergence:
                    # Find indices of pixels that are converging:
                    conv = np.logical_or(dnnew < dnprev[ind], dnnew < tol2)
                    if not np.all(conv):
                        conv = np.ones_like(dnnew, dtype=bool)
                    iconv = np.where(conv)
                    iiconv = ind[iconv]

                    # Apply correction:
                    dpixgood = dpixnew[iconv]
                    pix[iiconv] -= dpixgood
                    dpix[iiconv] = dpixgood

                    # Find indices of solutions that have not yet
                    # converged to the requested accuracy
                    # AND that do not diverge:
                    (subind,) = np.where((dnnew >= tol2) & conv)

                else:
                    # Apply correction:
                    pix[ind] -= dpixnew
                    dpix[ind] = dpixnew

                    # Find indices of solutions that have not yet
                    # converged to the requested accuracy:
                    (subind,) = np.where(dnnew >= tol2)

                # Choose solutions that need more iterations:
                ind = ind[subind]
                world = world[subind]

                k += 1

        # ############################################################
        # #         FINAL DETECTION OF INVALID, DIVERGING,          ##
        # #         AND FAILED-TO-CONVERGE POINTS                   ##
        # ############################################################
        # Identify diverging and/or invalid points:
        invalid = (~np.all(np.isfinite(pix), axis=1)) & (
            np.all(np.isfinite(world0), axis=1)
        )

        # When detect_divergence is False, dnprev is outdated
        # (it is the norm of the very first correction).
        # Still better than nothing...
        (inddiv,) = np.where(((dn >= tol2) & (dn >= dnprev)) | invalid)
        if inddiv.shape[0] == 0:
            inddiv = None

        # If there are divergent points, attempt to find a solution using
        # scipy's 'hybr' method:
        if detect_divergence and inddiv is not None and inddiv.size:
            bad = []
            for idx in inddiv:
                worldi = world0[idx]
                result = optimize.root(
                    froot,
                    pix0[idx],
                    method="hybr",
                    tol=tolerance / (np.linalg.norm(pix0[idx]) + 1),
                    options={"maxfev": 2 * maxiter},
                )

                if result["success"]:
                    pix[idx, :] = result["x"]
                    invalid[idx] = False
                else:
                    bad.append(idx)

            inddiv = np.array(bad, dtype=int) if bad else None

        # Identify points that did not converge within 'maxiter'
        # iterations:
        if k >= maxiter:
            (ind,) = np.where((dn >= tol2) & (dn < dnprev) & (~invalid))
            if ind.shape[0] == 0:
                ind = None
        else:
            ind = None

        # Restore previous numpy error settings:
        np.seterr(invalid=old_invalid, over=old_over)

        # ############################################################
        # #  RAISE EXCEPTION IF DIVERGING OR TOO SLOWLY CONVERGING  ##
        # #  DATA POINTS HAVE BEEN DETECTED:                        ##
        # ############################################################
        if (ind is not None or inddiv is not None) and not quiet:
            if inddiv is None:
                msg = (
                    "'WCS.numerical_inverse' failed to "
                    f"converge to the requested accuracy after {k:d} "
                    "iterations."
                )
                raise NoConvergence(
                    msg,
                    best_solution=pix,
                    accuracy=np.abs(dpix),
                    niter=k,
                    slow_conv=ind,
                    divergent=None,
                )
            msg = (
                "'WCS.numerical_inverse' failed to "
                "converge to the requested accuracy.\n"
                f"After {k:d} iterations, the solution is diverging "
                "at least for one input point."
            )
            raise NoConvergence(
                msg,
                best_solution=pix,
                accuracy=np.abs(dpix),
                niter=k,
                slow_conv=ind,
                divergent=inddiv,
            )

        if with_bounding_box and self.bounding_box is not None:
            # find points outside the bounding box and replace their values
            # with fill_value
            valid = np.logical_not(invalid)
            in_bb = np.ones_like(invalid, dtype=np.bool_)

            for c, (x1, x2) in zip(pix[valid].T, self.bounding_box, strict=False):
                in_bb[valid] &= (c >= x1) & (c <= x2)
            pix[np.logical_not(in_bb)] = fill_value

        return pix

    def transform(
        self,
        from_frame: str | BaseCoordinateFrame,
        to_frame: str | BaseCoordinateFrame,
        *args: LowOrHigh,
        with_units: bool = False,
        **kwargs,
    ) -> LowOrHigh:
        """
        Transform positions between two frames.

        Parameters
        ----------
        from_frame : str or `~gwcs.coordinate_frames.BaseCoordinateFrame`
            Initial coordinate frame.
        to_frame : str, or instance of `~gwcs.coordinate_frames.BaseCoordinateFrame`
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
        backward = to_step.index < from_step.index

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
    def name(self, value: str) -> None:
        """Set the name for the WCS."""
        self._name = value

    def __str__(self) -> str:
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

    def __repr__(self) -> str:
        return (
            f"<WCS(output_frame={self.output_frame}, input_frame={self.input_frame}, "
            f"forward_transform={self.forward_transform})>"
        )

    def footprint(
        self,
        bounding_box: BoundingBoxTuple | None = None,
        center: bool = False,
        axis_type: str = "all",
    ) -> np.ndarray:
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
            vertices = _order_clockwise(bb)
        else:
            vertices = np.array(list(itertools.product(*bb))).T

        # workaround an issue with bbox with quantity, interval needs to be a cquantity,
        # not a list of quantities strip units
        if center:
            vertices = to_index(vertices)

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

    def fix_inputs(self, fixed: dict) -> WCS:
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

    def _calc_approx_inv(self, max_inv_pix_error=5, inv_degree=None, npoints=16):
        """
        Compute polynomial fit for the inverse transformation to be used as
        initial approximation/guess for the iterative solution.
        """
        self._approx_inverse = None

        try:
            # try to use analytic inverse if available:
            self._approx_inverse = functools.partial(
                self.backward_transform, with_bounding_box=False
            )
        except (NotImplementedError, KeyError):
            pass
        else:
            return

        if not isinstance(self.output_frame, CelestialFrame):
            # The _calc_approx_inv method only works with celestial frame transforms
            return

        # Determine reference points.
        if self.bounding_box is None:
            # A bounding_box is needed to proceed.
            return

        crpix = np.mean(self.bounding_box, axis=1)

        crval1, crval2 = self.forward_transform(*crpix)

        # Rotate to native system and deproject. Set center of the projection
        # transformation to the middle of the bounding box ("image") in order
        # to minimize projection effects across the entire image,
        # thus the initial shift.
        ntransform = (
            (Shift(crpix[0]) & Shift(crpix[1]))
            | self.forward_transform
            | RotateCelestial2Native(crval1, crval2, 180)
            | Sky2Pix_TAN()
        )

        # standard sampling:
        u, v = make_sampling_grid(npoints, self.bounding_box, crpix=crpix)
        undist_x, undist_y = ntransform(u, v)

        # Double sampling to check if sampling is sufficient.
        ud, vd = make_sampling_grid(2 * npoints, self.bounding_box, crpix=crpix)
        undist_xd, undist_yd = ntransform(ud, vd)

        fit_inv_poly_u, fit_inv_poly_v, max_inv_resid = fit_2D_poly(
            None,
            max_inv_pix_error,
            1,
            undist_x,
            undist_y,
            u,
            v,
            undist_xd,
            undist_yd,
            ud,
            vd,
            verbose=True,
        )

        self._approx_inverse = (
            RotateCelestial2Native(crval1, crval2, 180)
            | Sky2Pix_TAN()
            | Mapping((0, 1, 0, 1))
            | (fit_inv_poly_u & fit_inv_poly_v)
            | (Shift(crpix[0]) & Shift(crpix[1]))
        )
