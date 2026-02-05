"""
This module contains the implementation of the numerical inverse for 2D imaging used
by GWCS by default
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Protocol

import numpy as np
from astropy.modeling.models import (
    Mapping,
    RotateCelestial2Native,
    Shift,
    Sky2Pix_TAN,
)
from astropy.modeling.projections import Projection
from scipy import optimize

from gwcs.api import LowLevelArray
from gwcs.coordinate_frames import CelestialFrame
from gwcs.utils import _compute_lon_pole

from ._exception import NoConvergence
from ._utils import (
    fit_2D_poly,
    make_sampling_grid,
)

if TYPE_CHECKING:
    from ._wcs import WCS

__all__ = ["NumericalInverse2D", "NumericalInverseProtocol"]


class NumericalInverseProtocol(Protocol):
    """
    Protocol for a callable that performs a numerical inverse transformation of the WCS.

    Note
    ----
        The default implementation of this protocol is `~gwcs.wcs.NumericalInverse2D`,
        which is only valid for 2D WCS transformations.

        You can implement your own version of this protocol to handle WCS
        transformations and set that as the ``numerical_inverse_method`` attribute of
        the WCS object. The method should satisfy this protocol.
    """

    def __call__(
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
    ) -> tuple[LowLevelArray, ...] | LowLevelArray: ...


class ApproxInverseProtocol(Protocol):
    def __call__(
        self, *args: LowLevelArray
    ) -> LowLevelArray | tuple[LowLevelArray]: ...


class NumericalInverse2D:
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
    wcs : `~gwcs.wcs.WCS`
        The WCS object for which the numerical inverse is to be computed.
    """

    def __init__(self, wcs: WCS) -> None:
        self.wcs = wcs
        self.approx_inverse: ApproxInverseProtocol | None = None

    def __call__(
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
        """
        Run the numerical inverse transformation.

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
                The :py:meth:`__call__` uses a vectorized
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
                which case :py:meth:`__call__` will continue
                iterating *only* over the points that have not yet
                converged to the required accuracy.

            .. note::
                When ``detect_divergence`` is `True`,
                :py:meth:`__call__` will automatically switch
                to the adaptive algorithm once divergence has been
                detected.

        detect_divergence : bool, optional
            Specifies whether to perform a more detailed analysis
            of the convergence to a solution. Normally
            :py:meth:`__call__` may not achieve the required
            accuracy if either the ``tolerance`` or ``maxiter`` arguments
            are too low. However, it may happen that for some
            geometric distortions the conditions of convergence for
            the the method of consecutive approximations used by
            :py:meth:`__call__` may not be satisfied, in which
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
            is `False`, :py:meth:`__call__`, at the end of the
            iterative process, will identify invalid results
            (``NaN`` or ``Inf``) as "diverging" solutions and will
            raise :py:class:`NoConvergence` unless the ``quiet``
            parameter is set to `True`.

            When ``detect_divergence`` is `True` (default),
            :py:meth:`__call__` will detect points for which
            current correction to the coordinates is larger than
            the correction applied during the previous iteration
            **if** the requested accuracy **has not yet been
            achieved**. In this case, if ``adaptive`` is `True`,
            these points will be excluded from further iterations and
            if ``adaptive`` is `False`, :py:meth:`__call__` will
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
        return self._numerical_inverse(
            *self.wcs.output_frame.remove_units(args),
            tolerance=tolerance,
            maxiter=maxiter,
            adaptive=adaptive,
            detect_divergence=detect_divergence,
            quiet=quiet,
            with_bounding_box=with_bounding_box,
            fill_value=fill_value,
            **kwargs,
        )

    def _numerical_inverse(
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
        args_shape = np.shape(args)
        nargs = args_shape[0]
        arg_dim = len(args_shape) - 1

        if nargs != self.wcs.world_n_dim:
            msg = (
                "Number of input coordinates is different from "
                "the number of defined world coordinates in the "
                f"WCS ({self.wcs.world_n_dim:d})"
            )
            raise ValueError(msg)

        if self.wcs.world_n_dim != self.wcs.pixel_n_dim:
            msg = (
                "Support for iterative inverse for transformations with "
                "different number of inputs and outputs was not implemented."
            )
            raise NotImplementedError(msg)

        # initial guess:
        if nargs == 2 and self.approx_inverse is None:
            self._calc_approx_inv(max_inv_pix_error=5, inv_degree=None)

        x0: LowLevelArray | tuple[LowLevelArray, ...]
        if self.approx_inverse is None:
            if self.wcs.bounding_box is None:
                x0 = np.ones(self.wcs.pixel_n_dim)
            else:
                x0 = np.mean(self.wcs.bounding_box, axis=-1)

        if arg_dim == 0:
            argsi = args

            if nargs == 2 and self.approx_inverse is not None:
                x0 = self.approx_inverse(*argsi)
                if not np.all(np.isfinite(x0)):
                    return tuple(np.array(np.nan) for _ in range(nargs))

            return tuple(
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

        arg_shape = args_shape[1:]
        nelem = np.prod(arg_shape)

        argsj = np.reshape(args, (nargs, nelem))

        if self.approx_inverse is None:
            x0 = np.full((nelem, nargs), x0)
        else:
            x0 = np.array(self.approx_inverse(*argsj)).T

        result = self._vectorized_fixed_point(
            x0,
            argsj.T,
            tolerance=tolerance,
            maxiter=maxiter,
            adaptive=adaptive,
            detect_divergence=detect_divergence,
            quiet=quiet,
            with_bounding_box=with_bounding_box,
            fill_value=fill_value,
        ).T

        return tuple(np.reshape(result, args_shape))

    def _vectorized_fixed_point(
        self,
        pix0: LowLevelArray | tuple[LowLevelArray, ...],
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
        if self.wcs.bounding_box is None:
            crpix = np.ones(self.wcs.pixel_n_dim)
        else:
            crpix = np.mean(self.wcs.bounding_box, axis=-1)

        l1, phi1 = np.deg2rad(self.wcs.evaluate(*(crpix - 0.5)))
        l2, phi2 = np.deg2rad(self.wcs.evaluate(*(crpix + [-0.5, 0.5])))  # noqa: RUF005
        l3, phi3 = np.deg2rad(self.wcs.evaluate(*(crpix + 0.5)))
        l4, phi4 = np.deg2rad(self.wcs.evaluate(*(crpix + [0.5, -0.5])))  # noqa: RUF005
        area = np.abs(
            0.5
            * (
                (l4 - l2) * (np.sin(phi1) - np.sin(phi3))
                + (l1 - l3) * (np.sin(phi2) - np.sin(phi4))
            )
        )
        inv_pscale = 1 / np.rad2deg(np.sqrt(area))

        # form equation:
        def f(x: LowLevelArray) -> LowLevelArray:
            w = np.array(self.wcs.evaluate(*(x.T), with_bounding_box=False)).T
            dw = np.mod(np.subtract(w, world) - 180.0, 360.0) - 180.0
            # This will return another low-level array
            return np.add(inv_pscale * dw, x)  # type: ignore[no-any-return]

        def froot(x):
            return (
                np.mod(
                    np.subtract(self.wcs.evaluate(*x, with_bounding_box=False), worldi)
                    - 180.0,
                    360.0,
                )
                - 180.0
            )

        # compute correction:
        def correction(pix: LowLevelArray) -> LowLevelArray:
            p1 = f(pix)
            p2 = f(p1)
            # This will work and return another low-level array
            d = p2 - 2.0 * p1 + pix  # type: ignore[operator]
            idx = np.where(d != 0)
            corr: LowLevelArray = pix - p2  # type: ignore[operator]
            corr[idx] = np.square(p1[idx] - pix[idx]) / d[idx]  # type: ignore[operator]
            return corr

        # initial iteration:
        dpix = correction(pix)

        # Update initial solution:
        pix = pix - dpix  # type: ignore[operator]

        # Norm (L2) squared of the correction:
        dn: LowLevelArray = np.sum(dpix * dpix, axis=1)  # type: ignore[operator]
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
                dn = np.sum(dpix * dpix, axis=1)  # type: ignore[operator]

                # Check for divergence (we do this in two stages
                # to optimize performance for the most common
                # scenario when successive approximations converge):

                if detect_divergence:
                    divergent = dn >= dnprev  # type: ignore[operator]
                    if np.any(divergent):
                        # Find solutions that have not yet converged:
                        slowconv = dn >= tol2
                        (inddiv,) = np.where(divergent & slowconv)

                        if inddiv.shape[0] > 0:
                            # Update indices of elements that
                            # still need correction:
                            conv = dn < dnprev  # type: ignore[operator]
                            iconv = np.where(conv)

                            # Apply correction:
                            dpixgood = dpix[iconv]
                            pix[iconv] = pix[iconv] - dpixgood  # type: ignore[operator]
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
                pix = pix - dpix  # type: ignore[operator]
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
                    pix[iiconv] = pix[iiconv] - dpixgood  # type: ignore[operator]
                    dpix[iiconv] = dpixgood

                    # Find indices of solutions that have not yet
                    # converged to the requested accuracy
                    # AND that do not diverge:
                    (subind,) = np.where((dnnew >= tol2) & conv)

                else:
                    # Apply correction:
                    pix[ind] = pix[ind] - dpixnew  # type: ignore[operator]
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
        (inddiv,) = np.where(((dn >= tol2) & (dn >= dnprev)) | invalid)  # type: ignore[operator]
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
            (ind,) = np.where((dn >= tol2) & (dn < dnprev) & (~invalid))  # type: ignore[operator]
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

        if with_bounding_box and self.wcs.bounding_box is not None:
            # find points outside the bounding box and replace their values
            # with fill_value
            valid = np.logical_not(invalid)
            in_bb = np.ones_like(invalid, dtype=np.bool_)

            for c, (x1, x2) in zip(pix[valid].T, self.wcs.bounding_box, strict=False):
                in_bb[valid] &= (c >= x1) & (c <= x2)
            pix[np.logical_not(in_bb)] = fill_value

        return pix

    def _calc_approx_inv(
        self,
        max_inv_pix_error: float = 5,
        inv_degree: int | None = None,
        npoints: int = 16,
    ) -> None:
        """
        Compute polynomial fit for the inverse transformation to be used as
        initial approximation/guess for the iterative solution.
        """
        self.approx_inverse = None

        try:
            # try to use analytic inverse if available:
            self.approx_inverse = partial(
                self.wcs.backward_transform, with_bounding_box=False
            )
        except (NotImplementedError, KeyError):
            pass
        else:
            return

        if not isinstance(self.wcs.output_frame, CelestialFrame):
            # The _calc_approx_inv method only works with celestial frame transforms
            return

        # Determine reference points.
        if self.wcs.bounding_box is None:
            # A bounding_box is needed to proceed.
            return

        crpix = np.mean(self.wcs.bounding_box, axis=1)

        crval1, crval2 = self.wcs.forward_transform(*crpix)

        # Rotate to native system and deproject. Set center of the projection
        # transformation to the middle of the bounding box ("image") in order
        # to minimize projection effects across the entire image,
        # thus the initial shift.
        sky2pix_proj = Sky2Pix_TAN()

        for transform in self.wcs.forward_transform:
            if isinstance(transform, Projection):
                sky2pix_proj = transform
                break
        if sky2pix_proj.__name__.startswith("Pix2Sky"):
            sky2pix_proj = sky2pix_proj.inverse
        lon_pole = _compute_lon_pole((crval1, crval2), sky2pix_proj)
        ntransform = (
            (Shift(crpix[0]) & Shift(crpix[1]))
            | self.wcs.forward_transform
            | RotateCelestial2Native(crval1, crval2, lon_pole)
            | sky2pix_proj()
        )

        # standard sampling:
        u, v = make_sampling_grid(npoints, self.wcs.bounding_box, crpix=crpix)
        undist_x, undist_y = ntransform(u, v)

        # Double sampling to check if sampling is sufficient.
        ud, vd = make_sampling_grid(2 * npoints, self.wcs.bounding_box, crpix=crpix)
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

        self.approx_inverse = (
            RotateCelestial2Native(crval1, crval2, lon_pole)
            | sky2pix_proj
            | Mapping((0, 1, 0, 1))
            | (fit_inv_poly_u & fit_inv_poly_v)
            | (Shift(crpix[0]) & Shift(crpix[1]))
        )
