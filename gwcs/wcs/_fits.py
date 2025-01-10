import sys
import warnings
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.modeling import projections
from astropy.modeling.bounding_box import ModelBoundingBox as Bbox
from astropy.modeling.models import (
    Mapping,
    Pix2SkyProjection,
    RotateCelestial2Native,
    Shift,
)
from astropy.wcs.utils import celestial_frame_to_wcs, proj_plane_pixel_scales

from gwcs._typing import BoundingBoxTuple, Real
from gwcs.api import BaseGwcs
from gwcs.coordinate_frames import (
    BaseCoordinateFrame,
    CelestialFrame,
    CompositeFrame,
    get_ctype_from_ucd,
)

from ._utils import (
    fit_2D_poly,
    fix_transform_inputs,
    make_sampling_grid,
    reform_poly_coefficients,
    store_2D_coefficients,
)

__all__ = ["FitsMixin"]


@dataclass
class _WorldAxisInfo:
    """
    A class for holding information about a world axis from an output frame.

    Parameters
    ----------
    axis
        Output axis number [in the forward transformation].
    frame
        Coordinate frame to which this axis belongs.
    world_axis_order
        Index of this axis in `gwcs.WCS.output_frame.axes_order`
    cunit
        Axis unit using FITS conversion (``CUNIT``).
    ctype
        Axis FITS type (``CTYPE``).
    input_axes : tuple of int
        Tuple of input axis indices contributing to this world axis.

    """

    axis: int
    frame: BaseCoordinateFrame
    world_axis_order: int
    cunit: str
    ctype: str
    input_axes: tuple[int, ...]


class FitsMixin(BaseGwcs):
    """
    This mixin class provides the functionality to write the WCS to fits
    """

    def to_fits_sip(
        self,
        bounding_box: BoundingBoxTuple | None = None,
        max_pix_error: Real = 0.25,
        degree: int | list[int] | None = None,
        max_inv_pix_error: Real = 0.25,
        inv_degree: int | list[int] | None = None,
        npoints: int = 32,
        crpix: list[Real] | None = None,
        projection: str | Pix2SkyProjection = "TAN",
        verbose: bool = False,
    ) -> fits.Header:
        """
        Construct a SIP-based approximation to the WCS for the axes
        corresponding to the `~gwcs.coordinate_frames.CelestialFrame`
        in the form of a FITS header.

        The default mode in using this attempts to achieve roughly 0.25 pixel
        accuracy over the whole image.

        Parameters
        ----------
        bounding_box
            A pair of tuples, each consisting of two numbers
            Represents the range of pixel values in both dimensions
            ((xmin, xmax), (ymin, ymax))

        max_pix_error
            Maximum allowed error over the domain of the pixel array. This
            error is the equivalent pixel error that corresponds to the maximum
            error in the output coordinate resulting from the fit based on
            a nominal plate scale. Ignored when ``degree`` is an integer or
            a list with a single degree.

        degree
            Degree of the SIP polynomial. Default value `None` indicates that
            all allowed degree values (``[1...9]``) will be considered and
            the lowest degree that meets accuracy requerements set by
            ``max_pix_error`` will be returned. Alternatively, ``degree`` can be
            an iterable containing allowed values for the SIP polynomial degree.
            This option is similar to default `None` but it allows caller to
            restrict the range of allowed SIP degrees used for fitting.
            Finally, ``degree`` can be an integer indicating the exact SIP degree
            to be fit to the WCS transformation. In this case
            ``max_pixel_error`` is ignored.

        max_inv_pix_error
            Maximum allowed inverse error over the domain of the pixel array
            in pixel units. If None, no inverse is generated. Ignored when
            ``degree`` is an integer or a list with a single degree.

        inv_degree
            Degree of the SIP polynomial. Default value `None` indicates that
            all allowed degree values (``[1...9]``) will be considered and
            the lowest degree that meets accuracy requerements set by
            ``max_pix_error`` will be returned. Alternatively, ``degree`` can be
            an iterable containing allowed values for the SIP polynomial degree.
            This option is similar to default `None` but it allows caller to
            restrict the range of allowed SIP degrees used for fitting.
            Finally, ``degree`` can be an integer indicating the exact SIP degree
            to be fit to the WCS transformation. In this case
            ``max_inv_pixel_error`` is ignored.

        npoints
            The number of points in each dimension to sample the bounding box
            for use in the SIP fit. Minimum number of points is 3.

        crpix
            Coordinates (1-based) of the reference point for the new FITS WCS.
            When not provided, i.e., when set to `None` (default) the reference
            pixel will be chosen near the center of the bounding box for axes
            corresponding to the celestial frame.

        projection
            Projection to be used for the created FITS WCS. It can be specified
            as a string of three characters specifying a FITS projection code
            from Table 13 in
            `Representations of World Coordinates in FITS \
            <https://doi.org/10.1051/0004-6361:20021326>`_
            (Paper I), Greisen, E. W., and Calabretta, M. R., A & A, 395,
            1061-1075, 2002. Alternatively, it can be an instance of one of the
            `astropy's Pix2Sky_* <https://docs.astropy.org/en/stable/modeling/\
            reference_api.html#module-astropy.modeling.projections>`_
            projection models inherited from
            :py:class:`~astropy.modeling.projections.Pix2SkyProjection`.

        verbose
            Print progress of fits.

        Returns
        -------
          FITS header with all SIP WCS keywords

        Raises
        ------
        ValueError
            If the WCS is not at least 2D, an exception will be raised. If the
            specified accuracy (both forward and inverse, both rms and maximum)
            is not achieved an exception will be raised.

        Notes
        -----

        Use of this requires a judicious choice of required accuracies.
        Attempts to use higher degrees (~7 or higher) will typically fail due
        to floating point problems that arise with high powers.

        """
        _, _, celestial_group = self._separable_groups(detect_celestial=True)
        if celestial_group is None:
            msg = "The to_fits_sip requires an output celestial frame."
            raise ValueError(msg)

        return self._to_fits_sip(
            celestial_group=celestial_group,
            keep_axis_position=False,
            bounding_box=bounding_box,
            max_pix_error=max_pix_error,
            degree=degree,
            max_inv_pix_error=max_inv_pix_error,
            inv_degree=inv_degree,
            npoints=npoints,
            crpix=crpix,
            projection=projection,
            matrix_type="CD",
            verbose=verbose,
        )

    def _to_fits_sip(
        self,
        celestial_group: list[_WorldAxisInfo],
        keep_axis_position: bool,
        bounding_box: BoundingBoxTuple | None,
        max_pix_error: Real,
        degree: int | list[int] | None,
        max_inv_pix_error: Real,
        inv_degree: int | list[int] | None,
        npoints: int,
        crpix: list[Real] | None,
        projection: str | Pix2SkyProjection,
        matrix_type: str,
        verbose: bool,
    ) -> fits.Header:
        r"""
        Construct a SIP-based approximation to the WCS for the axes
        corresponding to the `~gwcs.coordinate_frames.CelestialFrame`
        in the form of a FITS header.

        The default mode in using this attempts to achieve roughly 0.25 pixel
        accuracy over the whole image.

        Parameters
        ----------
        celestial_group
            A group of two celestial axes to be represented using standard
            image FITS WCS and maybe ``-SIP`` polynomials.

        keep_axis_position
            This parameter controls whether to keep/preserve output axes
            indices in this WCS object when creating FITS WCS and create a FITS
            header with ``CTYPE`` axes indices preserved from the ``frame``
            object or whether to reset the indices of output celestial axes
            to 1 and 2 with ``CTYPE1``, ``CTYPE2``. Default is `False`.

            .. warning::
                Returned header will have both ``NAXIS`` and ``WCSAXES`` set
                to 2. If ``max(axes_mapping) > 2`` this will lead to an invalid
                WCS. It is caller's responsibility to adjust NAXIS to a valid
                value.

            .. note::
                The ``lon``/``lat`` order is still preserved regardless of this
                setting.

        bounding_box
            A pair of tuples, each consisting of two numbers
            Represents the range of pixel values in both dimensions
            ((xmin, xmax), (ymin, ymax))

        max_pix_error
            Maximum allowed error over the domain of the pixel array. This
            error is the equivalent pixel error that corresponds to the maximum
            error in the output coordinate resulting from the fit based on
            a nominal plate scale. Ignored when ``degree`` is an integer or
            a list with a single degree.

        degree
            Degree of the SIP polynomial. Default value `None` indicates that
            all allowed degree values (``[1...9]``) will be considered and
            the lowest degree that meets accuracy requerements set by
            ``max_pix_error`` will be returned. Alternatively, ``degree`` can be
            an iterable containing allowed values for the SIP polynomial degree.
            This option is similar to default `None` but it allows caller to
            restrict the range of allowed SIP degrees used for fitting.
            Finally, ``degree`` can be an integer indicating the exact SIP degree
            to be fit to the WCS transformation. In this case
            ``max_pixel_error`` is ignored.

        max_inv_pix_error
            Maximum allowed inverse error over the domain of the pixel array
            in pixel units. If None, no inverse is generated. Ignored when
            ``degree`` is an integer or a list with a single degree.

        inv_degree
            Degree of the SIP polynomial. Default value `None` indicates that
            all allowed degree values (``[1...9]``) will be considered and
            the lowest degree that meets accuracy requerements set by
            ``max_pix_error`` will be returned. Alternatively, ``degree`` can be
            an iterable containing allowed values for the SIP polynomial degree.
            This option is similar to default `None` but it allows caller to
            restrict the range of allowed SIP degrees used for fitting.
            Finally, ``degree`` can be an integer indicating the exact SIP degree
            to be fit to the WCS transformation. In this case
            ``max_inv_pixel_error`` is ignored.

        npoints
            The number of points in each dimension to sample the bounding box
            for use in the SIP fit. Minimum number of points is 3.

        crpix
            Coordinates (1-based) of the reference point for the new FITS WCS.
            When not provided, i.e., when set to `None` (default) the reference
            pixel will be chosen near the center of the bounding box for axes
            corresponding to the celestial frame.

        projection
            Projection to be used for the created FITS WCS. It can be specified
            as a string of three characters specifying a FITS projection code
            from Table 13 in
            `Representations of World Coordinates in FITS \
            <https://doi.org/10.1051/0004-6361:20021326>`_
            (Paper I), Greisen, E. W., and Calabretta, M. R., A & A, 395,
            1061-1075, 2002. Alternatively, it can be an instance of one of the
            `astropy's Pix2Sky_* <https://docs.astropy.org/en/stable/modeling/\
            reference_api.html#module-astropy.modeling.projections>`_
            projection models inherited from
            :py:class:`~astropy.modeling.projections.Pix2SkyProjection`.

        matrix_type
            Specifies formalism (``PC`` or ``CD``) to be used for the linear
            transformation matrix and normalization for the ``PC`` matrix
            *when non-linear polynomial terms are not required to achieve
            requested accuracy*.

            .. note:: ``CD`` matrix is always used when requested SIP
                approximation accuracy requires non-linear terms (when
                ``CTYPE`` ends in ``-SIP``). This parameter is ignored when
                non-linear polynomial terms are used.

            - ``'CD'``: use ``CD`` matrix;

            - ``'PC-CDELT1'``: set ``PC=CD`` and ``CDELTi=1``. This is the
              behavior of `~astropy.wcs.WCS.to_header` method;

            - ``'PC-SUM1'``: normalize ``PC`` matrix such that sum
              of its squared elements is 1: :math:`\Sigma PC_{ij}^2=1`;

            - ``'PC-DET1'``: normalize ``PC`` matrix such that :math:`|\det(PC)|=1`;

            - ``'PC-SCALE'``: normalize ``PC`` matrix such that ``CDELTi``
              are estimates of the linear pixel scales.

        verbose
            Print progress of fits.

        Returns
        -------
        FITS header with all SIP WCS keywords

        Raises
        ------
        ValueError
            If the WCS is not at least 2D, an exception will be raised. If the
            specified accuracy (both forward and inverse, both rms and maximum)
            is not achieved an exception will be raised.

        """
        if isinstance(matrix_type, str):
            matrix_type = matrix_type.upper()

        if matrix_type not in ["CD", "PC-CDELT1", "PC-SUM1", "PC-DET1", "PC-SCALE"]:
            msg = f"Unsupported 'matrix_type' value: {matrix_type!r}."
            raise ValueError(msg)

        if npoints < 8:
            msg = "Number of sampling points is too small. 'npoints' must be >= 8."
            raise ValueError(msg)

        if isinstance(projection, str):
            projection = projection.upper()
            try:
                sky2pix_proj = getattr(projections, f"Sky2Pix_{projection}")(
                    name=projection
                )
            except AttributeError as err:
                msg = f"Unsupported FITS WCS sky projection: {projection}"
                raise ValueError(msg) from err

        elif isinstance(projection, projections.Sky2PixProjection):
            sky2pix_proj = projection
            projection = projection.name
            if (
                not projection
                or not isinstance(projection, str)
                or len(projection) != 3
            ):
                msg = f"Unsupported FITS WCS sky projection: {sky2pix_proj}"
                raise ValueError(msg)
            try:
                getattr(projections, f"Sky2Pix_{projection}")()
            except AttributeError as err:
                msg = f"Unsupported FITS WCS sky projection: {projection}"
                raise ValueError(msg) from err

        else:
            msg = (
                "'projection' must be either a FITS WCS string projection code "
                "or an instance of astropy.modeling.projections.Pix2SkyProjection."
            )
            raise TypeError(msg)

        frame = celestial_group[0].frame

        lon_axis = frame.axes_order[0]
        lat_axis = frame.axes_order[1]

        # identify input axes:
        input_axes = []
        for wax in celestial_group:
            input_axes.extend(wax.input_axes)
        input_axes = sorted(set(input_axes))

        if len(input_axes) != 2:
            msg = (
                "Only CelestialFrame that correspond to two "
                "input axes are supported."
            )
            raise ValueError(msg)

        # Axis number for FITS axes.
        # iax? - image axes; nlon, nlat - celestial axes:
        if keep_axis_position:
            nlon = lon_axis + 1
            nlat = lat_axis + 1
            iax1, iax2 = (i + 1 for i in input_axes)
        else:
            nlon, nlat = (1, 2) if lon_axis < lat_axis else (2, 1)
            iax1 = 1
            iax2 = 2

        # Determine reference points.
        if bounding_box is None and self.bounding_box is None:
            msg = "A bounding_box is needed to proceed."
            raise ValueError(msg)
        if bounding_box is None:
            bounding_box = self.bounding_box

        bb_center = np.mean(bounding_box, axis=1)

        fixi_dict = {
            k: bb_center[k] for k in set(range(self.pixel_n_dim)).difference(input_axes)
        }

        # Once that bug is fixed, the code below can be replaced with fix_inputs
        # statement commented out immediately above.
        transform = fix_transform_inputs(self.forward_transform, fixi_dict)

        transform = transform | Mapping(
            (lon_axis, lat_axis), n_inputs=self.forward_transform.n_outputs
        )

        (xmin, xmax) = bounding_box[input_axes[0]]
        (ymin, ymax) = bounding_box[input_axes[1]]

        # 0-based crpix:
        if crpix is None:
            crpix1 = round(bb_center[input_axes[0]], 1)
            crpix2 = round(bb_center[input_axes[1]], 1)
        else:
            crpix1 = crpix[0] - 1
            crpix2 = crpix[1] - 1

        # check that the bounding box has some reasonable size:
        if (xmax - xmin) < 1 or (ymax - ymin) < 1:
            msg = "Bounding box is too small for fitting a SIP polynomial"
            raise ValueError(msg)

        lon, lat = transform(crpix1, crpix2)

        # Now rotate to native system and deproject. Recall that transform
        # expects pixels in the original coordinate system, but the SIP
        # transform is relative to crpix coordinates, thus the initial shift.
        ntransform = (
            (Shift(crpix1) & Shift(crpix2))
            | transform
            | RotateCelestial2Native(lon, lat, 180)
            | sky2pix_proj
        )

        # standard sampling:
        u, v = make_sampling_grid(
            npoints, tuple(bounding_box[k] for k in input_axes), crpix=[crpix1, crpix2]
        )
        undist_x, undist_y = ntransform(u, v)

        # Double sampling to check if sampling is sufficient.
        ud, vd = make_sampling_grid(
            2 * npoints,
            tuple(bounding_box[k] for k in input_axes),
            crpix=[crpix1, crpix2],
        )
        undist_xd, undist_yd = ntransform(ud, vd)

        # Determine approximate pixel scale in order to compute error threshold
        # from the specified pixel error. Computed at the center of the array.
        x0, y0 = ntransform(0, 0)
        xx, xy = ntransform(1, 0)
        yx, yy = ntransform(0, 1)
        pixarea = np.abs((xx - x0) * (yy - y0) - (xy - y0) * (yx - x0))
        plate_scale = np.sqrt(pixarea)

        # The fitting section.
        if verbose:
            sys.stdout.write("\nFitting forward SIP ...")
        fit_poly_x, fit_poly_y, max_resid = fit_2D_poly(
            degree,
            max_pix_error,
            plate_scale,
            u,
            v,
            undist_x,
            undist_y,
            ud,
            vd,
            undist_xd,
            undist_yd,
            verbose=verbose,
        )

        # The following is necessary to put the fit into the SIP formalism.
        cdmat, sip_poly_x, sip_poly_y = reform_poly_coefficients(fit_poly_x, fit_poly_y)
        # cdmat = np.array([[fit_poly_x.c1_0.value, fit_poly_x.c0_1.value],
        #                   [fit_poly_y.c1_0.value, fit_poly_y.c0_1.value]])
        det = cdmat[0][0] * cdmat[1][1] - cdmat[0][1] * cdmat[1][0]
        U = (cdmat[1][1] * undist_x - cdmat[0][1] * undist_y) / det
        V = (-cdmat[1][0] * undist_x + cdmat[0][0] * undist_y) / det
        detd = cdmat[0][0] * cdmat[1][1] - cdmat[0][1] * cdmat[1][0]
        Ud = (cdmat[1][1] * undist_xd - cdmat[0][1] * undist_yd) / detd
        Vd = (-cdmat[1][0] * undist_xd + cdmat[0][0] * undist_yd) / detd

        if max_inv_pix_error:
            if verbose:
                sys.stdout.write("\nFitting inverse SIP ...")
            fit_inv_poly_u, fit_inv_poly_v, max_inv_resid = fit_2D_poly(
                inv_degree,
                max_inv_pix_error,
                1,
                U,
                V,
                u - U,
                v - V,
                Ud,
                Vd,
                ud - Ud,
                vd - Vd,
                verbose=verbose,
            )

        # create header with WCS info:
        w = celestial_frame_to_wcs(frame.reference_frame, projection=projection)
        w.wcs.crval = [lon, lat]
        w.wcs.crpix = [crpix1 + 1, crpix2 + 1]
        w.wcs.pc = cdmat if nlon < nlat else cdmat[::-1]
        w.wcs.set()
        hdr = w.to_header(True)

        # data array info:
        hdr.insert(0, ("NAXIS", 2, "number of array dimensions"))
        hdr.insert(1, (f"NAXIS{iax1:d}", int(xmax) + 1))
        hdr.insert(2, (f"NAXIS{iax2:d}", int(ymax) + 1))
        if len(hdr["NAXIS*"]) != 3:
            msg = "NAXIS* should have 3 axes"
            raise ValueError(msg)

        # list of celestial axes related keywords:
        cel_kwd = ["CRVAL", "CTYPE", "CUNIT"]

        # Add SIP info:
        if fit_poly_x.degree > 1:
            mat_kind = "CD"
            # CDELT is not used with CD matrix (PC->CD later):
            del hdr["CDELT?"]

            hdr["CTYPE1"] = hdr["CTYPE1"].strip() + "-SIP"
            hdr["CTYPE2"] = hdr["CTYPE2"].strip() + "-SIP"
            hdr["A_ORDER"] = fit_poly_x.degree
            hdr["B_ORDER"] = fit_poly_x.degree
            store_2D_coefficients(hdr, sip_poly_x, "A")
            store_2D_coefficients(hdr, sip_poly_y, "B")
            hdr["sipmxerr"] = (max_resid, "Max diff from GWCS (equiv pix).")

            if max_inv_pix_error:
                hdr["AP_ORDER"] = fit_inv_poly_u.degree
                hdr["BP_ORDER"] = fit_inv_poly_u.degree
                store_2D_coefficients(hdr, fit_inv_poly_u, "AP", keeplinear=True)
                store_2D_coefficients(hdr, fit_inv_poly_v, "BP", keeplinear=True)
                hdr["sipiverr"] = (max_inv_resid, "Max diff for inverse (pixels)")

        else:
            if matrix_type.startswith("PC"):
                mat_kind = "PC"
                cel_kwd.append("CDELT")

                if matrix_type == "PC-CDELT1":
                    cdelt = [1.0, 1.0]

                elif matrix_type == "PC-SUM1":
                    norm = np.sqrt(np.sum(w.wcs.pc**2))
                    cdelt = [norm, norm]

                elif matrix_type == "PC-DET1":
                    det_pc = np.linalg.det(w.wcs.pc)
                    norm = np.sqrt(np.abs(det_pc))
                    cdelt = [norm, np.sign(det_pc) * norm]

                elif matrix_type == "PC-SCALE":
                    cdelt = proj_plane_pixel_scales(w)

                for i in range(1, 3):
                    s = cdelt[i - 1]
                    hdr[f"CDELT{i}"] = s
                    for j in range(1, 3):
                        pc_kwd = f"PC{i}_{j}"
                        if pc_kwd in hdr:
                            hdr[pc_kwd] = w.wcs.pc[i - 1, j - 1] / s

            else:
                mat_kind = "CD"
                del hdr["CDELT?"]

            hdr["sipmxerr"] = (max_resid, "Max diff from GWCS (equiv pix).")

        # Construct CD matrix while remapping input axes.
        # We do not update comments to typical comments for CD matrix elements
        # (such as 'partial of second axis coordinate w.r.t. y'), because
        # when input frame has number of axes > 2, then imaging
        # axes arbitrary.
        old_nlon, old_nlat = (1, 2) if nlon < nlat else (2, 1)

        # Remap input axes (CRPIX) and output axes-related parameters
        # (CRVAL, CUNIT, CTYPE, CD/PC). This has to be done in two steps to avoid
        # name conflicts (i.e., swapping CRPIX1<->CRPIX2).

        # remap input axes:
        axis_rename = {}
        if iax1 != 1:
            axis_rename["CRPIX1"] = f"CRPIX{iax1}"
        if iax2 != 2:
            axis_rename["CRPIX2"] = f"CRPIX{iax2}"

        # CP/PC matrix:
        axis_rename[f"PC{old_nlon}_1"] = f"{mat_kind}{nlon}_{iax1}"
        axis_rename[f"PC{old_nlon}_2"] = f"{mat_kind}{nlon}_{iax2}"
        axis_rename[f"PC{old_nlat}_1"] = f"{mat_kind}{nlat}_{iax1}"
        axis_rename[f"PC{old_nlat}_2"] = f"{mat_kind}{nlat}_{iax2}"

        # remap celestial axes keywords:
        for kwd in cel_kwd:
            for iold, inew in [(1, nlon), (2, nlat)]:
                if iold != inew:
                    axis_rename[f"{kwd:s}{iold:d}"] = f"{kwd:s}{inew:d}"

        # construct new header cards with remapped axes:
        new_cards = [
            fits.Card(keyword=axis_rename[c.keyword], value=c.value, comment=c.comment)
            if c[0] in axis_rename
            else c
            for c in hdr.cards
        ]

        hdr = fits.Header(new_cards)
        hdr["WCSAXES"] = 2
        hdr.insert("WCSAXES", ("WCSNAME", f"{self.output_frame.name}"), after=True)

        # for PC matrix formalism, set diagonal elements to 0 if necessary
        # (by default, in PC formalism, diagonal matrix elements by default
        # are 0):
        if mat_kind == "PC":
            if nlon not in [iax1, iax2]:
                hdr.insert(
                    f"{mat_kind}{nlon}_{iax2}",
                    (
                        f"{mat_kind}{nlon}_{nlon}",
                        0.0,
                        "Coordinate transformation matrix element",
                    ),
                )
            if nlat not in [iax1, iax2]:
                hdr.insert(
                    f"{mat_kind}{nlat}_{iax2}",
                    (
                        f"{mat_kind}{nlat}_{nlat}",
                        0.0,
                        "Coordinate transformation matrix element",
                    ),
                )

        return hdr

    def _separable_groups(
        self, detect_celestial: bool = False
    ) -> tuple[
        list[list[_WorldAxisInfo]], list[_WorldAxisInfo], list[_WorldAxisInfo] | None
    ]:
        """
        This method finds sets (groups) of separable axes - axes that are
        dependent on other axes within a set/group but do not depend on
        axes from other groups. In other words, axes from different
        groups are separable.

        Parameters
        ----------
        detect_celestial
            If `True`, will return, as the third return value, the group of
            celestial axes separately from all other (groups of) axes. If
            no celestial frame is detected, then return value for the
            celestial axes group will be set to `None`.

        Returns
        -------
            List of one of the following:
            - Each inner list represents a group of non-separable (among
              themselves) axes and each axis in a group is independent of axes
              in *other* groups. Each axis in a group is represented through
              the `_WorldAxisInfo` class used to store relevant information about
              an axis. When ``detect_celestial`` is set to `True`, celestial axes
              group is not included in this list.

            - A flattened version of ``axes_groups``. Even though it is not
              difficult to flatten ``axes_groups``, this list is a by-product
              of other checks and returned here for efficiency. When
              ``detect_celestial`` is set to `True`, celestial axes
              group is not included in this list.

            - A group of two celestial axes. This group is returned *only when*
              ``detect_celestial`` is set to `True`.
        """

        def find_frame(axis_number):
            for frame in frames:
                if axis_number in frame.axes_order:
                    return frame
            msg = (
                "Encountered an output axes that does not "
                "belong to any output coordinate frames."
            )
            raise RuntimeError(msg)

        # use correlation matrix to find separable axes:
        corr_mat = self.axis_correlation_matrix
        axes_sets = [set(np.flatnonzero(r)) for r in corr_mat.T]

        k = 0
        while len(axes_sets) - 1 > k:
            for m in range(len(axes_sets) - 1, k, -1):
                if axes_sets[k].isdisjoint(axes_sets[m]):
                    continue
                axes_sets[k] = axes_sets[k].union(axes_sets[m])
                del axes_sets[m]
            k += 1

        # create a mapping of output axes to input/image axes groups:
        mapping = {k: tuple(np.flatnonzero(r)) for k, r in enumerate(corr_mat)}

        axes_groups = []
        world_axes = []  # flattened version of axes_groups
        input_axes = []  # all input axes

        if isinstance(self.output_frame, CompositeFrame):
            frames = self.output_frame.frames
        else:
            frames = [self.output_frame]

        celestial_group = None

        # identify which separable group of axes belong
        for s in axes_sets:
            axis_info_group = []  # group of separable output axes info

            # Find the frame to which the first axis in the group belongs.
            # Most likely this frame will be the frame of all other axes in
            # this group; if not, we will update it later.
            axis = sorted(s)
            frame = find_frame(axis[0])

            celestial = (
                detect_celestial
                and len(axis) == 2
                and len(frame.axes_order) == 2
                and isinstance(frame, CelestialFrame)
            )

            for axno in axis:
                if axno not in frame.axes_order:
                    frame = find_frame(axno)
                    celestial = False  # Celestial axes must belong to the same frame

                # index of the axis in this frame's
                fidx = frame.axes_order.index(axno)

                axis_info = _WorldAxisInfo(
                    axis=axno,
                    frame=frame,
                    world_axis_order=self.output_frame.axes_order.index(axno),
                    cunit=frame.unit[fidx].to_string("fits", fraction=True).upper(),
                    ctype=get_ctype_from_ucd(self.world_axis_physical_types[axno]),
                    input_axes=mapping[axno],
                )
                axis_info_group.append(axis_info)
                input_axes.extend(mapping[axno])

            world_axes.extend(axis_info_group)
            if celestial:
                celestial_group = axis_info_group
            else:
                axes_groups.append(axis_info_group)

        # sanity check:
        input_axes = set(
            sum(
                (ax.input_axes for ax in world_axes),
                world_axes[0].input_axes.__class__(),
            )
        )
        n_inputs = len(input_axes)

        if (
            n_inputs != self.pixel_n_dim
            or max(input_axes) + 1 != n_inputs
            or min(input_axes) < 0
        ):
            msg = (
                "Input axes indices are inconsistent with the "
                "forward transformation."
            )
            raise ValueError(msg)

        if detect_celestial:
            return axes_groups, world_axes, celestial_group
        return axes_groups, world_axes

    def to_fits_tab(
        self,
        bounding_box: BoundingBoxTuple | None = None,
        bin_ext_name: str = "WCS-TABLE",
        coord_col_name: str = "coordinates",
        sampling: Real | tuple[Real, ...] = 1,
    ) -> tuple[fits.Header, fits.BinTableHDU]:
        """
        Construct a FITS WCS ``-TAB``-based approximation to the WCS
        in the form of a FITS header and a binary table extension. For the
        description of the FITS WCS ``-TAB`` convention, see
        "Representations of spectral coordinates in FITS" in
        `Greisen, E. W. et al. A&A 446 (2) 747-771 (2006)
        <https://doi.org/10.1051/0004-6361:20053818>`_ .

        Parameters
        ----------
        bounding_box
            Specifies the range of acceptable values for each input axis.
            The order of the axes is
            `~gwcs.coordinate_frames.BaseCoordinateFrame.axes_order`.
            For two image axes ``bounding_box`` is of the form
            ``((xmin, xmax), (ymin, ymax))``.

        bin_ext_name
            Extension name for the `~astropy.io.fits.BinTableHDU` HDU for those
            axes groups that will be converted using FITW WCS' ``-TAB``
            algorithm. Extension version will be determined automatically
            based on the number of separable group of axes.

        coord_col_name
            Field name of the coordinate array in the structured array
            stored in `~astropy.io.fits.BinTableHDU` data. This corresponds to
            ``TTYPEi`` field in the FITS header of the binary table extension.

        sampling
            The target "density" of grid nodes per pixel to be used when
            creating the coordinate array for the ``-TAB`` FITS WCS convention.
            It is equal to ``1/step`` where ``step`` is the distance between
            grid nodes in pixels. ``sampling`` can be specified as a single
            number to be used for all axes or as a `tuple` of numbers
            that specify the sampling for each image axis.

        Returns
        -------
            Header with WCS-TAB information associated (to be used) with image
            data.

            Binary table extension containing the coordinate array.

        Raises
        ------
        ValueError
            When ``bounding_box`` is not defined either through the input
            ``bounding_box`` parameter or this object's ``bounding_box``
            property.

        ValueError
            When ``sampling`` is a `tuple` of length larger than 1 that
            does not match the number of image axes.

        RuntimeError
            If the number of image axes (``~gwcs.WCS.pixel_n_dim``) is larger
            than the number of world axes (``~gwcs.WCS.world_n_dim``).

        """
        if bounding_box is None:
            if self.bounding_box is None:
                msg = "Need a valid bounding_box to compute the footprint."
                raise ValueError(msg)
            bounding_box = self.bounding_box

        else:
            # validate user-supplied bounding box:
            frames = self.available_frames
            transform_0 = self.get_transform(frames[0], frames[1])
            Bbox.validate(transform_0, bounding_box)

        if self.forward_transform.n_inputs == 1:
            bounding_box = [bounding_box]

        if self.pixel_n_dim > self.world_n_dim:
            msg = (
                "The case when the number of input axes is larger than the "
                "number of output axes is not supported."
            )
            raise RuntimeError(msg)

        try:
            sampling = np.broadcast_to(sampling, (self.pixel_n_dim,))
        except ValueError as err:
            msg = (
                "Number of sampling values either must be 1 "
                "or it must match the number of pixel axes."
            )
            raise ValueError(msg) from err

        _, world_axes = self._separable_groups(detect_celestial=False)

        hdr, bin_table_hdu = self._to_fits_tab(
            hdr=None,
            world_axes_group=world_axes,
            use_cd=False,
            bounding_box=bounding_box,
            bin_ext=bin_ext_name,
            coord_col_name=coord_col_name,
            sampling=sampling,
        )

        return hdr, bin_table_hdu

    def _to_fits_tab(
        self,
        hdr: fits.Header | None,
        world_axes_group: tuple[dict, ...],
        use_cd: bool,
        bounding_box: BoundingBoxTuple | None,
        bin_ext: str | tuple[str | int, ...],
        coord_col_name: str,
        sampling: Real | tuple[Real, ...],
    ) -> tuple[fits.Header, fits.BinTableHDU]:
        """
        Construct a FITS WCS ``-TAB``-based approximation to the WCS
        in the form of a FITS header and a binary table extension. For the
        description of the FITS WCS ``-TAB`` convention, see
        "Representations of spectral coordinates in FITS" in
        `Greisen, E. W. et al. A&A 446 (2) 747-771 (2006)
        <https://doi.org/10.1051/0004-6361:20053818>`_ .


        .. warn::
            For this helper function, parameters ``bounding_box`` and
            ``sampling`` (when provided as a tuple) are expected to have
            the same length as the number of input axes in the *full* WCS
            object. That is, the number of elements in ``bounding_box`` and
            ``sampling`` is not be affected by ``ignore_axes``.

        Parameters
        ----------
        hdr
            The first time this function is called, ``hdr`` should be set to
            `None` or be an empty :py:class:`~astropy.io.fits.Header` object.
            On subsequent calls, updated header from the previous iteration
            should be provided.

        world_axes_group
            A list of world axes to represent through FITS' -TAB convention.
            This is a list of dictionaries with each dicti

        use_cd
            When `True` - CD-matrix formalism will be used instead of the
            PC-matrix formalism.

        bounding_box
            Specifies the range of acceptable values for each input axis.
            The order of the axes is
            `~gwcs.coordinate_frames.BaseCoordinateFrame.axes_order`.
            For two image axes ``bounding_box`` is of the form
            ``((xmin, xmax), (ymin, ymax))``.

        bin_ext
            Extension name  and optionally version for the
            `~astropy.io.fits.BinTableHDU` HDU. When only a string extension
            name is provided, extension version will be set to 1.
            When ``bin_ext`` is a tuple, first element should be extension
            name and the second element is a positive integer extension version
            number.

        coord_col_name
            Field name of the coordinate array in the structured array
            stored in `~astropy.io.fits.BinTableHDU` data. This corresponds to
            ``TTYPEi`` field in the FITS header of the binary table extension.

        sampling
            The target "density" of grid nodes per pixel to be used when
            creating the coordinate array for the ``-TAB`` FITS WCS convention.
            It is equal to ``1/step`` where ``step`` is the distance between
            grid nodes in pixels. ``sampling`` can be specified as a single
            number to be used for all axes or as a `tuple` of numbers
            that specify the sampling for each image axis.

        Returns
        -------
            Header with WCS-TAB information associated (to be used) with image
            data.

            Binary table extension containing the coordinate array.

        Raises
        ------
        ValueError
            When ``bounding_box`` is not defined either through the input
            ``bounding_box`` parameter or this object's ``bounding_box``
            property.

        ValueError
            When ``sampling`` is a `tuple` of length larger than 1 that
            does not match the number of image axes.

        ValueError
            When extension version is smaller than 1.

        TypeError

        RuntimeError
            If the number of image axes (``~gwcs.WCS.pixel_n_dim``) is larger
            than the number of world axes (``~gwcs.WCS.world_n_dim``).

        """
        if isinstance(bin_ext, str):
            bin_ext = (bin_ext, 1)

        if isinstance(bounding_box, Bbox):
            bounding_box = bounding_box.bounding_box(order="F")
        if isinstance(bounding_box, list):
            for index, bbox in enumerate(bounding_box):
                if isinstance(bbox, Bbox):
                    bounding_box[index] = bbox.bounding_box(order="F")

        # identify input axes:
        input_axes = []
        world_axes_idx = []
        for ax in world_axes_group:
            world_axes_idx.append(ax.axis)
            input_axes.extend(ax.input_axes)
        input_axes = sorted(set(input_axes))
        n_inputs = len(input_axes)
        n_outputs = len(world_axes_group)
        world_axes_idx.sort()

        # Create initial header and deal with non-degenerate axes
        if hdr is None:
            hdr = fits.Header()
            hdr["NAXIS"] = n_inputs, "number of array dimensions"
            hdr["WCSAXES"] = n_outputs
            hdr.insert("WCSAXES", ("WCSNAME", f"{self.output_frame.name}"), after=True)

        else:
            hdr["NAXIS"] += n_inputs
            hdr["WCSAXES"] += n_outputs

        # see what axes have been already populated in the header:
        used_hdr_axes = []
        for v in hdr["naxis*"]:
            value = v.split("NAXIS")[1]
            if not value:
                continue

            used_hdr_axes.append(int(value) - 1)

        degenerate_axis_start = max(
            self.pixel_n_dim + 1, max(used_hdr_axes) + 1 if used_hdr_axes else 1
        )

        # Deal with non-degenerate axes and add NAXISi to the header:
        offset = hdr.index("NAXIS")

        for iax in input_axes:
            iiax = int(np.searchsorted(used_hdr_axes, iax))
            hdr.insert(
                iiax + offset + 1,
                (f"NAXIS{iax + 1:d}", int(max(bounding_box[iiax])) + 1),
            )

        # 1D grid coordinates:
        gcrds = []
        cdelt = []
        bb = [bounding_box[k] for k in input_axes]
        for (xmin, xmax), s in zip(bb, sampling, strict=False):
            npix = max(2, 1 + int(np.ceil(abs((xmax - xmin) / s))))
            gcrds.append(np.linspace(xmin, xmax, npix))
            cdelt.append((npix - 1) / (xmax - xmin) if xmin != xmax else 1)

        # In the forward transformation, select only inputs and outputs
        # that we need given world_axes_group parameter:
        bb_center = np.mean(bounding_box, axis=1)

        fixi_dict = {
            k: bb_center[k] for k in set(range(self.pixel_n_dim)).difference(input_axes)
        }

        transform = fix_transform_inputs(self.forward_transform, fixi_dict)
        transform = transform | Mapping(
            world_axes_idx, n_inputs=self.forward_transform.n_outputs
        )

        xyz = np.meshgrid(*gcrds[::-1], indexing="ij")[::-1]

        shape = xyz[0].shape
        xyz = [v.ravel() for v in xyz]

        coord = np.stack(transform(*xyz), axis=-1)

        coord = coord.reshape(
            (
                *shape,
                len(world_axes_group),
            )
        )

        # create header with WCS info:
        if hdr is None:
            hdr = fits.Header()

        for axis_info in world_axes_group:
            k = axis_info.axis
            widx = world_axes_idx.index(k)
            k1 = k + 1
            ct = get_ctype_from_ucd(self.world_axis_physical_types[k])
            if len(ct) > 4:
                msg = "Axis type name too long."
                raise ValueError(msg)

            hdr[f"CTYPE{k1:d}"] = ct + (4 - len(ct)) * "-" + "-TAB"
            hdr[f"CUNIT{k1:d}"] = self.world_axis_units[k]
            hdr[f"PS{k1:d}_0"] = bin_ext[0]
            hdr[f"PV{k1:d}_1"] = bin_ext[1]
            hdr[f"PS{k1:d}_1"] = coord_col_name
            hdr[f"PV{k1:d}_3"] = widx + 1
            hdr[f"CRVAL{k1:d}"] = 1

            if widx < n_inputs:
                m1 = input_axes[widx] + 1
                hdr[f"CRPIX{m1:d}"] = gcrds[widx][0] + 1
                if use_cd:
                    hdr[f"CD{k1:d}_{m1:d}"] = cdelt[widx]
                else:
                    if k1 != m1:
                        hdr[f"PC{k1:d}_{k1:d}"] = 0.0
                    hdr[f"PC{k1:d}_{m1:d}"] = 1.0
                    hdr[f"CDELT{k1:d}"] = cdelt[widx]
            else:
                m1 = degenerate_axis_start
                degenerate_axis_start += 1

                hdr[f"CRPIX{m1:d}"] = 1
                if use_cd:
                    hdr[f"CD{k1:d}_{m1:d}"] = 1.0
                else:
                    if k1 != m1:
                        hdr[f"PC{k1:d}_{k1:d}"] = 0.0
                    hdr[f"PC{k1:d}_{m1:d}"] = 1.0
                    hdr[f"CDELT{k1:d}"] = 1

                coord = coord[None, :]

        # structured array (data) for binary table HDU:
        arr = np.array(
            [(coord,)],
            dtype=[
                (coord_col_name, np.float64, coord.shape),
            ],
        )

        # create binary table HDU:
        bin_table_hdu = fits.BinTableHDU(arr, name=bin_ext[0], ver=bin_ext[1])

        return hdr, bin_table_hdu

    def to_fits(
        self,
        bounding_box: BoundingBoxTuple | None = None,
        max_pix_error: Real = 0.25,
        degree: int | list[int] | None = None,
        max_inv_pix_error: Real = 0.25,
        inv_degree: int | list[int] | None = None,
        npoints: int = 32,
        crpix: list[Real] | None = None,
        projection: str | Pix2SkyProjection = "TAN",
        bin_ext_name: str = "WCS-TABLE",
        coord_col_name: str = "coordinates",
        sampling: Real | tuple[Real, ...] = 1,
        verbose: bool = False,
    ) -> tuple[fits.Header, fits.BinTableHDU]:
        """
        Construct a FITS WCS ``-TAB``-based approximation to the WCS
        in the form of a FITS header and a binary table extension. For the
        description of the FITS WCS ``-TAB`` convention, see
        "Representations of spectral coordinates in FITS" in
        `Greisen, E. W. et al. A&A 446 (2) 747-771 (2006)
        <https://doi.org/10.1051/0004-6361:20053818>`_ . If WCS contains
        celestial frame, PC/CD formalism will be used for the celestial axes.

        .. note::
            SIP distortion fitting requires that the WCS object has only two
            celestial axes. When WCS does not contain celestial axes,
            SIP fitting parameters (``max_pix_error``, ``degree``,
            ``max_inv_pix_error``, ``inv_degree``, and ``projection``)
            are ignored. When a WCS, in addition to celestial
            frame, contains other types of axes, SIP distortion fitting is
            disabled (only linear terms are fitted for celestial frame).

        Parameters
        ----------
        bounding_box
            Specifies the range of acceptable values for each input axis.
            The order of the axes is
            `~gwcs.coordinate_frames.BaseCoordinateFrame.axes_order`.
            For two image axes ``bounding_box`` is of the form
            ``((xmin, xmax), (ymin, ymax))``.

        max_pix_error
            Maximum allowed error over the domain of the pixel array. This
            error is the equivalent pixel error that corresponds to the maximum
            error in the output coordinate resulting from the fit based on
            a nominal plate scale.

        degree
            Degree of the SIP polynomial. Default value `None` indicates that
            all allowed degree values (``[1...9]``) will be considered and
            the lowest degree that meets accuracy requerements set by
            ``max_pix_error`` will be returned. Alternatively, ``degree`` can be
            an iterable containing allowed values for the SIP polynomial degree.
            This option is similar to default `None` but it allows caller to
            restrict the range of allowed SIP degrees used for fitting.
            Finally, ``degree`` can be an integer indicating the exact SIP degree
            to be fit to the WCS transformation. In this case
            ``max_pixel_error`` is ignored.

            .. note::
                When WCS object has When ``degree`` is `None` and the WCS object has

        max_inv_pix_error
            Maximum allowed inverse error over the domain of the pixel array
            in pixel units. If None, no inverse is generated.

        inv_degree
            Degree of the SIP polynomial. Default value `None` indicates that
            all allowed degree values (``[1...9]``) will be considered and
            the lowest degree that meets accuracy requerements set by
            ``max_pix_error`` will be returned. Alternatively, ``degree`` can be
            an iterable containing allowed values for the SIP polynomial degree.
            This option is similar to default `None` but it allows caller to
            restrict the range of allowed SIP degrees used for fitting.
            Finally, ``degree`` can be an integer indicating the exact SIP degree
            to be fit to the WCS transformation. In this case
            ``max_inv_pixel_error`` is ignored.

        npoints
            The number of points in each dimension to sample the bounding box
            for use in the SIP fit. Minimum number of points is 3.

        crpix
            Coordinates (1-based) of the reference point for the new FITS WCS.
            When not provided, i.e., when set to `None` (default) the reference
            pixel will be chosen near the center of the bounding box for axes
            corresponding to the celestial frame.

        projection
            Projection to be used for the created FITS WCS. It can be specified
            as a string of three characters specifying a FITS projection code
            from Table 13 in
            `Representations of World Coordinates in FITS \
            <https://doi.org/10.1051/0004-6361:20021326>`_
            (Paper I), Greisen, E. W., and Calabretta, M. R., A & A, 395,
            1061-1075, 2002. Alternatively, it can be an instance of one of the
            `astropy's Pix2Sky_* <https://docs.astropy.org/en/stable/modeling/\
            reference_api.html#module-astropy.modeling.projections>`_
            projection models inherited from
            :py:class:`~astropy.modeling.projections.Pix2SkyProjection`.

        bin_ext_name
            Extension name for the `~astropy.io.fits.BinTableHDU` HDU for those
            axes groups that will be converted using FITW WCS' ``-TAB``
            algorithm. Extension version will be determined automatically
            based on the number of separable group of axes.

        coord_col_name
            Field name of the coordinate array in the structured array
            stored in `~astropy.io.fits.BinTableHDU` data. This corresponds to
            ``TTYPEi`` field in the FITS header of the binary table extension.

        sampling
            The target "density" of grid nodes per pixel to be used when
            creating the coordinate array for the ``-TAB`` FITS WCS convention.
            It is equal to ``1/step`` where ``step`` is the distance between
            grid nodes in pixels. ``sampling`` can be specified as a single
            number to be used for all axes or as a `tuple` of numbers
            that specify the sampling for each image axis.

        verbose
            Print progress of fits.

        Returns
        -------
            Header with WCS-TAB information associated (to be used) with image
            data.

            A Python list of binary table extensions containing the coordinate
            array for TAB extensions; one extension per separable axes group.

        Raises
        ------
        ValueError
            When ``bounding_box`` is not defined either through the input
            ``bounding_box`` parameter or this object's ``bounding_box``
            property.

        ValueError
            When ``sampling`` is a `tuple` of length larger than 1 that
            does not match the number of image axes.

        RuntimeError
            If the number of image axes (``~gwcs.WCS.pixel_n_dim``) is larger
            than the number of world axes (``~gwcs.WCS.world_n_dim``).

        """
        if bounding_box is None:
            if self.bounding_box is None:
                msg = "Need a valid bounding_box to compute the footprint."
                raise ValueError(msg)
            bounding_box = self.bounding_box

        else:
            # validate user-supplied bounding box:
            frames = self.available_frames
            transform_0 = self.get_transform(frames[0], frames[1])
            Bbox.validate(transform_0, bounding_box)

        if self.forward_transform.n_inputs == 1:
            bounding_box = [bounding_box]

        if self.pixel_n_dim > self.world_n_dim:
            msg = (
                "The case when the number of input axes is larger than the "
                "number of output axes is not supported."
            )
            raise RuntimeError(msg)

        try:
            sampling = np.broadcast_to(sampling, (self.pixel_n_dim,))
        except ValueError as err:
            msg = (
                "Number of sampling values either must be 1 "
                "or it must match the number of pixel axes."
            )
            raise ValueError(msg) from err

        world_axes_groups, _, celestial_group = self._separable_groups(
            detect_celestial=True
        )

        # Find celestial axes group and treat it separately from other axes:
        if celestial_group:
            # if world_axes_groups is empty, then we have only celestial axes
            # and so we can allow arbitrary degree for SIP. When there are
            # other axes types present, issue a warning and set 'degree' to 1
            # because use of SIP when world_n_dim > 2 currently is not supported by
            # astropy.wcs.WCS - see https://github.com/astropy/astropy/pull/11452
            if world_axes_groups and (degree is None or np.max(degree) != 2):
                if degree is not None:
                    warnings.warn(
                        "SIP distortion is not supported when the number\n"
                        "of axes in WCS is larger than 2. Setting 'degree'\n"
                        "to 1 and 'max_inv_pix_error' to None.",
                        stacklevel=2,
                    )
                degree = 1
                max_inv_pix_error = None

            hdr = self._to_fits_sip(
                celestial_group=celestial_group,
                keep_axis_position=True,
                bounding_box=bounding_box,
                max_pix_error=max_pix_error,
                degree=degree,
                max_inv_pix_error=max_inv_pix_error,
                inv_degree=inv_degree,
                npoints=npoints,
                crpix=crpix,
                projection=projection,
                matrix_type="PC-CDELT1",
                verbose=verbose,
            )
            use_cd = "A_ORDER" in hdr

        else:
            use_cd = False
            hdr = fits.Header()
            hdr["NAXIS"] = 0
            hdr["WCSAXES"] = 0

        # now handle non-celestial axes using -TAB convention for each
        # separable axes group:
        hdulist = []
        for extver0, world_axes_group in enumerate(world_axes_groups):
            # For each subset of separable axes call _to_fits_tab to
            # convert that group to a single Bin TableHDU with a
            # coordinate array for this group of axes:
            hdr, bin_table_hdu = self._to_fits_tab(
                hdr=hdr,
                world_axes_group=world_axes_group,
                use_cd=use_cd,
                bounding_box=bounding_box,
                bin_ext=(bin_ext_name, extver0 + 1),
                coord_col_name=coord_col_name,
                sampling=sampling,
            )
            hdulist.append(bin_table_hdu)

        hdr.add_comment("FITS WCS created by approximating a gWCS")

        return hdr, hdulist
