from contextlib import nullcontext

import numpy as np
import pytest
from astropy import coordinates as coord
from astropy import time
from astropy import units as u
from numpy.testing import assert_allclose

from gwcs.coordinate_frames import (
    CoordinateFrameProtocol,
    EmptyFrame,
    EmptyFrameUnitsWarning,
)
from gwcs.wcs import WCS

fixture_names = (
    "gwcs_2d_spatial_shift",
    "gwcs_2d_spatial_reordered",
    "gwcs_2d_quantity_shift",
    "gwcs_1d_freq",
    "gwcs_1d_spectral",
    "gwcs_3d_spatial_wave",
    "gwcs_3d_identity_units",
    "gwcs_4d_identity_units",
    "gwcs_stokes_lookup",
    "gwcs_3d_galactic_spectral",
    "gwcs_2d_shift_scale",
    "gwcs_2d_shift_scale_quantity",
    "gwcs_1d_freq_quantity",
    "gwcs_simple_2d",
    "gwcs_empty_output_2d",
    "gwcs_simple_imaging",
    "gwcs_simple_imaging_units",
    "gwcs_with_frames_strings",
    "gwcs_high_level_pixel",
    "gwcs_3spectral_orders",
    "gwcs_spec_cel_time_4d",
    "gwcs_7d_complex_mapping",
    "gwcs_with_pipeline_celestial",
    "gwcs_romanisim",
    "gwcs_2d_spatial_shift_reverse",
    # This fixture exposes a bug in the from_high_level_coordinates method of the
    #     Frame class
    # "gwcs_multi_stage",
)


@pytest.fixture(params=fixture_names)
def fixture_name(request):
    return request.param


@pytest.fixture
def wcs_object(request, fixture_name) -> WCS:
    return request.getfixturevalue(fixture_name)


@pytest.fixture(params=["scalar", "array"])
def dimension(request):
    return request.param


@pytest.fixture(params=["low", "quantity", "high"])
def level(request):
    return request.param


@pytest.fixture
def pixel_scalar(fixture_name):  # noqa: PLR0911
    match fixture_name:
        case (
            "gwcs_1d_freq"
            | "gwcs_1d_freq_quantity"
            | "gwcs_stokes_lookup"
            | "gwcs_multi_stage"
        ):
            return (1,)
        case "gwcs_1d_spectral":
            return (30,)
        case (
            "gwcs_2d_spatial_shift"
            | "gwcs_2d_spatial_reordered"
            | "gwcs_2d_quantity_shift"
            | "gwcs_2d_shift_scale"
            | "gwcs_2d_shift_scale_quantity"
            | "gwcs_simple_2d"
            | "gwcs_empty_output_2d"
            | "gwcs_simple_imaging"
            | "gwcs_simple_imaging_units"
            | "gwcs_high_level_pixel"
            | "gwcs_3spectral_orders"
            | "gwcs_with_pipeline_celestial"
            | "gwcs_romanisim"
            | "gwcs_2d_spatial_shift_reverse"
        ):
            return 1, 2
        case "gwcs_3d_spatial_wave" | "gwcs_3d_identity_units":
            return 1, 2, 3
        case "gwcs_3d_galactic_spectral":
            return 10, 20, 30
        case (
            "gwcs_4d_identity_units"
            | "gwcs_with_frames_strings"
            | "gwcs_spec_cel_time_4d"
        ):
            return 1, 2, 3, 4
        case "gwcs_7d_complex_mapping":
            return 1, 2, 3, 4, 5, 0.5
        case _:
            msg = f"Unknown fixture name: {fixture_name}"
            raise ValueError(msg)


@pytest.fixture
def pixel_low(pixel_scalar, dimension):
    if dimension == "scalar":
        return pixel_scalar

    return tuple(np.ones((3, 4)) * arg for arg in pixel_scalar)


@pytest.fixture
def array_index_low(pixel_low):
    return pixel_low[::-1]


def pixel_to_quantity(wcs_object, pixel):
    """Convert pixel coordinates to world coordinates as quantities."""

    with (
        pytest.warns(EmptyFrameUnitsWarning, match=r"EmptyFrame.*")
        if isinstance(wcs_object.input_frame, EmptyFrame)
        else nullcontext()
    ):
        return wcs_object.input_frame.add_units(pixel)


@pytest.fixture
def pixel_quantity(wcs_object, pixel_low):
    """Convert pixel coordinates to world coordinates as quantities."""

    return pixel_to_quantity(wcs_object, pixel_low)


def pixel_to_high_level(wcs_object, pixel, correct_1d=True):
    """Convert pixel coordinates to high-level world coordinates."""
    return wcs_object.input_frame.to_high_level_coordinates(
        *pixel, correct_1d=correct_1d
    )


@pytest.fixture
def pixel_high(wcs_object, pixel_low):
    """Convert pixel coordinates to high-level world coordinates."""

    return pixel_to_high_level(wcs_object, pixel_low, correct_1d=False)


@pytest.fixture
def pixel(pixel_low, pixel_quantity, pixel_high, level):
    if level == "quantity":
        return pixel_quantity

    if level == "high":
        return pixel_high

    return pixel_low


# TODO: Should this be turned into a method on the Frame?
def pixel_to_low_level(wcs_object: WCS, pixel):
    """Convert world coordinates to low-level world coordinates."""

    remove_tuple = False
    if not isinstance(pixel, tuple):
        remove_tuple = True
        pixel = (pixel,)

    # If we are high-level, convert to low-level
    if wcs_object.input_frame.is_high_level(*pixel):
        return wcs_object.input_frame.from_high_level_coordinates(*pixel)

    # Remove units just in case
    pixel = wcs_object.input_frame.remove_units(pixel)

    if remove_tuple:
        return pixel[0]

    return pixel


@pytest.fixture
def array_index(wcs_object, pixel):
    """Convert pixel coordinates to array index coordinates."""

    # True high-level inputs have no corresponding array index
    if wcs_object.input_frame.is_high_level(*pixel):
        return pixel

    return pixel[::-1]


@pytest.fixture
def world_scalar(fixture_name):  # noqa: PLR0911
    match fixture_name:
        case "gwcs_1d_freq" | "gwcs_stokes_lookup":
            return (2,)
        case "gwcs_1d_spectral":
            return (112.5,)
        case "gwcs_1d_freq_quantity":
            return (1,)
        case (
            "gwcs_2d_spatial_shift"
            | "gwcs_2d_quantity_shift"
            | "gwcs_simple_2d"
            | "gwcs_empty_output_2d"
            | "gwcs_high_level_pixel"
            | "gwcs_2d_spatial_shift_reverse"
        ):
            return 2, 4
        case "gwcs_2d_spatial_reordered":
            return 4, 2
        case "gwcs_2d_shift_scale" | "gwcs_2d_shift_scale_quantity":
            return 10, 40
        case "gwcs_simple_imaging" | "gwcs_simple_imaging_units":
            return 5.525098, -72.051902
        case "gwcs_with_pipeline_celestial":
            return 3620.0, 115200.0
        case "gwcs_romanisim":
            return 3.0555555554143875e-05, 6.111111e-05
        case "gwcs_multi_stage":
            return -22.0, -11.0
        case "gwcs_3d_spatial_wave":
            return 2, 4, 6
        case "gwcs_3d_identity_units":
            return 1, 2, 3
        case "gwcs_3d_galactic_spectral":
            return 79.80904918923017, 10.5, 205.79129497471095
        case "gwcs_4d_identity_units":
            return 2.777778e-04, 5.555556e-04, 3, 4
        case "gwcs_with_frames_strings":
            return 2, 3, 0
        case "gwcs_3spectral_orders":
            return 2, 4, 4
        case "gwcs_spec_cel_time_4d":
            return 5.2, -72.049906, 5.629411, 4
        case "gwcs_7d_complex_mapping":
            return 10.8, -72.054537, 5.627349, 6.75066, -0.14, 1.87885, 10.8
        case _:
            msg = f"Unknown fixture name: {fixture_name}"
            raise ValueError(msg)


@pytest.fixture
def world_low(world_scalar, dimension):
    if dimension == "scalar":
        return world_scalar

    return tuple(np.ones((3, 4)) * arg for arg in world_scalar)


def world_to_quantity(wcs_object, world):
    """Convert world coordinates to world coordinates as quantities."""

    with (
        pytest.warns(EmptyFrameUnitsWarning, match=r"EmptyFrame.*")
        if isinstance(wcs_object.output_frame, EmptyFrame)
        else nullcontext()
    ):
        return wcs_object.output_frame.add_units(world)


@pytest.fixture
def world_quantity(wcs_object, world_low):
    """Convert world coordinates to world coordinates as quantities."""

    return world_to_quantity(wcs_object, world_low)


def world_to_high_level(wcs_object, world, correct_1d=True):
    """Convert world coordinates to high-level world coordinates."""
    return wcs_object.output_frame.to_high_level_coordinates(
        *world, correct_1d=correct_1d
    )


@pytest.fixture
def world_high(wcs_object, world_low):
    """Convert world coordinates to high-level world coordinates."""

    return world_to_high_level(wcs_object, world_low, correct_1d=False)


@pytest.fixture
def world(world_low, world_quantity, world_high, level):
    if level == "quantity":
        return world_quantity

    if level == "high":
        return world_high

    return world_low


# TODO: Should this be turned into a method on the Frame?
def world_to_low_level(wcs_object: WCS, world):
    """Convert world coordinates to low-level world coordinates."""

    remove_tuple = False
    if not isinstance(world, tuple):
        remove_tuple = True
        world = (world,)

    # If we are high-level, convert to
    if wcs_object.output_frame.is_high_level(*world):
        world = wcs_object.output_frame.from_high_level_coordinates(*world)
    else:
        world = wcs_object.output_frame.remove_units(world)

    if remove_tuple:
        return world[0]

    return world


@pytest.fixture
def xfail_gwcs_with_frames_strings(fixture_name, request):
    if fixture_name == "gwcs_with_frames_strings":
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Numerical inverse not supported when n_inputs != n_outputs"
            )
        )


@pytest.fixture
def xfail_gwcs_with_frames_strings_high(level, request):
    if level == "high":
        request.getfixturevalue("xfail_gwcs_with_frames_strings")


@pytest.fixture
def rtol(fixture_name):
    """
    gwcs_simple_imaging's inverse is not exact, so we need a looser tolerance for it.
    """

    match fixture_name:
        case (
            "gwcs_simple_imaging"
            | "gwcs_simple_imaging_units"
            | "gwcs_spec_cel_time_4d"
        ):
            return 0.1
        case "gwcs_7d_complex_mapping":
            return 1e-2
        case _:
            return 1e-07


def check_is_low_level(frame: CoordinateFrameProtocol, output):
    if isinstance(output, tuple | list):
        assert not any(isinstance(p, u.Quantity) for p in output)
        assert not frame.is_high_level(*output)
    else:
        assert not isinstance(output, u.Quantity)
        assert not frame.is_high_level(*(output,))


def check_is_high_level(frame: CoordinateFrameProtocol, output):
    if isinstance(output, tuple | list):
        assert all(isinstance(w, u.Quantity) for w in output) or frame.is_high_level(
            *output
        )
    else:
        assert isinstance(output, u.Quantity) or frame.is_high_level(*(output,))


def is_quantity(output):
    """Check we get exactly Quantities"""
    if isinstance(output, tuple | list):
        return all(type(p) is u.Quantity for p in output)

    return type(output) is u.Quantity


def empty_frame_warning_context(frame: CoordinateFrameProtocol, inputs):
    return (
        pytest.warns(EmptyFrameUnitsWarning, match=r"EmptyFrame.*")
        if isinstance(frame, EmptyFrame)
        and all(isinstance(arr, u.Quantity) for arr in inputs)
        else nullcontext()
    )


def compare_frame_output(wc1, wc2, rtol=1e-07):
    if isinstance(wc1, coord.SkyCoord):
        assert isinstance(wc1.frame, type(wc2.frame))
        assert u.allclose(
            wc1.spherical.lon, wc2.spherical.lon, equal_nan=True, rtol=rtol
        )
        assert u.allclose(
            wc1.spherical.lat, wc2.spherical.lat, equal_nan=True, rtol=rtol
        )
        assert u.allclose(
            wc1.spherical.distance, wc2.spherical.distance, equal_nan=True, rtol=rtol
        )

    elif isinstance(wc1, u.Quantity):
        assert u.allclose(wc1, wc2, equal_nan=True, rtol=rtol)

    elif isinstance(wc1, time.Time):
        assert u.allclose((wc1 - wc2).to(u.s), 0 * u.s, rtol=rtol)

    elif isinstance(wc1, str | coord.StokesCoord):
        assert np.array(wc1 == wc2, dtype=bool).all()

    elif isinstance(wc1, np.ndarray | np.float64 | float):
        assert_allclose(wc1, wc2, equal_nan=True, rtol=rtol)

    elif isinstance(wc1, (tuple, list)):
        for w1, w2 in zip(wc1, wc2, strict=True):
            compare_frame_output(w1, w2, rtol=rtol)

    else:
        msg = f"Can't Compare {type(wc1)}"
        raise TypeError(msg)
