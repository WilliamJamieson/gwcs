"""
Test the high-level API functions defined in APE 14 (https://doi.org/10.5281/zenodo.1188875).
"""

import numpy as np
import pytest
from astropy import coordinates as coord
from astropy import units as u
from astropy.wcs.wcsapi import HighLevelWCSWrapper
from numpy.testing import assert_allclose

from gwcs.wcs import WCS

from .conftest import (
    check_is_high_level,
    check_is_low_level,
    compare_frame_output,
    empty_frame_warning_context,
    pixel_to_low_level,
)


class TestPixToWorld:
    """Test the high-level pixel to world API functions defined in APE 14."""

    def test_pixel_to_world(self, wcs_object: WCS, pixel, world_high):
        """Test the pixel to world API function."""

        # The pixel_to_world always returns a high-level world coordinate,
        #    which is the world_high fixture
        world = world_high

        # For 1D world coordinates, the fixture will return a tuple, but we want to
        #   have a single value here
        if isinstance(world, tuple | list) and len(world) == 1:
            world = world[0]

        # Check that we can pass low-level pixel coordinates into this without
        #   any issues and get the high-level world coordinates out
        # Note this works only because APE 14 does not really describe any
        #    non-quantity high-level pixel coordinates.
        # Note we need to catch the warning when an EmptyFrame and
        #     non-low-level is involved
        with empty_frame_warning_context(wcs_object.output_frame, pixel):
            pixel_to_world = wcs_object.pixel_to_world(*pixel)

        # This should always be a high-level world coordinate
        check_is_high_level(wcs_object.output_frame, pixel_to_world)

        # For API consistency, GWCS should always return a tuple or a single object
        #    astropy.wcs may return a list instead of a tuple
        assert not isinstance(pixel_to_world, list)

        # Check the values are correct
        compare_frame_output(pixel_to_world, world)

        # Note we need to catch the warning when an EmptyFrame and
        #     non-low-level is involved
        with empty_frame_warning_context(wcs_object.output_frame, pixel):
            native_pixel_to_world = wcs_object(*pixel, force_high_level=True)

        # Now we check against the native API call
        compare_frame_output(pixel_to_world, native_pixel_to_world)

    def test_array_index_to_world(self, wcs_object: WCS, array_index, world_high):
        """Test the array index to world API function."""

        # The array_index_to_world always returns a high-level world coordinate,
        #    which is the world_high fixture
        world = world_high

        # For 1D world coordinates, the fixture will return a tuple, but we want to
        #   have a single value here
        if isinstance(world, tuple | list) and len(world) == 1:
            world = world[0]

        # Check that we can pass low-level pixel coordinates into this without
        #   any issues and get the high-level world coordinates out
        # Note this works only because APE 14 does not really describe any
        #    non-quantity high-level pixel coordinates.
        # Note we need to catch the warning when an EmptyFrame and
        #     non-low-level is involved
        with empty_frame_warning_context(wcs_object.output_frame, array_index):
            array_index_to_world = wcs_object.array_index_to_world(*array_index)

        # This should always be a high-level world coordinate
        check_is_high_level(wcs_object.output_frame, array_index_to_world)

        # For API consistency, GWCS should always return a tuple or a single object
        #    astropy.wcs may return a list instead of a tuple
        assert not isinstance(array_index_to_world, list)

        # Check the values are correct
        compare_frame_output(array_index_to_world, world)

        # Note we need to catch the warning when an EmptyFrame and
        #     non-low-level is involved
        with empty_frame_warning_context(wcs_object.output_frame, array_index):
            native_array_index_to_world = wcs_object(
                *array_index[::-1], force_high_level=True
            )

        # Now we check against the native API call
        compare_frame_output(array_index_to_world, native_array_index_to_world)

    def test_high_level_wrapper_pixel(
        self, fixture_name, wcs_object: WCS, pixel, world_high, level, request
    ):
        """Test the high-level wrapper with pixel_to_world."""
        if fixture_name == "gwcs_high_level_pixel" and level == "high":
            msg = (
                "The `astropy.wcs` wrapper does not support wcs that are not"
                " purely pixel in the input_frame."
            )
            request.node.add_marker(pytest.mark.xfail(reason=msg))

        # The high-level wrapper only works with high-level world coordinates
        world = world_high

        # For 1D world coordinates, the fixture will return a tuple, but we want to
        #   have a single value here
        if isinstance(world, tuple | list) and len(world) == 1:
            world = world[0]

        hlvl = HighLevelWCSWrapper(wcs_object)

        with empty_frame_warning_context(wcs_object.output_frame, pixel):
            pixel_to_world = hlvl.pixel_to_world(*pixel)

        # This should always be a high-level world coordinate
        check_is_high_level(wcs_object.output_frame, pixel_to_world)

        # Check that the values are correct
        compare_frame_output(pixel_to_world, world)

    def test_high_level_wrapper_array_index(
        self, fixture_name, wcs_object: WCS, array_index, world_high, level, request
    ):
        """Test the high-level wrapper with array_index_to_world."""
        if fixture_name == "gwcs_high_level_pixel" and level == "high":
            msg = (
                "The `astropy.wcs` wrapper does not support wcs that are not"
                " purely pixel in the input_frame."
            )
            request.node.add_marker(pytest.mark.xfail(reason=msg))

        # The high-level wrapper only works with high-level world coordinates
        world = world_high

        # For 1D world coordinates, the fixture will return a tuple, but we want to
        #   have a single value here
        if isinstance(world, tuple | list) and len(world) == 1:
            world = world[0]

        hlvl = HighLevelWCSWrapper(wcs_object)

        with empty_frame_warning_context(wcs_object.output_frame, array_index):
            array_index_to_world = hlvl.array_index_to_world(*array_index)

        # This should always be a high-level world coordinate
        check_is_high_level(wcs_object.output_frame, array_index_to_world)

        # Check that the values are correct
        compare_frame_output(array_index_to_world, world)


class TestWorldToPixel:
    """Test the high-level world to pixel API functions defined in APE 14."""

    @pytest.mark.usefixtures("xfail_gwcs_with_frames_strings_high")
    def test_world_to_pixel(
        self, wcs_object: WCS, world, world_high, pixel_low, pixel_high, dimension, rtol
    ):
        """Test the world to pixel API function."""

        # The world_to_pixel always returns a low-level pixel coordinate, which is
        #    the pixel_low fixture
        pixel = pixel_low

        # For the 1D pixel we need to turn the tuple into a single value
        if (dimension == "array") and len(pixel) == 1:
            pixel = pixel[0]

        # Test that we cannot pass low-level world coordinates into this
        try:
            world_to_pixel = wcs_object.world_to_pixel(*world)

        # Any failure here should be because the world fixture is not identical
        #   to the high-level world fixture. In some cases, Quantities maybe
        #   considered high-level for APE 14
        # In this case we want to make sure we are failing when a non-high-level
        #    world coordinate object is passed into the high-level API
        except ValueError:
            assert world is not world_high
            # Cannot progress further in these cases
            return

        # APE 14 always returns a low-level pixel coordinate from the world_to_pixel
        #   function
        check_is_low_level(wcs_object.input_frame, world_to_pixel)

        # For API consistency, GWCS should always return a tuple or a single object
        #    astropy.wcs may return a list instead of a tuple
        assert not isinstance(world_to_pixel, list)

        assert_allclose(world_to_pixel, pixel, rtol=rtol)

        with empty_frame_warning_context(wcs_object.input_frame, world):
            native_world_to_pixel = wcs_object.invert(*world, force_high_level=True)

        # Since force_hign_level=True this should always be a high_level pixel
        #    coordinate
        check_is_high_level(wcs_object.input_frame, native_world_to_pixel)

        # For 1D pixel coordinates, the fixture will return a tuple, but we want to
        #   have a single value here
        if isinstance(pixel_high, tuple | list) and len(pixel_high) == 1:
            pixel_high = pixel_high[0]

        compare_frame_output(native_world_to_pixel, pixel_high, rtol=rtol)

        with empty_frame_warning_context(wcs_object.input_frame, native_world_to_pixel):
            native_world_to_pixel_value = pixel_to_low_level(
                wcs_object, native_world_to_pixel
            )

        # Now we check against the native API call
        assert_allclose(world_to_pixel, native_world_to_pixel_value)

    @pytest.mark.usefixtures("xfail_gwcs_with_frames_strings_high")
    def test_world_to_array_index(
        self,
        wcs_object: WCS,
        world,
        world_high,
        array_index_low,
        pixel_high,
        dimension,
        rtol,
    ):
        """Test the world to array index API function."""

        # The world_to_pixel always returns a low-level pixel coordinate, which is
        #    the pixel_low fixture
        array_index = array_index_low

        # For the 1D array_index we need to turn the tuple into a single value
        if (dimension == "array") and len(array_index) == 1:
            array_index = array_index[0]

        # Test that we cannot pass low-level world coordinates into this
        try:
            world_to_array_index = wcs_object.world_to_array_index(*world)

        # Any failure here should be because the world fixture is not identical
        #   to the high-level world fixture. In some cases, Quantities maybe
        #   considered high-level for APE 14
        # In this case we want to make sure we are failing when a non-high-level
        #    world coordinate object is passed into the high-level API
        except ValueError:
            assert world is not world_high
            # Cannot progress further in these cases
            return

        # APE 14 always returns a low-level array_index coordinate from the
        #   world_to_array_index function
        check_is_low_level(wcs_object.input_frame, world_to_array_index)

        # For API consistency, GWCS should always return a tuple or a single object
        #    astropy.wcs may return a list instead of a tuple
        assert not isinstance(world_to_array_index, list)

        assert_allclose(world_to_array_index, array_index, rtol=rtol)

        with empty_frame_warning_context(wcs_object.input_frame, world):
            native_world_to_pixel = wcs_object.invert(*world, force_high_level=True)

        # Since force_hign_level=True this should always be a high_level array_index
        #    coordinate
        check_is_high_level(wcs_object.input_frame, native_world_to_pixel)

        # For 1D pixel coordinates, the fixture will return a tuple, but we want to
        #   have a single value here
        if isinstance(pixel_high, tuple | list) and len(pixel_high) == 1:
            pixel_high = pixel_high[0]

        compare_frame_output(native_world_to_pixel, pixel_high, rtol=rtol)

        with empty_frame_warning_context(wcs_object.input_frame, native_world_to_pixel):
            native_world_to_pixel_value = pixel_to_low_level(
                wcs_object, native_world_to_pixel
            )
        # Switch from pixel to array_index coordinates if needed
        native_world_to_array_index = (
            native_world_to_pixel_value[::-1]
            if isinstance(native_world_to_pixel_value, tuple | list)
            else native_world_to_pixel_value
        )

        # Now we check against the native API call
        assert_allclose(world_to_array_index, native_world_to_array_index, rtol=rtol)

    def test_high_level_wrapper_pixel(
        self,
        fixture_name,
        wcs_object: WCS,
        pixel_low,
        world,
        world_high,
        dimension,
        level,
        request,
        rtol,
    ):
        """Test the high-level wrapper with world_to_pixel."""
        if fixture_name == "gwcs_with_frames_strings" and level == "high":
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="Numerical inverse not supported when n_inputs != n_outputs"
                )
            )
        # APE 14 always returns a low-level pixel coordinate from the world_to_pixel
        pixel = pixel_low

        # For the 1D pixel we need to turn the tuple into a single value
        if (dimension == "array") and len(pixel) == 1:
            pixel = pixel[0]

        hlvl = HighLevelWCSWrapper(wcs_object)

        # Test that we cannot pass low-level world coordinates into this
        try:
            world_to_pixel = hlvl.world_to_pixel(*world)

        # Any failure here should be because the world fixture is not identical
        #   to the high-level world fixture. In some cases, Quantities maybe
        #   considered high-level for APE 14
        # In this case we want to make sure we are failing when a non-high-level
        #    world coordinate object is passed into the high-level API
        except ValueError:
            assert world is not world_high
            # Cannot progress further in these cases
            return

        # This should always be a low-level pixel coordinate
        check_is_low_level(wcs_object.input_frame, world_to_pixel)

        # Check that the values are correct
        assert_allclose(world_to_pixel, pixel, rtol=rtol)

    def test_high_level_wrapper_array_index(
        self,
        fixture_name,
        wcs_object: WCS,
        array_index_low,
        world,
        world_high,
        dimension,
        level,
        request,
        rtol,
    ):
        """Test the high-level wrapper with world_to_array_index."""
        if fixture_name == "gwcs_with_frames_strings" and level == "high":
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="Numerical inverse not supported when n_inputs != n_outputs"
                )
            )
        # APE 14 always returns a low-level array_index coordinate from the
        #    world_to_array_index
        array_index = array_index_low

        # For the 1D array_index we need to turn the tuple into a single value
        if (dimension == "array") and len(array_index) == 1:
            array_index = array_index[0]

        hlvl = HighLevelWCSWrapper(wcs_object)

        # Test that we cannot pass low-level world coordinates into this
        try:
            world_to_array_index = hlvl.world_to_array_index(*world)

        # Any failure here should be because the world fixture is not identical
        #   to the high-level world fixture. In some cases, Quantities maybe
        #   considered high-level for APE 14
        # In this case we want to make sure we are failing when a non-high-level
        #    world coordinate object is passed into the high-level API
        except ValueError:
            assert world is not world_high
            # Cannot progress further in these cases
            return

        # This should always be a low-level array_index coordinate
        check_is_low_level(wcs_object.input_frame, world_to_array_index)

        # Check that the values are correct
        assert_allclose(world_to_array_index, array_index)


def test_stokes_wrapper(gwcs_stokes_lookup):
    hlvl = HighLevelWCSWrapper(gwcs_stokes_lookup)

    pixel_input = [0, 1, 2, 3]

    out = hlvl.pixel_to_world(pixel_input * u.pix)

    assert list(out) == ["I", "Q", "U", "V"]

    pixel_input = [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]

    out = hlvl.pixel_to_world(pixel_input * u.pix)

    expected = coord.StokesCoord(
        [
            ["I", "Q", "U", "V"],
            ["I", "Q", "U", "V"],
            ["I", "Q", "U", "V"],
            ["I", "Q", "U", "V"],
        ]
    )

    assert (out == expected).all()

    pixel_input = [-1, 4]

    out = hlvl.pixel_to_world(pixel_input * u.pix)

    assert np.isnan(out.value).all()

    pixel_input = [[-1, 4], [1, 2]]

    out = hlvl.pixel_to_world(pixel_input * u.pix)

    assert np.isnan(out[0].value).all()
    assert (out[1] == ["Q", "U"]).all()

    out = hlvl.pixel_to_world(1 * u.pix)

    assert isinstance(out, coord.StokesCoord)
    assert out == "Q"
