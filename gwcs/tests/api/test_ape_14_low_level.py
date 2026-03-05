"""
Test the low-level API functions defined in APE 14 (https://doi.org/10.5281/zenodo.1188875).
"""

import pytest
from numpy.testing import assert_allclose

from gwcs.wcs import WCS

from .conftest import (
    check_is_low_level,
    empty_frame_warning_context,
    pixel_to_low_level,
    world_to_low_level,
)


class TestPixToWorld:
    """Test the low-level pixel to world API functions defined in APE 14."""

    def test_pixel_to_world_values(self, wcs_object: WCS, pixel, world_low, dimension):
        """
        Test that the pixel_to_world_values function returns the expected values
        """
        # The pixel_to_world_values always returns a low level output, which is
        #    is the world_low fixture
        world = world_low

        # For the 1D world we need to turn the tuple into a single value
        if (dimension == "array") and len(world) == 1:
            world = world[0]

        # High-level non-quantity inputs should be rejected by the low-level API
        #   Note that we allow high-level quantity inputs so these are not
        #   rejected by the low-level API
        if wcs_object.input_frame.is_high_level(*pixel):
            with pytest.raises(ValueError, match=r"High-Level inputs.*"):
                wcs_object.pixel_to_world_values(*pixel)
            return

        # Note we need to catch the warning when an EmptyFrame and
        #     non-low-level is involved
        with empty_frame_warning_context(wcs_object.output_frame, pixel):
            pixel_to_world = wcs_object.pixel_to_world_values(*pixel)

        # Check that this never returns a Quantity, even if the input is a Quantity
        check_is_low_level(wcs_object.output_frame, pixel_to_world)

        # For API consistency, GWCS should always return a tuple or a single object
        #    astropy.wcs may return a list instead of a tuple
        assert not isinstance(pixel_to_world, list)

        # Check that the values are correct
        assert_allclose(pixel_to_world, world)

        # Note we need to catch the warning when an EmptyFrame and
        #     non-low-level is involved
        with empty_frame_warning_context(wcs_object.output_frame, pixel):
            native_pixel_to_world = world_to_low_level(wcs_object, wcs_object(*pixel))

        # Now we check against the native API call
        assert_allclose(pixel_to_world, native_pixel_to_world)

    def test_array_index_to_world_values(
        self, wcs_object: WCS, array_index, world_low, dimension
    ):
        """
        Test that the array_index_to_world function returns the expected values
        """
        # The array_index_to_world_values always returns a low level output, which
        #    is the world_low fixture
        world = world_low

        # For the 1D world we need to turn the tuple into a single value
        if (dimension == "array") and len(world) == 1:
            world = world[0]

        # High-level non-quantity inputs should be rejected by the low-level API
        #   Note that we allow high-level quantity inputs so these are not
        #   rejected by the low-level API
        if wcs_object.input_frame.is_high_level(*array_index):
            with pytest.raises(ValueError, match=r"High-Level inputs.*"):
                wcs_object.array_index_to_world_values(*array_index)
            return

        # Note we need to catch the warning when an EmptyFrame and
        #     non-low-level is involved
        with empty_frame_warning_context(wcs_object.output_frame, array_index):
            array_index_to_world = wcs_object.array_index_to_world_values(*array_index)

        # Check that this never returns a Quantity, even if the input is a Quantity
        check_is_low_level(wcs_object.output_frame, array_index_to_world)

        # Check that the values are correct
        assert_allclose(array_index_to_world, world)

        # For API consistency, GWCS should always return a tuple or a single object
        #    astropy.wcs may return a list instead of a tuple
        assert not isinstance(array_index_to_world, list)

        # Note we need to catch the warning when an EmptyFrame and
        #     non-low-level is involved
        with empty_frame_warning_context(wcs_object.output_frame, array_index):
            native_array_index_to_world = world_to_low_level(
                # Native API does not work with array_index only pixel
                wcs_object,
                wcs_object(*array_index[::-1]),
            )

        # Now we check against the native API call
        assert_allclose(array_index_to_world, native_array_index_to_world)


class TestWorldToPix:
    """Test the low-level world to pixel API functions defined in APE 14."""

    @pytest.mark.usefixtures("xfail_gwcs_with_frames_strings")
    def test_world_to_pixel_values(
        self, wcs_object: WCS, pixel_low, world, dimension, rtol
    ):
        """
        Test that the world_to_pixel_values function returns the expected values
        """
        # The world_to_pixel_values always returns a low level output, which is
        #    is the pixel_low fixture
        pixel = pixel_low

        # For the 1D pixel we need to turn the tuple into a single value
        if (dimension == "array") and len(pixel) == 1:
            pixel = pixel[0]

        # High-level non-quantity inputs should be rejected by the low-level API
        #   Note that we allow high-level quantity inputs so these are not
        #   rejected by the low-level API
        if wcs_object.output_frame.is_high_level(*world):
            with pytest.raises(ValueError, match=r"High-Level inputs.*"):
                wcs_object.world_to_pixel_values(*world)
            return

        # Note we need to catch the warning when an EmptyFrame and
        #     non-low-level is involved
        with empty_frame_warning_context(wcs_object.input_frame, world):
            world_to_pixel = wcs_object.world_to_pixel_values(*world)

        # Check that this never returns a Quantity, even if the input is a Quantity
        check_is_low_level(wcs_object.input_frame, world_to_pixel)

        # For API consistency, GWCS should always return a tuple or a single object
        #    astropy.wcs may return a list instead of a tuple
        assert not isinstance(world_to_pixel, list)

        # Check that the values are correct
        assert_allclose(world_to_pixel, pixel, rtol=rtol)

        # Note we need to catch the warning when an EmptyFrame and
        #     non-low-level is involved
        with empty_frame_warning_context(wcs_object.input_frame, world):
            native_world_to_pixel = pixel_to_low_level(
                wcs_object, wcs_object.invert(*world)
            )

        # Now we check against the native API call
        assert_allclose(world_to_pixel, native_world_to_pixel)

    @pytest.mark.usefixtures("xfail_gwcs_with_frames_strings")
    def test_world_to_array_index_values(
        self, fixture_name, wcs_object: WCS, array_index_low, world, dimension, rtol
    ):
        """
        Test that the world_to_array_index function returns the expected values
        """
        # The world_to_pixel_values always returns a low level output, which is
        #    is the array_index_low fixture
        array_index = array_index_low

        # For the 1D pixel we need to turn the tuple into a single value
        if (dimension == "array") and len(array_index) == 1:
            array_index = array_index[0]

        # High-level non-quantity inputs should be rejected by the low-level API
        #   Note that we allow high-level quantity inputs so these are not
        #   rejected by the low-level API
        if wcs_object.output_frame.is_high_level(*world):
            with pytest.raises(ValueError, match=r"High-Level inputs.*"):
                wcs_object.world_to_array_index_values(*world)
            return

        # Note we need to catch the warning when an EmptyFrame and
        #     non-low-level is involved
        with empty_frame_warning_context(wcs_object.input_frame, world):
            world_to_array_index = wcs_object.world_to_array_index_values(*world)

        # Check that this never returns a Quantity, even if the input is a Quantity
        check_is_low_level(wcs_object.input_frame, world_to_array_index)

        # The utils.to_index function throws this off
        if fixture_name == "gwcs_7d_complex_mapping":
            rtol = 1

        # Check that the values are correct
        assert_allclose(world_to_array_index, array_index, rtol=rtol)

        # For API consistency, GWCS should always return a tuple or a single object
        #    astropy.wcs may return a list instead of a tuple
        assert not isinstance(world_to_array_index, list)

        # Note we need to catch the warning when an EmptyFrame and
        #     non-low-level is involved
        with empty_frame_warning_context(wcs_object.input_frame, world):
            native_world_to_pixel = pixel_to_low_level(
                wcs_object, wcs_object.invert(*world)
            )

        # The native API does not work with array_index only pixel,
        #     so we need to reverse the order of the output
        native_world_to_array_index = (
            native_world_to_pixel[::-1]
            if isinstance(native_world_to_pixel, tuple)
            else native_world_to_pixel
        )

        # Now we check against the native API call
        assert_allclose(world_to_array_index, native_world_to_array_index, rtol=rtol)
