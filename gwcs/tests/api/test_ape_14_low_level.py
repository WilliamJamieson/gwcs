"""
Test the low-level API functions defined in APE 14 (https://doi.org/10.5281/zenodo.1188875).
"""

from contextlib import nullcontext

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from gwcs import EmptyFrame, EmptyFrameUnitsWarning


@pytest.mark.parametrize("with_units", [True, False])
class TestPixToWorld:
    """Test the low-level pixel to world API functions defined in APE 14."""

    def _get_input_as_unit(self, wcs_object, pixels):
        """Attached units to the input pixels if the WCS object has units defined."""
        with (
            pytest.warns(EmptyFrameUnitsWarning, match=r"EmptyFrame.*")
            if isinstance(wcs_object.input_frame, EmptyFrame)
            else nullcontext()
        ):
            return wcs_object.input_frame.add_units(pixels)

    def test_pixel_to_world_values_scalar(self, wcs_object, pixels, world, with_units):
        """
        Test that the pixel_to_world_values function returns the expected values
            with scalar inputs
        """
        # Test that it matches the native API call
        assert_allclose(wcs_object.pixel_to_world_values(*pixels), wcs_object(*pixels))

        # Introduce units (gwcs only supported)
        if with_units:
            pixels = self._get_input_as_unit(wcs_object, pixels)

        # Check that this never returns a Quantity, even if the input is a Quantity
        pixel_to_world = wcs_object.pixel_to_world_values(*pixels)
        if isinstance(pixel_to_world, tuple):
            assert not any(isinstance(w, u.Quantity) for w in pixel_to_world)
        else:
            assert not isinstance(pixel_to_world, u.Quantity)

        # Check that the values are correct
        assert_allclose(pixel_to_world, world)

    def test_array_index_to_world_values_scalar(
        self, wcs_object, pixels, world, with_units
    ):
        """
        Test that the array_index_to_world function returns the expected values
            with scalar inputs
        """
        # Test that it matches the native API call
        assert_allclose(
            wcs_object.array_index_to_world_values(*pixels[::-1]),
            wcs_object(*pixels),
        )

        # Introduce units (gwcs only supported)
        if with_units:
            pixels = self._get_input_as_unit(wcs_object, pixels)

        # Check that this never returns a Quantity, even if the input is a Quantity
        array_index_to_world = wcs_object.array_index_to_world_values(*pixels[::-1])
        if isinstance(array_index_to_world, tuple):
            assert not any(isinstance(w, u.Quantity) for w in array_index_to_world)
        else:
            assert not isinstance(array_index_to_world, u.Quantity)

        # Check that the values are correct
        assert_allclose(array_index_to_world, world)

    def test_pixel_to_world_value_array(self, wcs_object, pixels, world, with_units):
        """
        Test that the pixel_to_world_values function returns the expected values
            with array inputs
        """
        # Turn the input pixels and world into arrays
        pixels = tuple(np.ones((3, 4)) * arg for arg in pixels)
        if len(world) == 1:
            world = np.ones((3, 4)) * world[0]
        else:
            world = tuple(np.ones((3, 4)) * arg for arg in world)

        # Test that it matches the native API call
        assert_allclose(wcs_object.pixel_to_world_values(*pixels), wcs_object(*pixels))

        # Check that the low-level API rejects high-level inputs.
        high_level = wcs_object.input_frame.to_high_level_coordinates(
            *pixels, correct_1d=False
        )
        # Technically, we support quantities as low-level inputs, so we need to allow
        #    them to pass through
        if all(type(hlo) is u.Quantity for hlo in high_level):
            with (
                pytest.warns(EmptyFrameUnitsWarning, match=r"EmptyFrame.*")
                if isinstance(wcs_object.input_frame, EmptyFrame)
                else nullcontext()
            ):
                assert_allclose(wcs_object.pixel_to_world_values(*high_level), world)
        else:
            with pytest.raises(ValueError, match=r"High-Level inputs.*"):
                wcs_object.pixel_to_world_values(*high_level)

        # Introduce units (gwcs only supported)
        if with_units:
            pixels = self._get_input_as_unit(wcs_object, pixels)

        # Check that this never returns a Quantity, even if the input is a Quantity
        pixel_to_world = wcs_object.pixel_to_world_values(*pixels)
        if isinstance(pixel_to_world, tuple):
            assert not any(isinstance(w, u.Quantity) for w in pixel_to_world)
        else:
            assert not isinstance(pixel_to_world, u.Quantity)

        # Check that the values are correct
        assert_allclose(pixel_to_world, world)

    def test_array_index_to_world_values_array(
        self, wcs_object, pixels, world, with_units
    ):
        """
        Test that the array_index_to_world_values function returns the expected values
        with array inputs
        """
        # Turn the input pixels and world into arrays
        pixels = tuple(np.ones((3, 4)) * arg for arg in pixels)
        if len(world) == 1:
            world = np.ones((3, 4)) * world[0]
        else:
            world = tuple(np.ones((3, 4)) * arg for arg in world)

        # Test that it matches the native API call
        assert_allclose(wcs_object.array_index_to_world_values(*pixels[::-1]), world)

        # Check that the low-level API rejects high-level inputs.
        high_level = wcs_object.input_frame.to_high_level_coordinates(
            *pixels[::-1], correct_1d=False
        )
        # Technically, we support quantities as low-level inputs, so we need to allow
        #    them to pass through
        if all(type(hlo) is u.Quantity for hlo in high_level):
            with (
                pytest.warns(EmptyFrameUnitsWarning, match=r"EmptyFrame.*")
                if isinstance(wcs_object.input_frame, EmptyFrame)
                else nullcontext()
            ):
                assert_allclose(
                    wcs_object.array_index_to_world_values(*high_level), world
                )
        else:
            with pytest.raises(ValueError, match=r"High-Level inputs.*"):
                wcs_object.array_index_to_world_values(*high_level)

        # Introduce units (gwcs only supported)
        if with_units:
            pixels = self._get_input_as_unit(wcs_object, pixels)

        # Check that this never returns a Quantity, even if the input is a Quantity
        array_index_to_world = wcs_object.array_index_to_world_values(*pixels[::-1])
        if isinstance(array_index_to_world, tuple):
            assert not any(isinstance(w, u.Quantity) for w in array_index_to_world)
        else:
            assert not isinstance(array_index_to_world, u.Quantity)

        # Check that the values are correct
        assert_allclose(array_index_to_world, world)


@pytest.mark.parametrize("with_units", [True, False])
class TestWorldToPix:
    """Test the low-level world to pixel API functions defined in APE 14."""

    def _get_input_as_unit(self, wcs_object, world):
        """Attached units to the input pixels if the WCS object has units defined."""
        with (
            pytest.warns(EmptyFrameUnitsWarning, match=r"EmptyFrame.*")
            if isinstance(wcs_object.output_frame, EmptyFrame)
            else nullcontext()
        ):
            return wcs_object.output_frame.add_units(world)

    def test_world_to_pixel_values_scalar(
        self, fixture_name, wcs_object, pixels, world, request, with_units
    ):
        """
        Test that the world_to_pixel_values function returns the expected values
            with scalar inputs
        """
        if fixture_name == "gwcs_with_frames_strings":
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="Numerical inverse not supported when n_inputs != n_outputs"
                )
            )

        # The gwcs_simple_imaging's inverse is not exact so we need to relax the
        #    tolerance for this test
        rtol = 0.1 if fixture_name == "gwcs_simple_imaging" else 1e-07

        # Test that it matches the native API call
        assert_allclose(
            wcs_object.world_to_pixel_values(*world),
            wcs_object.invert(*world),
        )
        # Introduce units (gwcs only supported)
        if with_units:
            world = self._get_input_as_unit(wcs_object, world)

        # Check that this never returns a Quantity, even if the input is a Quantity
        world_to_pixel = wcs_object.world_to_pixel_values(*world)
        if isinstance(world_to_pixel, tuple):
            assert not any(isinstance(p, u.Quantity) for p in world_to_pixel)
        else:
            assert not isinstance(world_to_pixel, u.Quantity)

        # Check that the values are correct
        assert_allclose(world_to_pixel, pixels, rtol=rtol)

    def test_world_to_array_index_values_scalar(
        self, fixture_name, wcs_object, pixels, world, request, with_units
    ):
        """
        Test that the world_to_array_index function returns the expected values
            with scalar inputs
        """
        if fixture_name == "gwcs_with_frames_strings":
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="Numerical inverse not supported when n_inputs != n_outputs"
                )
            )

        # The gwcs_simple_imaging's inverse is not exact so we need to relax the
        #    tolerance for this test
        rtol = 0.1 if fixture_name == "gwcs_simple_imaging" else 1e-07

        # Test that it matches the native API call
        if isinstance(output := wcs_object.world_to_array_index_values(*world), tuple):
            output = output[::-1]
        assert_allclose(output, wcs_object.invert(*world), rtol=rtol)

        # Introduce units (gwcs only supported)
        if with_units:
            world = self._get_input_as_unit(wcs_object, world)

        # Check that this never returns a Quantity, even if the input is a Quantity
        world_to_array_index = wcs_object.world_to_array_index_values(*world)
        if isinstance(world_to_array_index, tuple):
            assert not any(isinstance(p, u.Quantity) for p in world_to_array_index)
        else:
            assert not isinstance(world_to_array_index, u.Quantity)

        # Check that the values are correct
        assert_allclose(world_to_array_index, pixels[::-1])

    def test_world_to_pixel_value_array(
        self, fixture_name, wcs_object, pixels, world, request, with_units
    ):
        """
        Test that the world_to_pixel_values function returns the expected values
            with array inputs
        """
        if fixture_name == "gwcs_with_frames_strings":
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="Numerical inverse not supported when n_inputs != n_outputs"
                )
            )

        # Turn the input pixels and world into arrays
        pixels = tuple(np.ones((3, 4)) * arg for arg in pixels)
        if len(pixels) == 1:
            pixels = pixels[0]

        if isinstance(world, tuple):
            world = tuple(np.ones((3, 4)) * arg for arg in world)
        else:
            world = np.ones((3, 4)) * world

        # Test that it matches the native API call
        assert_allclose(
            wcs_object.world_to_pixel_values(*world), wcs_object.invert(*world)
        )

        # Check that the low-level API rejects high-level inputs.
        high_level = wcs_object.output_frame.to_high_level_coordinates(
            *world, correct_1d=False
        )
        # Technically, we support quantities as low-level inputs, so we need to allow
        #    them to pass through
        if all(type(hlo) is u.Quantity for hlo in high_level):
            with (
                pytest.warns(EmptyFrameUnitsWarning, match=r"EmptyFrame.*")
                if isinstance(wcs_object.output_frame, EmptyFrame)
                else nullcontext()
            ):
                assert_allclose(wcs_object.world_to_pixel_values(*high_level), pixels)
        else:
            with pytest.raises(ValueError, match=r"High-Level inputs.*"):
                wcs_object.world_to_pixel_values(*high_level)

        # The gwcs_simple_imaging's inverse is not exact so we need to relax the
        #    tolerance for this test
        rtol = 0.1 if fixture_name == "gwcs_simple_imaging" else 1e-07

        # Introduce units (gwcs only supported)
        if with_units:
            world = self._get_input_as_unit(wcs_object, world)

        # Check that this never returns a Quantity, even if the input is a Quantity
        world_to_pixel = wcs_object.world_to_pixel_values(*world)
        if isinstance(world_to_pixel, tuple):
            assert not any(isinstance(p, u.Quantity) for p in world_to_pixel)
        else:
            assert not isinstance(world_to_pixel, u.Quantity)

        # Check that the values are correct
        assert_allclose(world_to_pixel, pixels, rtol=rtol)

    def test_world_to_array_index_values_array(
        self, fixture_name, wcs_object, pixels, world, request, with_units
    ):
        """
        Test that the world_to_array_index_values function returns the expected values
        with array inputs
        """
        if fixture_name == "gwcs_with_frames_strings":
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="Numerical inverse not supported when n_inputs != n_outputs"
                )
            )

        # Turn the input pixels and world into arrays
        pixels = tuple(np.ones((3, 4)) * arg for arg in pixels)[::-1]
        # Note the reverse here
        if len(pixels) == 1:
            pixels = pixels[0]

        if isinstance(world, tuple):
            world = tuple(np.ones((3, 4)) * arg for arg in world)
        else:
            world = np.ones((3, 4)) * world

        # The gwcs_simple_imaging's inverse is not exact so we need to relax the
        #    tolerance for this test
        rtol = 0.1 if fixture_name == "gwcs_simple_imaging" else 1e-07

        # Test that it matches the native API call
        if isinstance(output := wcs_object.world_to_array_index_values(*world), tuple):
            output = output[::-1]
        assert_allclose(output, wcs_object.invert(*world), rtol=rtol)

        # Check that the low-level API rejects high-level inputs.
        high_level = wcs_object.output_frame.to_high_level_coordinates(
            *world, correct_1d=False
        )
        # Technically, we support quantities as low-level inputs, so we need to allow
        #    them to pass through
        if all(type(hlo) is u.Quantity for hlo in high_level):
            with (
                pytest.warns(EmptyFrameUnitsWarning, match=r"EmptyFrame.*")
                if isinstance(wcs_object.output_frame, EmptyFrame)
                else nullcontext()
            ):
                assert_allclose(
                    wcs_object.world_to_array_index_values(*high_level), pixels
                )
        else:
            with pytest.raises(ValueError, match=r"High-Level inputs.*"):
                wcs_object.world_to_array_index_values(*high_level)

        # Introduce units (gwcs only supported)
        if with_units:
            world = self._get_input_as_unit(wcs_object, world)

        # # Check that this never returns a Quantity, even if the input is a Quantity
        world_to_array_index = wcs_object.world_to_array_index_values(*world)
        if isinstance(world_to_array_index, tuple):
            assert not any(isinstance(p, u.Quantity) for p in world_to_array_index)
        else:
            assert not isinstance(world_to_array_index, u.Quantity)

        # # Check that the values are correct
        assert_allclose(world_to_array_index, pixels)
