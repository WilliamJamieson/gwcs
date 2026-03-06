"""
Test the high-level API functions defined in APE 14 (https://doi.org/10.5281/zenodo.1188875).
"""

from contextlib import nullcontext

import numpy as np
import pytest
from astropy import coordinates as coord
from astropy import time
from astropy import units as u
from astropy.wcs.wcsapi import HighLevelWCSWrapper
from numpy.testing import assert_allclose

from gwcs import EmptyFrame, EmptyFrameUnitsWarning


def _compare_frame_output(wc1, wc2, rtol=1e-07):
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
            _compare_frame_output(w1, w2, rtol=rtol)

    else:
        msg = f"Can't Compare {type(wc1)}"
        raise TypeError(msg)


def _pixel_to_high_level(wcs_object, pixels, correct_1d=True):
    """Convert pixel coordinates to high-level world coordinates."""
    return wcs_object.input_frame.to_high_level_coordinates(
        *pixels, correct_1d=correct_1d
    )


def _world_to_high_level(wcs_object, world, correct_1d=True):
    """Convert world coordinates to high-level world coordinates."""
    return wcs_object.output_frame.to_high_level_coordinates(
        *world, correct_1d=correct_1d
    )


@pytest.mark.parametrize("use_array", [False, True])
class TestPixToWorld:
    """Test the high-level pixel to world API functions defined in APE 14."""

    def test_pixel_to_world(self, wcs_object, pixels, world, use_array):
        """Test the pixel to world API function."""
        # Turn the input pixels and world into arrays
        if use_array:
            pixels = tuple(np.ones((3, 4)) * arg for arg in pixels)
            if len(world) == 1:
                world = np.ones((3, 4)) * world[0]
            else:
                world = tuple(np.ones((3, 4)) * arg for arg in world)

        # Turn world into high-level world coordinates
        world = _world_to_high_level(wcs_object, world)

        # Check that we can pass low-level pixel coordinates into this without
        #   any issues and get the high-level world coordinates out
        # Note this works only because APE 14 does not really describe any
        #    non-quantity high-level pixel coordinates.
        _compare_frame_output(wcs_object.pixel_to_world(*pixels), world)

        # Check that we can also get high-level world coordinates matching the
        #   high-level pixel coordinates
        _compare_frame_output(wcs_object(*pixels, force_high_level=True), world)

        # Check that we can also pass high-level pixel coordinates into this and
        #   get the same high-level world coordinates out
        with (
            pytest.warns(EmptyFrameUnitsWarning, match=r"EmptyFrame.*")
            if isinstance(wcs_object.input_frame, EmptyFrame)
            else nullcontext()
        ):
            _compare_frame_output(
                wcs_object.pixel_to_world(
                    *_pixel_to_high_level(wcs_object, pixels, correct_1d=False)
                ),
                world,
            )

    def test_array_index_to_world(self, wcs_object, pixels, world, use_array):
        """Test the array index to world API function."""
        # Turn the input pixels and world into arrays
        if use_array:
            pixels = tuple(np.ones((3, 4)) * arg for arg in pixels)
            if len(world) == 1:
                world = np.ones((3, 4)) * world[0]
            else:
                world = tuple(np.ones((3, 4)) * arg for arg in world)

        # Turn world into high-level world coordinates
        world = _world_to_high_level(wcs_object, world)

        # Check that we can pass low-level pixel coordinates into this without
        #   any issues and get the high-level world coordinates out
        # Note this works only because APE 14 does not really describe any
        #    non-quantity high-level pixel coordinates.
        _compare_frame_output(wcs_object.array_index_to_world(*pixels[::-1]), world)

        # Check that we can also pass high-level pixel coordinates into this and
        #   get the same high-level world coordinates out
        with (
            pytest.warns(EmptyFrameUnitsWarning, match=r"EmptyFrame.*")
            if isinstance(wcs_object.input_frame, EmptyFrame)
            else nullcontext()
        ):
            _compare_frame_output(
                wcs_object.array_index_to_world(
                    *_pixel_to_high_level(wcs_object, pixels, correct_1d=False)[::-1]
                ),
                world,
            )

    def test_high_level_wrapper(
        self, fixture_name, wcs_object, pixels, world, use_array, request
    ):
        """Test the high-level wrapper."""
        if fixture_name == "gwcs_high_level_pixel":
            msg = (
                "The `astropy.wcs` wrapper does not support wcs that are not"
                " purely pixel in the input_frame."
            )
            request.node.add_marker(pytest.mark.xfail(reason=msg))

        hlvl = HighLevelWCSWrapper(wcs_object)

        # Turn the input pixels and world into arrays
        if use_array:
            pixels = tuple(np.ones((3, 4)) * arg for arg in pixels)
            if len(world) == 1:
                world = np.ones((3, 4)) * world[0]
            else:
                world = tuple(np.ones((3, 4)) * arg for arg in world)

        # Turn world into high-level world coordinates
        world = _world_to_high_level(wcs_object, world)

        # Check that we can pass low-level pixel coordinates into this without
        #   any issues and get the high-level world coordinates out
        _compare_frame_output(hlvl.pixel_to_world(*pixels), world)

        # Check that we can also pass high-level pixel coordinates into this and
        #   get the same high-level world coordinates out
        with (
            pytest.warns(EmptyFrameUnitsWarning, match=r"EmptyFrame.*")
            if isinstance(wcs_object.input_frame, EmptyFrame)
            else nullcontext()
        ):
            # Note this is where the xfail occurs
            _compare_frame_output(
                hlvl.pixel_to_world(
                    *_pixel_to_high_level(wcs_object, pixels, correct_1d=False)
                ),
                world,
            )


@pytest.mark.parametrize("use_array", [False, True])
class TestWorldToPixel:
    """Test the high-level world to pixel API functions defined in APE 14."""

    def test_world_to_pixel(
        self, fixture_name, wcs_object, pixels, world, use_array, request
    ):
        """Test the world to pixel API function."""
        if fixture_name == "gwcs_with_frames_strings":
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="Numerical inverse not supported when n_inputs != n_outputs"
                )
            )
        # Turn the input pixels and world into arrays
        if use_array:
            pixels = tuple(np.ones((3, 4)) * arg for arg in pixels)
            if len(pixels) == 1:
                pixels = pixels[0]

            if isinstance(world, tuple):
                world = tuple(np.ones((3, 4)) * arg for arg in world)
            else:
                world = np.ones((3, 4)) * world

        # Test that we cannot pass low-level world coordinates into this
        with pytest.raises(ValueError, match=r".*"):
            wcs_object.world_to_pixel(*world)

        # The gwcs_simple_imaging's inverse is not exact so we need to relax the
        #    tolerance for this test
        rtol = 0.1 if fixture_name == "gwcs_simple_imaging" else 1e-07

        # Check that we can also pass high-level world coordinates into this and
        #   get the same low-level pixel coordinates out
        # Note that a simple read of APE 14 might make it seem like we should get
        #   high-level (Quantity) pixel coordinates out here, but the APE 14 API
        #   but it is not quite stated that there are no high-level like objects
        #   for pixel coordinates, so it will just return simple arrays here.
        # This is consistent with `astropy.wcs` so it is what we have to live with.
        _compare_frame_output(
            wcs_object.world_to_pixel(
                *_world_to_high_level(wcs_object, world, correct_1d=False)
            ),
            pixels,
            rtol=rtol,
        )

        # Turn pixel into high-level pixel coordinates
        pixels = _pixel_to_high_level(wcs_object, pixels)

        # Check that we can also get low-level pixel coordinates matching the
        #   high-level world coordinates
        _compare_frame_output(
            wcs_object.invert(*world, force_high_level=True), pixels, rtol=rtol
        )

    def test_world_to_array_index(
        self, fixture_name, wcs_object, pixels, world, use_array, request
    ):
        """Test the world to array index API function."""
        if fixture_name == "gwcs_with_frames_strings":
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="Numerical inverse not supported when n_inputs != n_outputs"
                )
            )
        # Turn the input pixels and world into arrays
        if use_array:
            pixels = tuple(np.ones((3, 4)) * arg for arg in pixels)
            if len(pixels) == 1:
                pixels = pixels[0]

            if isinstance(world, tuple):
                world = tuple(np.ones((3, 4)) * arg for arg in world)
            else:
                world = np.ones((3, 4)) * world

        # Test that we cannot pass low-level world coordinates into this
        with pytest.raises(ValueError, match=r".*"):
            wcs_object.world_to_array_index(*world)

        # The gwcs_simple_imaging's inverse is not exact so we need to relax the
        #    tolerance for this test
        rtol = 0.1 if fixture_name == "gwcs_simple_imaging" else 1e-07

        # Check that we can also pass high-level world coordinates into this and
        #   get the same low-level pixel coordinates out
        # See note in the world_to_pixel test about the expected output here.
        _compare_frame_output(
            wcs_object.world_to_array_index(
                *_world_to_high_level(wcs_object, world, correct_1d=False)
            ),
            pixels[::-1],
            rtol=rtol,
        )

    def test_high_level_wrapper(
        self, fixture_name, wcs_object, pixels, world, use_array, request
    ):
        """Test the high-level wrapper."""
        if fixture_name == "gwcs_with_frames_strings":
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="Numerical inverse not supported when n_inputs != n_outputs"
                )
            )

        hlvl = HighLevelWCSWrapper(wcs_object)

        # Turn the input pixels and world into arrays
        if use_array:
            pixels = tuple(np.ones((3, 4)) * arg for arg in pixels)
            if len(pixels) == 1:
                pixels = pixels[0]

            if isinstance(world, tuple):
                world = tuple(np.ones((3, 4)) * arg for arg in world)
            else:
                world = np.ones((3, 4)) * world

        # The gwcs_simple_imaging's inverse is not exact so we need to relax the
        #    tolerance for this test
        rtol = 0.1 if fixture_name == "gwcs_simple_imaging" else 1e-07

        # Turn world into high-level world coordinates
        world = _world_to_high_level(wcs_object, world, correct_1d=False)

        # Check that we can pass high-level world coordinates into this and get the
        #     same low-level pixel coordinates out
        _compare_frame_output(hlvl.world_to_pixel(*world), pixels, rtol=rtol)


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
