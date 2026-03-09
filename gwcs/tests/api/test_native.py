"""
Tests the native API for GWCS.
"""

import pytest
from numpy.testing import assert_allclose

from gwcs.coordinate_frames import EmptyFrame
from gwcs.wcs import WCS

from .conftest import (
    check_is_high_level,
    check_is_low_level,
    compare_frame_output,
    empty_frame_warning_context,
    is_quantity,
)


def test_evaluate(wcs_object: WCS, pixel, world, world_quantity, level):
    """Test the evaluate/__call__ API for GWCS's Native API"""

    # For 1D world coordinates, the fixture will return a tuple, but we want to
    #   have a single value here
    if isinstance(world, tuple | list) and len(world) == 1:
        world = world[0]
    with empty_frame_warning_context(wcs_object.output_frame, pixel):
        world_output = wcs_object.evaluate(*pixel)

    ## Check the input/output level is correct ##
    # If the level is low, we should have both a low-level input and output from
    #   evaluate.
    if level == "low":
        check_is_low_level(wcs_object.input_frame, pixel)
        check_is_low_level(wcs_object.output_frame, world_output)

        assert_allclose(world_output, world)

    # If the pixel is a true (non-quantity) high-level object (e.g. a SkyCoord),
    #   Then we should get a true high-level output (e.g. a SkyCoord)
    # Note there should be another case under this where the output frame does
    #   not support true high-level objects.
    elif wcs_object.input_frame.is_high_level(*pixel):
        check_is_high_level(wcs_object.input_frame, pixel)
        check_is_high_level(wcs_object.output_frame, world_output)

        compare_frame_output(world_output, world)

    # If the pixel is a quantity, then (in theory) we should get a quantity out.
    elif is_quantity(pixel):
        # However, if the output frame is an EmptyFrame, then we should get a
        #   low-level, simply because the EmptyFrame cannot convert something into
        #   a quantity.
        if isinstance(wcs_object.output_frame, EmptyFrame):
            check_is_low_level(wcs_object.output_frame, world_output)
            assert_allclose(world_output, world)

        # Otherwise, we should get a quantity out.
        else:
            assert is_quantity(world_output)

            # world maybe a true high-level object, so we should compare the output
            #    to the quantity version because the level-should match the
            #    level of the input.
            compare_frame_output(world_output, world_quantity)

    # Otherwise, we have the case where the input_frame was an EmptyFrame, so
    #    we couldn't generate a non-low-level input in the fixture
    else:
        assert isinstance(wcs_object.input_frame, EmptyFrame)
        check_is_low_level(wcs_object.input_frame, pixel)
        check_is_low_level(wcs_object.output_frame, world_output)

        assert_allclose(world_output, world)

    # Check that the __call__ and evaluate give the same result
    with empty_frame_warning_context(wcs_object.output_frame, pixel):
        world_call = wcs_object(*pixel)

    compare_frame_output(world_call, world_output)


def test_evaluate_high_level(wcs_object: WCS, pixel, world_high):
    """
    Test that if we force a high-level output, we match the high-level version of
        the world coordinates.
    """
    # The force_high_level=True always returns a high-level world coordinate,
    #    which is the world_high fixture
    world = world_high

    if isinstance(world, tuple | list) and len(world) == 1:
        world = world[0]

    with empty_frame_warning_context(wcs_object.output_frame, pixel):
        world_output = wcs_object.evaluate(*pixel, force_high_level=True)

    # Some output frames may have their "high-level" object as a quantity
    if not is_quantity(world_output):
        check_is_high_level(wcs_object.output_frame, world_output)
    else:
        assert is_quantity(world_output)

    compare_frame_output(world_output, world)

    # Check that the __call__ and evaluate give the same result
    with empty_frame_warning_context(wcs_object.output_frame, pixel):
        world_call = wcs_object(*pixel, force_high_level=True)

    compare_frame_output(world_call, world_output)


@pytest.mark.usefixtures("xfail_gwcs_with_frames_strings")
def test_invert(wcs_object: WCS, pixel, pixel_quantity, world, level, rtol):
    """Test that the invert API correctly inverts the transformation."""
    # For 1D pixels coordinates, the fixture will return a tuple, but we want to
    #   have a single value here
    if isinstance(pixel, tuple | list) and len(pixel) == 1:
        pixel = pixel[0]
    with empty_frame_warning_context(wcs_object.input_frame, world):
        pixel_output = wcs_object.invert(*world)

    ## Check the input/output level is correct ##
    # If the level is low, we should have both a low-level input and output from
    #   evaluate.
    if level == "low":
        check_is_low_level(wcs_object.output_frame, world)
        check_is_low_level(wcs_object.input_frame, pixel_output)

        assert_allclose(pixel_output, pixel, rtol=rtol)

    # If the pixel is a true (non-quantity) high-level object (e.g. a SkyCoord),
    #   Then we should get a true high-level output (e.g. a SkyCoord)
    # Note there should be another case under this where the output frame does
    #   not support true high-level objects.
    elif wcs_object.output_frame.is_high_level(*world):
        check_is_high_level(wcs_object.output_frame, world)
        check_is_high_level(wcs_object.input_frame, pixel_output)

        compare_frame_output(pixel_output, pixel, rtol=rtol)

    # If the pixel is a quantity, then (in theory) we should get a quantity out.
    elif is_quantity(world):
        # However, if the output frame is an EmptyFrame, then we should get a
        #   low-level, simply because the EmptyFrame cannot convert something into
        #   a quantity.
        if isinstance(wcs_object.input_frame, EmptyFrame):
            check_is_low_level(wcs_object.input_frame, pixel_output)
            assert_allclose(pixel_output, pixel)

        # Otherwise, we should get a quantity out.
        else:
            assert is_quantity(pixel_output)

            # world maybe a true high-level object, so we should compare the output
            #    to the quantity version because the level-should match the
            #    level of the input.
            compare_frame_output(pixel_output, pixel_quantity, rtol=rtol)

    # Otherwise, we have the case where the input_frame was an EmptyFrame, so
    #    we couldn't generate a non-low-level input in the fixture
    else:
        assert isinstance(wcs_object.output_frame, EmptyFrame)
        check_is_low_level(wcs_object.output_frame, world)
        check_is_low_level(wcs_object.input_frame, pixel_output)

        assert_allclose(pixel_output, pixel)


@pytest.mark.usefixtures("xfail_gwcs_with_frames_strings")
def test_invert_high_level(wcs_object: WCS, pixel_high, world, rtol):
    """
    Test that if we force a high-level output, we match the high-level version of
        the pixel coordinates.
    """
    # The force_high_level=True always returns a high-level pixel coordinate,
    #    which is the pixel_high fixture
    pixel = pixel_high

    if isinstance(pixel, tuple | list) and len(pixel) == 1:
        pixel = pixel[0]

    with empty_frame_warning_context(wcs_object.input_frame, world):
        pixel_output = wcs_object.invert(*world, force_high_level=True)

    # Some output frames may have their "high-level" object as a quantity
    if not is_quantity(pixel_output):
        check_is_high_level(wcs_object.input_frame, pixel_output)
    else:
        assert is_quantity(pixel_output)

    compare_frame_output(pixel_output, pixel, rtol=rtol)
