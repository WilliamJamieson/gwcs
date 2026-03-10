"""
Tests the native API for GWCS.

The WCS functions that are considered part of the native API are:
- evaluate/__call__
- invert

Rule of thumb for the native API:
    kind of input returns same kind of output.

In practice this looks a little more complicated due to the difficulties around
the definition of what a "high-level" object is. In APE 14, this is relatively
simple (for world coordinates) as a "high-level" object is one that has additional
metadata or structure beyond a simple array. So
- A Quantity is a high-level object because it has units attached to it.
- A SkyCoord is a high-level object because it has additional information about
    the coordinate system.

For GWCS, we have a little bit more complexity because we natively support Quantities
throughout the API, rather than treading them as "high-level" objects. This is because
Quantities are essentially just arrays with units attached and are designed to be usable
wherever arrays are used (though this is not always perfectly true in practice).
So for GWCS, we really have three levels of input/output:
- Low-level: purely numerical inputs/outputs (e.g. arrays, lists, tuples, scalars)
- Quantity: inputs/outputs that are Quantities
- High-level: inputs/outputs that are not Quantities but have additional metadata
    (e.g. SkyCoord, Time, etc.)

GWCS is also bound by the limitations of the input and output frames of the WCS.
These frames determine what level of input and output we can actually support. Namely,
if one looks at the ``frame.world_axis_object_classes`` attribute, the first element
of each of the tuples lists the "high-level" object in APE 14 terms that the frame
supports. Often this is a Quantity, but it could also be a High-level object in the
GWCS sense like SkyCoord. A tricky situation arises when the output frame supports
a GWCS high-level object but the input frame only supports a Quantity (or the reverse
situation). In this case, it is impossible to infer from the user's input to `evaluate`
(or `invert`) whether they want a Quantity or a high-level object as output when they
input a Quantity, so we default to returning a Quantity in this case. The
``force_high_level`` keyword is provided to allow users to force a high-level output
in this case.

This leads to the following rules of the native API for GWCS:
1. If the input is low-level, we return low-level output.
2. If the input is a Quantity, we return a Quantity output
3. If the input is a high-level object, we return:
    - a high-level output if the corresponding frame supports a GWCS high-level object
    - a Quantity output if the corresponding frame only supports a Quantity.

There is slightly more complexity around all of this because the actual transform
used may support Quantities, require Quantities, or not support Quantities at all.
Furthermore transforms do not natively support the GWCS high-level objects. This
means we have the following control flow for the native API:

1. If the input is a GWCS "high-level" object, we use the associated frame's
    `from_high_level_coordinates` method to convert the "high-level" input into
    a low-level input that the transform can understand. If the input is not
    a GWCS "high-level" object, we use the input as-is. During this step
    we record whether the input was a "high-level" object or not, which we will
    use later to manipulate the output to be the same "level" as the input.
2. We then determine if the input at this point is a "Quantity" or "low-level" input
    and record this as well. We will use this information later to manipulate the
    output to be the same "level" as the input.
3. Next we determine if the transform uses quantities. Here we use the `uses_quantities`
    attribute of the transform to determine this. Note that this attribute is not
    perfect as it defaults to a heuristic in Astropy to guess this, so we have some
    special handling for common cases where this heuristic fails (e.g. Tabular
    transforms).
4a. If the transform uses quantities, then we manipulate the input to insure we
    have a quantity with the correct units. At this point the "input" is either
    a "low-level" input or a Quantity, so we process this by
    - If the input is low-level, we assume the units of the input are the units
        of the frame and so we simply attach the associated units to the input
        to make it a Quantity.
    - If the input is already a Quantity, we attempt to convert it to the units
        of the input frame. (if the units are not convertible a unit conversion
        error is raised)
4b. If the transform does not use quantities, then we manipulate the input to insure
    that it becomes a "low-level" input. So this is processed as follows:
    - If the input is low-level, we use it as-is.
    - If the input is a Quantity, we first convert it to the units of the frame
        and then we take the value of the new Quantity to get a low-level input.
5. We then use the input at this point to call the transform.
6a. If the input was a "high-level" object OR ``force_high_level`` is True, then
    we use the output frame's `to_high_level_coordinates` method to convert the
    output from the transform into a "high-level" output (which may just be a
    Quantity) and return the result.
6b. If the input was not a "high-level" object but was a Quantity, then
    - If the output from the transform is not a Quantity, we attach the units of
        the output frame to make it a Quantity and return the result.
    - If the output from the transform is already a Quantity, we attempt to convert
        it to the units of the output frame and return the result. (if the units are
        not convertible a unit conversion error is raised)
6c. If the input was not a "high-level" object and was not a Quantity, then
    - If the output from the transform is a Quantity, we convert that output to
        the units of the output frame and then return the resulting value as a
        low-level output.
    - If the output from the transform is not a Quantity, we return it as-is as a
        low-level output.

Strings as Frames (i.e. ``EmptyFrame``) create strange behavior in this case because
they do not have any information about the units or level of the inputs/outputs they
support. For reasonable support of these frames, we assume they use Quantities as their
"high-level" objects, but their units are "dimensionless" (i.e. they have no units).
This however means they maynot interact properly with the Quantities or high-level
objects and so they will raise warnings when called upon to do something with these
inputs/outputs.

Note that if the units for a frame are None (e.g. ``EmptyFrame``), then when a
frame is called to add units (which maybe just a conversion) to an input nothing
is changed about the input and it is returned as-is. So we may get a Quantity or
a low-level object back in this case.

Note that if a frame is called to remove units from some input then it first adds
units to the input (which is to force unit conversions) before it then strips the
units off the result to return a low-level object. In the units being None case,
this means that we simply remove the units from the input if there are any.
"""

import pytest
from numpy.testing import assert_allclose

from gwcs.coordinate_frames import EmptyFrame
from gwcs.wcs import WCS

from .conftest import (
    check_is_gwcs_high_level,
    check_is_high_level,
    check_is_low_level,
    compare_frame_output,
    empty_frame_warning_context,
    is_dimensionless_quantity,
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

        check_is_low_level(wcs_object.output_frame, world)
        check_is_low_level(wcs_object.output_frame, world_output)

        assert_allclose(world_output, world)

    # If the pixel is a true (non-quantity) high-level object (e.g. a SkyCoord),
    #   Then we should get a true high-level output (e.g. a SkyCoord) or a Quantity
    elif wcs_object.input_frame.is_high_level(*pixel):
        # Note the output maybe a Quantity or a true high-level object depending
        #    on the output frame, so we check both cases here.
        if is_quantity(world_output):
            assert is_quantity(world)

        else:
            check_is_gwcs_high_level(wcs_object.output_frame, world)
            check_is_gwcs_high_level(wcs_object.output_frame, world_output)

        compare_frame_output(world_output, world)

    # If the pixel is a quantity, then (in theory) we should get a quantity out.
    elif is_quantity(pixel):
        # However, if the output frame is an EmptyFrame, then we should get a
        #   low-level, simply because the EmptyFrame cannot convert something into
        #   a quantity.
        if isinstance(wcs_object.output_frame, EmptyFrame):
            # In order to get a world is a quantity and the output frame
            #  is an empty frame, then we must be in the high-level input case
            if is_quantity(world):
                assert is_dimensionless_quantity(world)
                assert level == "high"

            # If the world is not a quantity, but the output frame is an EmptyFrame,
            #   then we must be in the quantity input case, where we have to return
            #   a low-level output.
            # Currently we don't have any fixtures that generate this case
            else:
                check_is_low_level(wcs_object.output_frame, world)
                assert level == "quantity"

            check_is_low_level(wcs_object.output_frame, world_output)
            assert_allclose(world_output, world)

        # Otherwise, we should get a quantity out.
        else:
            # If world is not a quantity, then it should be a true gwcs high-level
            if not is_quantity(world):
                check_is_gwcs_high_level(wcs_object.output_frame, world)
                assert level == "high"

            # The output should never be a high-level object
            assert is_quantity(world_output)
            with pytest.raises(AssertionError, match=r".*"):
                check_is_gwcs_high_level(wcs_object.output_frame, world_output)

            # world maybe a true high-level object, so we should compare the output
            #    to the quantity version because the level-should match the
            #    level of the input.
            compare_frame_output(world_output, world_quantity)

    # Otherwise, we have the case where the input_frame was an EmptyFrame, so
    #    we couldn't generate a non-low-level input in the fixture in the quantity
    #    level
    else:
        assert isinstance(wcs_object.input_frame, EmptyFrame)
        assert level == "quantity"

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

    # The output should always be a APE 14 "high-level" object
    check_is_high_level(wcs_object.output_frame, world)
    check_is_high_level(wcs_object.output_frame, world_output)

    # If the output is not a Quantity then it should be a true high-level object
    #     in the GWCS sense (e.g. a SkyCoord)
    if not is_quantity(world_output):
        check_is_gwcs_high_level(wcs_object.output_frame, world)
        check_is_gwcs_high_level(wcs_object.output_frame, world_output)
    else:
        # World is a quantity, and so not gwcs high-level
        is_quantity(world)

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

        check_is_low_level(wcs_object.input_frame, pixel)
        check_is_low_level(wcs_object.input_frame, pixel_output)

        assert_allclose(pixel_output, pixel, rtol=rtol)

    # If the world is a true (non-quantity) high-level object (e.g. a SkyCoord),
    #   Then we should get a true high-level output (e.g. a SkyCoord)
    elif wcs_object.output_frame.is_high_level(*world):
        # Note the output maybe a Quantity or a true high-level object depending
        #    on the input frame, so we check both cases here.
        if is_quantity(pixel_output):
            assert is_quantity(pixel)

        else:
            check_is_gwcs_high_level(wcs_object.input_frame, pixel)
            check_is_gwcs_high_level(wcs_object.input_frame, pixel_output)

        compare_frame_output(pixel_output, pixel, rtol=rtol)

    # If the world is a quantity, then (in theory) we should get a quantity out.
    elif is_quantity(world):
        # However, if the output frame is an EmptyFrame, then we should get a
        #   low-level, simply because the EmptyFrame cannot convert something into
        #   a quantity.
        if isinstance(wcs_object.input_frame, EmptyFrame):
            # In order to get a pixel is a quantity and the input frame
            #  is an empty frame, then we must be in the high-level input case
            if is_quantity(pixel):
                assert is_dimensionless_quantity(pixel)
                assert level == "high"

            # If the pixel is not a quantity, but the input frame is an EmptyFrame,
            #   then we must be in the quantity input case, where we have to return
            #   a low-level output.
            # Currently we don't have any fixtures that generate this case
            else:
                check_is_low_level(wcs_object.input_frame, pixel)
                assert level == "quantity"

            check_is_low_level(wcs_object.input_frame, pixel_output)
            assert_allclose(pixel_output, pixel)

        # Otherwise, we should get a quantity out.
        else:
            # If pixel is not a quantity, then it should be a true gwcs high-level
            #   object
            if not is_quantity(pixel):
                check_is_gwcs_high_level(wcs_object.input_frame, pixel)
                assert level == "high"

            # The output should never be a high-level object
            assert is_quantity(pixel_output)
            with pytest.raises(AssertionError, match=r".*"):
                check_is_gwcs_high_level(wcs_object.input_frame, pixel_output)

            # world maybe a true high-level object, so we should compare the output
            #    to the quantity version because the level-should match the
            #    level of the input.
            compare_frame_output(pixel_output, pixel_quantity, rtol=rtol)

    # Otherwise, we have the case where the input_frame was an EmptyFrame, so
    #    we couldn't generate a non-low-level input in the fixture in the quantity
    #    level
    else:
        assert isinstance(wcs_object.output_frame, EmptyFrame)
        assert level == "quantity"

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

    # The output should always be a APE 14 "high-level" object
    check_is_high_level(wcs_object.input_frame, pixel)
    check_is_high_level(wcs_object.input_frame, pixel_output)

    # If the output is not a Quantity then it should be a true high-level object
    #     in the GWCS sense (e.g. a SkyCoord)
    if not is_quantity(pixel_output):
        check_is_gwcs_high_level(wcs_object.input_frame, pixel)
        check_is_gwcs_high_level(wcs_object.input_frame, pixel_output)
    else:
        # Pixel is a quantity, and so not gwcs high-level
        assert is_quantity(pixel)

    compare_frame_output(pixel_output, pixel, rtol=rtol)
