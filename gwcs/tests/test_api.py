# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests the API defined in astropy APE 14 (https://doi.org/10.5281/zenodo.1188875).
"""

import astropy.modeling.models as m
import astropy.units as u
import pytest
from astropy import coordinates as coord

import gwcs
import gwcs.coordinate_frames as cf


def test_composite_many_base_frame():
    q_frame_1 = cf.CoordinateFrame(
        name="distance", axes_order=(0,), naxes=1, axes_type="SPATIAL", unit=(u.m,)
    )
    q_frame_2 = cf.CoordinateFrame(
        name="distance", axes_order=(1,), naxes=1, axes_type="SPATIAL", unit=(u.m,)
    )
    frame = cf.CompositeFrame([q_frame_1, q_frame_2])

    wao_classes = frame.world_axis_object_classes

    assert len(wao_classes) == 2
    assert not set(wao_classes.keys()).difference({"SPATIAL", "SPATIAL1"})

    wao_components = frame.world_axis_object_components

    assert len(wao_components) == 2
    assert not {c[0] for c in wao_components}.difference({"SPATIAL", "SPATIAL1"})


def test_coordinate_frame_api():
    forward = m.Linear1D(slope=0.1 * u.deg / u.pix, intercept=0 * u.deg)

    output_frame = cf.CoordinateFrame(
        1, "SPATIAL", (0,), unit=(u.deg,), name="sep_frame"
    )
    input_frame = cf.CoordinateFrame(1, "PIXEL", (0,), unit=(u.pix,))

    wcs = gwcs.WCS(
        forward_transform=forward, input_frame=input_frame, output_frame=output_frame
    )

    world = wcs.pixel_to_world(0)
    assert isinstance(world, u.Quantity)

    pixel = wcs.world_to_pixel(world)
    assert isinstance(pixel, float)

    pixel2 = wcs.invert(world)
    assert u.allclose(pixel2, 0 * u.pix)


def test_mismatched_high_level_types(gwcs_3d_identity_units):
    wcs = gwcs_3d_identity_units

    with pytest.raises(
        TypeError,
        match=(
            "Invalid types were passed.*(tuple, SpectralCoord)"
            ".*(SkyCoord, SpectralCoord).*"
        ),
    ):
        wcs.invert((1 * u.deg, 2 * u.deg), coord.SpectralCoord(10 * u.nm))

    # Oh astropy why do you make us do this
    with pytest.raises(
        TypeError,
        match="Invalid types were passed.*got.*Quantity.*expected.*SpectralCoord.*",
    ):
        wcs.invert(coord.SkyCoord(1 * u.deg, 2 * u.deg), 10 * u.nm)
