"""
Test the API properties defined in APE 14 (https://doi.org/10.5281/zenodo.1188875).
"""

import numpy as np
import pytest
from astropy import coordinates as coord
from astropy import time
from astropy import units as u
from numpy.testing import assert_array_equal

from gwcs import coordinate_frames as cf

RNG = np.random.default_rng(42)


@pytest.fixture
def wcs_ndim(fixture_name):
    return {
        "gwcs_2d_spatial_shift": (2, 2),
        "gwcs_2d_spatial_reordered": (2, 2),
        "gwcs_2d_quantity_shift": (2, 2),
        "gwcs_1d_freq": (1, 1),
        "gwcs_3d_spatial_wave": (3, 3),
        "gwcs_3d_identity_units": (3, 3),
        "gwcs_4d_identity_units": (4, 4),
        "gwcs_stokes_lookup": (1, 1),
        "gwcs_3d_galactic_spectral": (3, 3),
        "gwcs_2d_shift_scale": (2, 2),
        "gwcs_2d_shift_scale_quantity": (2, 2),
        "gwcs_1d_freq_quantity": (1, 1),
        "gwcs_simple_2d": (2, 2),
        "gwcs_empty_output_2d": (2, 2),
        "gwcs_simple_imaging": (2, 2),
        "gwcs_with_frames_strings": (4, 3),
        "gwcs_high_level_pixel": (2, 2),
    }[fixture_name]


def test_pixel_n_dim(wcs_object, wcs_ndim):
    """Test the pixel_n_dim returns the number of pixel axes."""
    assert wcs_object.pixel_n_dim == wcs_ndim[0]


def test_world_n_dim(wcs_object, wcs_ndim):
    """Test the world_n_dim returns the number of world axes."""
    assert wcs_object.world_n_dim == wcs_ndim[1]


@pytest.fixture
def wcs_shape(fixture_name):
    return {
        "gwcs_2d_spatial_shift": None,
        "gwcs_2d_spatial_reordered": None,
        "gwcs_2d_quantity_shift": None,
        "gwcs_1d_freq": None,
        "gwcs_3d_spatial_wave": None,
        "gwcs_3d_identity_units": None,
        "gwcs_4d_identity_units": None,
        "gwcs_stokes_lookup": None,
        "gwcs_3d_galactic_spectral": (30, 20, 10),
        "gwcs_2d_shift_scale": None,
        "gwcs_2d_shift_scale_quantity": None,
        "gwcs_1d_freq_quantity": None,
        "gwcs_simple_2d": None,
        "gwcs_empty_output_2d": None,
        "gwcs_simple_imaging": None,
        "gwcs_with_frames_strings": None,
        "gwcs_high_level_pixel": None,
    }[fixture_name]


def test_array_shape_and_pixel_shape(wcs_object, wcs_shape):
    """Test the array_shape and pixel_shape properties and how they relate."""
    assert wcs_object.array_shape == wcs_shape

    array_shape = tuple(RNG.integers(1020, 4096, size=wcs_object.pixel_n_dim))

    wcs_object.array_shape = array_shape
    assert_array_equal(wcs_object.array_shape, array_shape)

    assert wcs_object.array_shape == wcs_object.pixel_shape[::-1]

    wcs_object.array_shape = wcs_shape
    pixel_shape = tuple(RNG.integers(1020, 4096, size=wcs_object.pixel_n_dim))
    assert not (np.array(pixel_shape) == np.array(array_shape)).all()

    wcs_object.pixel_shape = pixel_shape
    assert wcs_object.array_shape == pixel_shape[::-1]

    # Fix for future tests
    wcs_object.array_shape = wcs_shape


@pytest.fixture
def wcs_pixel_bounds(fixture_name):
    return {
        "gwcs_2d_spatial_shift": None,
        "gwcs_2d_spatial_reordered": None,
        "gwcs_2d_quantity_shift": None,
        "gwcs_1d_freq": None,
        "gwcs_3d_spatial_wave": None,
        "gwcs_3d_identity_units": None,
        "gwcs_4d_identity_units": None,
        "gwcs_stokes_lookup": ((0, 3),),
        "gwcs_3d_galactic_spectral": ((-1, 35), (-2, 45), (5, 50)),
        "gwcs_2d_shift_scale": None,
        "gwcs_2d_shift_scale_quantity": None,
        "gwcs_1d_freq_quantity": None,
        "gwcs_simple_2d": None,
        "gwcs_empty_output_2d": None,
        "gwcs_simple_imaging": None,
        "gwcs_with_frames_strings": None,
        "gwcs_high_level_pixel": None,
    }[fixture_name]


def test_pixel_bounds(wcs_object, wcs_pixel_bounds):
    """Test the pixel_bounds property is the bounding box."""
    assert wcs_object.pixel_bounds == wcs_pixel_bounds

    bbox = tuple((-0.5, RNG.uniform(1020, 4096)) for _ in range(wcs_object.pixel_n_dim))

    wcs_object.bounding_box = bbox
    assert_array_equal(wcs_object.pixel_bounds, wcs_object.bounding_box)
    # Reset the bounding box or this will affect other tests
    wcs_object.bounding_box = wcs_pixel_bounds


@pytest.fixture
def wcs_types(fixture_name):
    return {
        "gwcs_2d_spatial_shift": ("pos.eq.ra", "pos.eq.dec"),
        "gwcs_2d_spatial_reordered": ("pos.eq.dec", "pos.eq.ra"),
        "gwcs_2d_quantity_shift": ("custom:SPATIAL", "custom:SPATIAL"),
        "gwcs_1d_freq": ("em.freq",),
        "gwcs_3d_spatial_wave": ("pos.eq.ra", "pos.eq.dec", "em.wl"),
        "gwcs_3d_identity_units": ("pos.eq.ra", "pos.eq.dec", "em.wl"),
        "gwcs_4d_identity_units": ("pos.eq.ra", "pos.eq.dec", "em.wl", "time"),
        "gwcs_stokes_lookup": ("phys.polarization.stokes",),
        "gwcs_3d_galactic_spectral": (
            "pos.galactic.lat",
            "em.freq",
            "pos.galactic.lon",
        ),
        "gwcs_2d_shift_scale": ("pos.eq.ra", "pos.eq.dec"),
        "gwcs_2d_shift_scale_quantity": ("pos.eq.ra", "pos.eq.dec"),
        "gwcs_1d_freq_quantity": ("em.freq",),
        "gwcs_simple_2d": ("custom:x", "custom:y"),
        "gwcs_empty_output_2d": ("custom:UNKNOWN", "custom:UNKNOWN"),
        "gwcs_simple_imaging": ("pos.eq.ra", "pos.eq.dec"),
        "gwcs_with_frames_strings": (
            "custom:UNKNOWN",
            "custom:UNKNOWN",
            "custom:UNKNOWN",
        ),
        "gwcs_high_level_pixel": ("pos.eq.ra", "pos.eq.dec"),
    }[fixture_name]


def test_world_axis_physical_types(wcs_object, wcs_types):
    """Test the world_axis_physical_types property returns the physical types."""
    assert wcs_object.world_axis_physical_types == wcs_types


@pytest.fixture
def wcs_units(fixture_name):
    return {
        "gwcs_2d_spatial_shift": ("deg", "deg"),
        "gwcs_2d_spatial_reordered": ("deg", "deg"),
        "gwcs_2d_quantity_shift": ("km", "km"),
        "gwcs_1d_freq": ("Hz",),
        "gwcs_3d_spatial_wave": ("deg", "deg", "m"),
        "gwcs_3d_identity_units": ("arcsec", "arcsec", "nm"),
        "gwcs_4d_identity_units": ("deg", "deg", "nm", "s"),
        "gwcs_stokes_lookup": ("",),
        "gwcs_3d_galactic_spectral": ("deg", "Hz", "deg"),
        "gwcs_2d_shift_scale": ("deg", "deg"),
        "gwcs_2d_shift_scale_quantity": ("deg", "deg"),
        "gwcs_1d_freq_quantity": ("Hz",),
        "gwcs_simple_2d": ("pixel", "pixel"),
        "gwcs_empty_output_2d": ("None", "None"),
        "gwcs_simple_imaging": ("deg", "deg"),
        "gwcs_with_frames_strings": ("None", "None", "None"),
        "gwcs_high_level_pixel": ("deg", "deg"),
    }[fixture_name]


def test_world_axis_units(wcs_object, wcs_units):
    """Test the world_axis_units property returns the units."""
    assert wcs_object.world_axis_units == wcs_units


@pytest.fixture
def wcs_axis_correlation_matrix(fixture_name):
    return {
        "gwcs_2d_spatial_shift": np.array(
            [
                [True, False],
                [False, True],
            ]
        ),
        "gwcs_2d_spatial_reordered": np.array(
            [
                [False, True],
                [True, False],
            ]
        ),
        "gwcs_2d_quantity_shift": np.array(
            [
                [True, False],
                [False, True],
            ]
        ),
        "gwcs_1d_freq": np.array([[True]]),
        "gwcs_3d_spatial_wave": np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
            ]
        ),
        "gwcs_3d_identity_units": np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
            ]
        ),
        "gwcs_4d_identity_units": np.array(
            [
                [True, False, False, False],
                [False, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
            ]
        ),
        "gwcs_stokes_lookup": np.array([[True]]),
        "gwcs_3d_galactic_spectral": np.array(
            [
                [True, False, True],
                [False, True, False],
                [True, False, True],
            ]
        ),
        "gwcs_2d_shift_scale": np.array(
            [
                [True, False],
                [False, True],
            ]
        ),
        "gwcs_2d_shift_scale_quantity": np.array(
            [
                [True, False],
                [False, True],
            ]
        ),
        "gwcs_1d_freq_quantity": np.array([[True]]),
        "gwcs_simple_2d": np.array(
            [
                [True, False],
                [False, True],
            ]
        ),
        "gwcs_empty_output_2d": np.array(
            [
                [True, False],
                [False, True],
            ]
        ),
        "gwcs_simple_imaging": np.array(
            [
                [True, True],
                [True, True],
            ]
        ),
        "gwcs_with_frames_strings": np.array(
            [
                [True, False, False, False],
                [False, True, False, False],
                [False, False, True, True],
            ]
        ),
        "gwcs_high_level_pixel": np.array(
            [
                [True, False],
                [False, True],
            ]
        ),
    }[fixture_name]


def test_axis_correlation_matrix(wcs_object, wcs_axis_correlation_matrix):
    """Test the axis_correlation_matrix property returns the correct matrix."""

    assert_array_equal(wcs_object.axis_correlation_matrix, wcs_axis_correlation_matrix)


def test_serialized_classes(wcs_object):
    """Test the serialized_classes property returns `False`."""
    assert wcs_object.serialized_classes is False


@pytest.fixture
def wcs_component_names(fixture_name):
    return {
        "gwcs_2d_spatial_shift": ("celestial", "celestial"),
        "gwcs_2d_spatial_reordered": ("celestial", "celestial"),
        "gwcs_2d_quantity_shift": ("SPATIAL", "SPATIAL1"),
        "gwcs_1d_freq": ("spectral",),
        "gwcs_3d_spatial_wave": ("celestial", "celestial", "spectral"),
        "gwcs_3d_identity_units": ("celestial", "celestial", "spectral"),
        "gwcs_4d_identity_units": ("celestial", "celestial", "spectral", "temporal"),
        "gwcs_stokes_lookup": ("stokes",),
        "gwcs_3d_galactic_spectral": ("celestial", "spectral", "celestial"),
        "gwcs_2d_shift_scale": ("celestial", "celestial"),
        "gwcs_2d_shift_scale_quantity": ("celestial", "celestial"),
        "gwcs_1d_freq_quantity": ("spectral",),
        "gwcs_simple_2d": ("SPATIAL", "SPATIAL1"),
        "gwcs_empty_output_2d": ("UNKNOWN", "UNKNOWN1"),
        "gwcs_simple_imaging": ("celestial", "celestial"),
        "gwcs_with_frames_strings": (
            "UNKNOWN",
            "UNKNOWN1",
            "UNKNOWN2",
        ),
        "gwcs_high_level_pixel": ("celestial", "celestial"),
    }[fixture_name]


@pytest.fixture
def wcs_component_positions(fixture_name):
    return {
        "gwcs_2d_spatial_shift": (0, 1),
        "gwcs_2d_spatial_reordered": (1, 0),
        "gwcs_2d_quantity_shift": (0, 0),
        "gwcs_1d_freq": (0,),
        "gwcs_3d_spatial_wave": (0, 1, 0),
        "gwcs_3d_identity_units": (0, 1, 0),
        "gwcs_4d_identity_units": (0, 1, 0, 0),
        "gwcs_stokes_lookup": (0,),
        "gwcs_3d_galactic_spectral": (1, 0, 0),
        "gwcs_2d_shift_scale": (0, 1),
        "gwcs_2d_shift_scale_quantity": (0, 1),
        "gwcs_1d_freq_quantity": (0,),
        "gwcs_simple_2d": (0, 0),
        "gwcs_empty_output_2d": (0, 0),
        "gwcs_simple_imaging": (0, 1),
        "gwcs_with_frames_strings": (0, 0, 0),
        "gwcs_high_level_pixel": (0, 1),
    }[fixture_name]


@pytest.fixture
def wcs_component_properties(fixture_name):
    return {
        "gwcs_2d_spatial_shift": (
            (coord.SkyCoord(27, 90, unit="deg"), 27),
            (coord.SkyCoord(34.7, -45.7, unit="deg"), -45.7),
        ),
        "gwcs_2d_spatial_reordered": (
            (coord.SkyCoord(3.14, 90, unit="deg"), 90),
            (coord.SkyCoord(34.7, -2.72, unit="deg"), 34.7),
        ),
        "gwcs_2d_quantity_shift": ("value", "value"),
        "gwcs_1d_freq": ((coord.SpectralCoord(3, unit="Hz"), 3),),
        "gwcs_3d_spatial_wave": (
            (coord.SkyCoord(27, 90, unit="deg"), 27),
            (coord.SkyCoord(34.7, -45.7, unit="deg"), -45.7),
            (coord.SpectralCoord(3, unit="m"), 3),
        ),
        "gwcs_3d_identity_units": (
            (coord.SkyCoord(27, 90, unit="arcsec"), 27),
            (coord.SkyCoord(34.7, -45.7, unit="arcsec"), -45.7),
            (coord.SpectralCoord(3, unit="nm"), 3),
        ),
        "gwcs_4d_identity_units": (
            (coord.SkyCoord(27, 90, unit="deg"), 27),
            (coord.SkyCoord(34.7, -45.7, unit="deg"), -45.7),
            (coord.SpectralCoord(3, unit="nm"), 3),
            (time.Time("2020-01-01T00:00:03", format="isot", scale="utc"), 631152008.0),
        ),
        "gwcs_stokes_lookup": ("value",),
        "gwcs_3d_galactic_spectral": (
            (coord.SkyCoord(27, 90, unit="deg", frame="galactic"), 90),
            (coord.SpectralCoord(3, unit="Hz"), 3),
            (coord.SkyCoord(34.7, -45.7, unit="deg", frame="galactic"), 34.7),
        ),
        "gwcs_2d_shift_scale": (
            (coord.SkyCoord(27, 90, unit="deg"), 27),
            (coord.SkyCoord(34.7, -45.7, unit="deg"), -45.7),
        ),
        "gwcs_2d_shift_scale_quantity": (
            (coord.SkyCoord(27, 90, unit="deg"), 27),
            (coord.SkyCoord(34.7, -45.7, unit="deg"), -45.7),
        ),
        "gwcs_1d_freq_quantity": ((coord.SpectralCoord(3, unit="Hz"), 3),),
        "gwcs_simple_2d": ("value", "value"),
        "gwcs_empty_output_2d": ("value", "value"),
        "gwcs_simple_imaging": (
            (coord.SkyCoord(27, 90, unit="deg"), 27),
            (coord.SkyCoord(34.7, -45.7, unit="deg"), -45.7),
        ),
        "gwcs_with_frames_strings": ("value", "value", "value"),
        "gwcs_high_level_pixel": (
            (coord.SkyCoord(27, 90, unit="deg"), 27),
            (coord.SkyCoord(34.7, -45.7, unit="deg"), -45.7),
        ),
    }[fixture_name]


def test_world_axis_object_components(
    wcs_object,
    wcs_component_names,
    wcs_component_positions,
    wcs_component_properties,
):
    """
    Test the world_axis_object_components property returns the correct components.
    """
    waoc = wcs_object.world_axis_object_components
    assert isinstance(waoc, list)
    assert len(waoc) == wcs_object.world_n_dim

    for component, component_name, component_position, component_property in zip(
        waoc,
        wcs_component_names,
        wcs_component_positions,
        wcs_component_properties,
        strict=True,
    ):
        assert isinstance(component, tuple)
        assert len(component) == 3
        name, position, property_ = component
        assert component_name == name
        assert component_position == position

        if not callable(property_):
            assert component_property == property_
        else:
            assert property_(component_property[0]) == component_property[1]


@pytest.fixture
def wcs_component_object_classes(fixture_name, wcs_object):
    match fixture_name:
        case "gwcs_3d_identity_units":
            celestial_unit = (u.arcsec, u.arcsec)
        case _:
            celestial_unit = (u.deg, u.deg)
    celestial = (
        coord.SkyCoord,
        (),
        {
            "frame": wcs_object.output_frame.frames[0].reference_frame
            if isinstance(wcs_object.output_frame, cf.CompositeFrame)
            else wcs_object.output_frame.reference_frame,
            "unit": celestial_unit,
        },
    )
    match fixture_name:
        case "gwcs_1d_freq" | "gwcs_3d_galactic_spectral" | "gwcs_1d_freq_quantity":
            spectral_unit = u.Hz
        case "gwcs_3d_spatial_wave":
            spectral_unit = u.m
        case _:
            spectral_unit = u.nm
    spectral = (
        coord.SpectralCoord,
        (),
        {"unit": spectral_unit},
    )

    return {
        "gwcs_2d_spatial_shift": {"celestial": celestial},
        "gwcs_2d_spatial_reordered": {"celestial": celestial},
        "gwcs_2d_quantity_shift": {
            "SPATIAL": (u.Quantity, (), {"unit": u.km}),
            "SPATIAL1": (u.Quantity, (), {"unit": u.km}),
        },
        "gwcs_1d_freq": {"spectral": spectral},
        "gwcs_3d_spatial_wave": {
            "celestial": celestial,
            "spectral": spectral,
        },
        "gwcs_3d_identity_units": {
            "celestial": celestial,
            "spectral": spectral,
        },
        "gwcs_4d_identity_units": {
            "celestial": celestial,
            "spectral": spectral,
            "temporal": (
                time.Time,
                (),
                {
                    "format": "isot",
                    "in_subfmt": "*",
                    "location": None,
                    "out_subfmt": "*",
                    "precision": 3,
                    "scale": "utc",
                    "unit": u.s,
                },
            ),
        },
        "gwcs_stokes_lookup": {"stokes": (coord.StokesCoord, (), {})},
        "gwcs_3d_galactic_spectral": {
            "celestial": celestial,
            "spectral": spectral,
        },
        "gwcs_2d_shift_scale": {"celestial": celestial},
        "gwcs_2d_shift_scale_quantity": {"celestial": celestial},
        "gwcs_1d_freq_quantity": {"spectral": spectral},
        "gwcs_simple_2d": {
            "SPATIAL": (u.Quantity, (), {"unit": u.pixel}),
            "SPATIAL1": (u.Quantity, (), {"unit": u.pixel}),
        },
        "gwcs_empty_output_2d": {
            "UNKNOWN": (u.Quantity, (), {"unit": None}),
            "UNKNOWN1": (u.Quantity, (), {"unit": None}),
        },
        "gwcs_simple_imaging": {"celestial": celestial},
        "gwcs_with_frames_strings": {
            "UNKNOWN": (u.Quantity, (), {"unit": None}),
            "UNKNOWN1": (u.Quantity, (), {"unit": None}),
            "UNKNOWN2": (u.Quantity, (), {"unit": None}),
        },
        "gwcs_high_level_pixel": {"celestial": celestial},
    }[fixture_name]


def test_world_axis_object_classes(
    wcs_object, wcs_component_names, wcs_component_object_classes
):
    """
    Test the world_axis_object_classes property returns the correct dictionarly.
    """
    waoc = wcs_object.world_axis_object_classes
    assert set(waoc.keys()) == set(wcs_component_names)

    for component_name, component in waoc.items():
        assert isinstance(component, tuple)
        if component_name == "temporal":
            assert len(component) == 4
            assert callable(component[3])
        else:
            assert len(component) == 3

        assert component[0] == wcs_component_object_classes[component_name][0]
        assert component[1] == wcs_component_object_classes[component_name][1]
        assert component[2] == wcs_component_object_classes[component_name][2]


@pytest.fixture
def wcs_names(fixture_name):
    return {
        "gwcs_2d_spatial_shift": (("x", "y"), ("lon", "lat")),
        "gwcs_2d_spatial_reordered": (("x", "y"), ("lat", "lon")),
        "gwcs_2d_quantity_shift": (("x", "y"), ("", "")),
        "gwcs_1d_freq": (("",), ("",)),
        "gwcs_3d_spatial_wave": (("x", "y", "z"), ("lon", "lat", "lambda")),
        "gwcs_3d_identity_units": (
            ("x", "y", "z"),
            ("longitude", "latitude", "wavelength"),
        ),
        "gwcs_4d_identity_units": (
            ("x", "y", "z", "s"),
            ("lon", "lat", "", "isot(utc; None"),
        ),
        "gwcs_stokes_lookup": (("x",), ("stokes",)),
        "gwcs_3d_galactic_spectral": (
            ("", "", ""),
            ("Latitude", "Frequency", "Longitude"),
        ),
        "gwcs_2d_shift_scale": (("x", "y"), ("lon", "lat")),
        "gwcs_2d_shift_scale_quantity": (("x", "y"), ("lon", "lat")),
        "gwcs_1d_freq_quantity": (("",), ("",)),
        "gwcs_simple_2d": (("x", "y"), ("x", "y")),
        "gwcs_empty_output_2d": (("", ""), ("", "")),
        "gwcs_simple_imaging": (("x", "y"), ("lon", "lat")),
        "gwcs_with_frames_strings": (("", "", "", ""), ("", "", "")),
        "gwcs_high_level_pixel": (("lon", "lat"), ("lon", "lat")),
    }[fixture_name]


def test_pixel_axis_names(wcs_object, wcs_names):
    """Test the pixel_axis_names property returns the names of the pixel axes."""
    assert wcs_object.pixel_axis_names == wcs_names[0]


def test_world_axis_names(wcs_object, wcs_names):
    """Test the world_axis_names property returns the names of the world axes."""
    assert wcs_object.world_axis_names == wcs_names[1]


def test_low_level_wcs(wcs_object):
    """Test the low_level_wcs property returns self."""
    assert wcs_object.low_level_wcs is wcs_object
