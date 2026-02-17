# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests the API defined in astropy APE 14 (https://doi.org/10.5281/zenodo.1188875).
"""

import astropy.modeling.models as m
import astropy.units as u
import numpy as np
import pytest
from astropy import coordinates as coord
from astropy import time
from astropy.wcs.wcsapi import HighLevelWCSWrapper
from astropy.wcs.wcsapi.high_level_api import values_to_high_level_objects
from numpy.testing import assert_allclose, assert_array_equal

import gwcs
import gwcs.coordinate_frames as cf

RNG = np.random.default_rng(42)


# Shorthand the name of the 2d gwcs fixture
@pytest.fixture
def wcsobj(request):
    return request.getfixturevalue(request.param)


wcs_objs = pytest.mark.parametrize("wcsobj", ["gwcs_2d_spatial_shift"], indirect=True)


@pytest.fixture
def wcs_ndim_types_units(request):
    """
    Generate a wcs and the expected ndim, types, and units.
    """
    ndim = {
        "gwcs_2d_spatial_shift": (2, 2),
        "gwcs_2d_spatial_reordered": (2, 2),
        "gwcs_1d_freq": (1, 1),
        "gwcs_3d_spatial_wave": (3, 3),
        "gwcs_4d_identity_units": (4, 4),
    }
    types = {
        "gwcs_2d_spatial_shift": ("pos.eq.ra", "pos.eq.dec"),
        "gwcs_2d_spatial_reordered": ("pos.eq.dec", "pos.eq.ra"),
        "gwcs_1d_freq": ("em.freq",),
        "gwcs_3d_spatial_wave": ("pos.eq.ra", "pos.eq.dec", "em.wl"),
        "gwcs_4d_identity_units": ("pos.eq.ra", "pos.eq.dec", "em.wl", "time"),
    }
    units = {
        "gwcs_2d_spatial_shift": ("deg", "deg"),
        "gwcs_2d_spatial_reordered": ("deg", "deg"),
        "gwcs_1d_freq": ("Hz",),
        "gwcs_3d_spatial_wave": ("deg", "deg", "m"),
        "gwcs_4d_identity_units": ("deg", "deg", "nm", "s"),
    }

    return (
        request.getfixturevalue(request.param),
        ndim[request.param],
        types[request.param],
        units[request.param],
    )


# # x, y inputs - scalar and array
x, y = 1, 2
xarr, yarr = np.ones((3, 4)), np.ones((3, 4)) + 1

_fixture_names = [
    "gwcs_2d_spatial_shift",
    "gwcs_2d_spatial_reordered",
    "gwcs_1d_freq",
    "gwcs_3d_spatial_wave",
    "gwcs_4d_identity_units",
]
fixture_wcses = pytest.mark.parametrize("wcsobj", _fixture_names, indirect=True)
fixture_wcs_ndim_types_units = pytest.mark.parametrize(
    "wcs_ndim_types_units", _fixture_names, indirect=True
)
all_wcses_names = [
    *_fixture_names,
    "gwcs_3d_identity_units",
    "gwcs_stokes_lookup",
    "gwcs_3d_galactic_spectral",
]
fixture_all_wcses = pytest.mark.parametrize("wcsobj", all_wcses_names, indirect=True)


@fixture_all_wcses
def test_names(wcsobj):
    assert wcsobj.world_axis_names == wcsobj.output_frame.axes_names
    assert wcsobj.pixel_axis_names == wcsobj.input_frame.axes_names


def test_names_split(gwcs_3d_galactic_spectral):
    wcs = gwcs_3d_galactic_spectral
    assert (
        wcs.world_axis_names
        == wcs.output_frame.axes_names
        == ("Latitude", "Frequency", "Longitude")
    )


_fixture_names_ = [
    "gwcs_2d_spatial_shift",
    "gwcs_2d_spatial_reordered",
    "gwcs_2d_quantity_shift",
    "gwcs_1d_freq",
    "gwcs_3d_spatial_wave",
    "gwcs_3d_identity_units",
    "gwcs_4d_identity_units",
]


@pytest.fixture(params=_fixture_names_)
def fixture_name(request):
    return request.param


@pytest.fixture
def wcs_object(request, fixture_name):
    return request.getfixturevalue(fixture_name)


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
    }[fixture_name]


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
    }[fixture_name]


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
    }[fixture_name]


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
    }[fixture_name]


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
        case "gwcs_1d_freq":
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
    }[fixture_name]


class TestAPE14LowLevelAPIProperties:
    """Test the low level API Properties defined in APE 14."""

    def test_pixel_n_dim(self, wcs_object, wcs_ndim):
        """Test the pixel_n_dim returns the number of pixel axes."""
        assert wcs_object.pixel_n_dim == wcs_ndim[0]

    def test_world_n_dim(sellf, wcs_object, wcs_ndim):
        """Test the world_n_dim returns the number of world axes."""
        assert wcs_object.world_n_dim == wcs_ndim[1]

    def test_array_shape_and_pixel_shape(self, wcs_object):
        """Test the array_shape and pixel_shape properties and how they relate."""
        assert wcs_object.array_shape is None

        array_shape = tuple(RNG.integers(1020, 4096, size=wcs_object.pixel_n_dim))

        wcs_object.array_shape = array_shape
        assert_array_equal(wcs_object.array_shape, array_shape)

        assert wcs_object.array_shape == wcs_object.pixel_shape[::-1]

        pixel_shape = tuple(RNG.integers(1020, 4096, size=wcs_object.pixel_n_dim))
        assert not (np.array(pixel_shape) == np.array(array_shape)).all()

        wcs_object.pixel_shape = pixel_shape
        assert wcs_object.array_shape == pixel_shape[::-1]

    def test_pixel_bounds(self, wcs_object):
        """Test the pixel_bounds property is the bounding box."""
        assert wcs_object.pixel_bounds is None

        bbox = tuple(
            (-0.5, RNG.uniform(1020, 4096)) for _ in range(wcs_object.pixel_n_dim)
        )

        wcs_object.bounding_box = bbox
        assert_array_equal(wcs_object.pixel_bounds, wcs_object.bounding_box)
        # Reset the bounding box or this will affect other tests
        wcs_object.bounding_box = None

    def test_world_axis_physical_types(self, wcs_object, wcs_types):
        """Test the world_axis_physical_types property returns the physical types."""
        assert wcs_object.world_axis_physical_types == wcs_types

    def test_world_axis_units(self, wcs_object, wcs_units):
        """Test the world_axis_units property returns the units."""
        assert wcs_object.world_axis_units == wcs_units

    def test_axis_correlation_matrix(self, fixture_name, wcs_object):
        """Test the axis_correlation_matrix property returns the correct matrix."""
        matrix = np.identity(wcs_object.pixel_n_dim)
        # spatial_reorderd happens to be flipped
        if fixture_name == "gwcs_2d_spatial_reordered":
            matrix = np.flipud(matrix)

        assert_array_equal(wcs_object.axis_correlation_matrix, matrix)

    def test_serialized_classes(self, wcs_object):
        """Test the serialized_classes property returns `False`."""
        assert wcs_object.serialized_classes is False

    def test_world_axis_object_components(
        self,
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

    def test_world_axis_object_classes(
        self, wcs_object, wcs_component_names, wcs_component_object_classes
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


@pytest.mark.parametrize(("x", "y"), zip((x, xarr), (y, yarr), strict=False))
def test_pixel_to_world_values(gwcs_2d_spatial_shift, x, y):
    wcsobj = gwcs_2d_spatial_shift
    assert_allclose(wcsobj.pixel_to_world_values(x, y), wcsobj(x, y))


@pytest.mark.parametrize(("x", "y"), zip((x, xarr), (y, yarr), strict=False))
def test_pixel_to_world_values_units_2d(gwcs_2d_shift_scale_quantity, x, y):
    wcsobj = gwcs_2d_shift_scale_quantity

    call_pixel = x * u.pix, y * u.pix
    api_pixel = x, y

    call_world = wcsobj(*call_pixel)
    api_world = wcsobj.pixel_to_world_values(*api_pixel)

    # Check that call returns quantities and api doesn't
    assert all(isinstance(a, u.Quantity) for a in call_world)
    assert all(not isinstance(a, u.Quantity) for a in api_world)

    # Check that they are the same (and implicitly in the same units)
    assert_allclose(u.Quantity(call_world).value, api_world)

    new_call_pixel = wcsobj.invert(*call_world)
    [assert_allclose(n, p) for n, p in zip(new_call_pixel, call_pixel, strict=False)]

    new_api_pixel = wcsobj.world_to_pixel_values(*api_world)
    [assert_allclose(n, p) for n, p in zip(new_api_pixel, api_pixel, strict=False)]


@pytest.mark.parametrize(("x"), [x, xarr])
def test_pixel_to_world_values_units_1d(gwcs_1d_freq_quantity, x):
    wcsobj = gwcs_1d_freq_quantity

    call_pixel = x * u.pix
    api_pixel = x

    call_world = wcsobj(call_pixel)
    api_world = wcsobj.pixel_to_world_values(api_pixel)

    # Check that call returns quantities and api doesn't
    assert isinstance(call_world, u.Quantity)
    assert not isinstance(api_world, u.Quantity)

    # Check that they are the same (and implicitly in the same units)
    assert_allclose(u.Quantity(call_world).value, api_world)

    new_call_pixel = wcsobj.invert(call_world)
    assert_allclose(new_call_pixel, call_pixel)

    new_api_pixel = wcsobj.world_to_pixel_values(api_world)
    assert_allclose(new_api_pixel, api_pixel)


@pytest.mark.parametrize(("x", "y"), zip((x, xarr), (y, yarr), strict=False))
def test_array_index_to_world_values(gwcs_2d_spatial_shift, x, y):
    wcsobj = gwcs_2d_spatial_shift
    assert_allclose(wcsobj.array_index_to_world_values(x, y), wcsobj(y, x))


def _compare_frame_output(wc1, wc2):
    if isinstance(wc1, coord.SkyCoord):
        assert isinstance(wc1.frame, type(wc2.frame))
        assert u.allclose(wc1.spherical.lon, wc2.spherical.lon, equal_nan=True)
        assert u.allclose(wc1.spherical.lat, wc2.spherical.lat, equal_nan=True)
        assert u.allclose(
            wc1.spherical.distance, wc2.spherical.distance, equal_nan=True
        )

    elif isinstance(wc1, u.Quantity):
        assert u.allclose(wc1, wc2, equal_nan=True)

    elif isinstance(wc1, time.Time):
        assert u.allclose((wc1 - wc2).to(u.s), 0 * u.s)

    elif isinstance(wc1, str | coord.StokesCoord):
        assert wc1 == wc2

    else:
        msg = f"Can't Compare {type(wc1)}"
        raise TypeError(msg)


@fixture_all_wcses
def test_high_level_wrapper(wcsobj, request):
    hlvl = HighLevelWCSWrapper(wcsobj)

    pixel_input = [3] * wcsobj.pixel_n_dim
    if wcsobj.bounding_box is not None:
        for i, interval in wcsobj.bounding_box.intervals.items():
            bbox_min = u.Quantity(interval.lower).value
            pixel_input[i] = max(bbox_min + 1, pixel_input[i])

    # Assert that both APE 14 API and GWCS give the same answer The APE 14 API
    # uses the mixin class and __call__ calls values_to_high_level_objects
    wc1 = hlvl.pixel_to_world(*pixel_input)
    wc2 = wcsobj(*pixel_input)
    results = wcsobj.output_frame.remove_units(wc2)

    wc2 = values_to_high_level_objects(*results, low_level_wcs=wcsobj)
    if len(wc2) == 1:
        wc2 = wc2[0]
    assert type(wc1) is type(wc2)

    if isinstance(wc1, list | tuple):
        for w1, w2 in zip(wc1, wc2, strict=False):
            _compare_frame_output(w1, w2)
    else:
        _compare_frame_output(wc1, wc2)

    # we have just asserted that wc1 and wc2 are equal
    if not isinstance(wc1, list | tuple):
        wc1 = (wc1,)

    pix_out1 = hlvl.world_to_pixel(*wc1)
    pix_out2 = wcsobj.invert(*wc1)

    pix_out2 = wcsobj._remove_quantity_frame(pix_out2, wcsobj.input_frame)

    if not isinstance(pix_out2, list | tuple):
        pix_out2 = (pix_out2,)

    np.testing.assert_allclose(pix_out1, pixel_input)
    np.testing.assert_allclose(pix_out2, pixel_input)


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


@wcs_objs
def test_low_level_wcs(wcsobj):
    assert id(wcsobj.low_level_wcs) == id(wcsobj)


@wcs_objs
def test_pixel_to_world(wcsobj):
    values = wcsobj(x, y)
    result = wcsobj.pixel_to_world(x, y)
    assert isinstance(result, coord.SkyCoord)
    assert_allclose(values[0] * u.deg, result.data.lon)
    assert_allclose(values[1] * u.deg, result.data.lat)


@wcs_objs
def test_array_index_to_world(wcsobj):
    values = wcsobj(x, y)
    result = wcsobj.array_index_to_world(y, x)
    assert isinstance(result, coord.SkyCoord)
    assert_allclose(values[0] * u.deg, result.data.lon)
    assert_allclose(values[1] * u.deg, result.data.lat)


def test_pixel_to_world_quantity(gwcs_2d_shift_scale, gwcs_2d_shift_scale_quantity):
    result1 = gwcs_2d_shift_scale.pixel_to_world(x, y)
    result2 = gwcs_2d_shift_scale_quantity.pixel_to_world(x, y)
    assert isinstance(result2, coord.SkyCoord)
    assert_allclose(result1.data.lon, result2.data.lon)
    assert_allclose(result1.data.lat, result2.data.lat)

    # test with Quantity pixel inputs
    result1 = gwcs_2d_shift_scale.pixel_to_world(x * u.pix, y * u.pix)
    result2 = gwcs_2d_shift_scale_quantity.pixel_to_world(x * u.pix, y * u.pix)
    assert isinstance(result2, coord.SkyCoord)
    assert_allclose(result1.data.lon, result2.data.lon)
    assert_allclose(result1.data.lat, result2.data.lat)

    # test for pixel units
    with pytest.raises(ValueError):  # noqa: PT011
        gwcs_2d_shift_scale.pixel_to_world(x * u.Jy, y * u.Jy)


def test_array_index_to_world_quantity(
    gwcs_2d_shift_scale, gwcs_2d_shift_scale_quantity
):
    result0 = gwcs_2d_shift_scale.pixel_to_world(x, y)
    result1 = gwcs_2d_shift_scale.array_index_to_world(y, x)
    result2 = gwcs_2d_shift_scale_quantity.array_index_to_world(y, x)
    assert isinstance(result2, coord.SkyCoord)
    assert_allclose(result1.data.lon, result2.data.lon)
    assert_allclose(result1.data.lat, result2.data.lat)
    assert_allclose(result0.data.lon, result1.data.lon)
    assert_allclose(result0.data.lat, result1.data.lat)

    # test with Quantity pixel inputs
    result0 = gwcs_2d_shift_scale.pixel_to_world(x * u.pix, y * u.pix)
    result1 = gwcs_2d_shift_scale.array_index_to_world(y * u.pix, x * u.pix)
    result2 = gwcs_2d_shift_scale_quantity.array_index_to_world(y * u.pix, x * u.pix)
    assert isinstance(result2, coord.SkyCoord)
    assert_allclose(result1.data.lon, result2.data.lon)
    assert_allclose(result1.data.lat, result2.data.lat)
    assert_allclose(result0.data.lon, result1.data.lon)
    assert_allclose(result0.data.lat, result1.data.lat)

    # test for pixel units
    with pytest.raises(ValueError):  # noqa: PT011
        gwcs_2d_shift_scale.array_index_to_world(x * u.Jy, y * u.Jy)


def test_world_to_pixel_quantity(gwcs_2d_shift_scale, gwcs_2d_shift_scale_quantity):
    skycoord = gwcs_2d_shift_scale.pixel_to_world(x, y)
    result1 = gwcs_2d_shift_scale.world_to_pixel(skycoord)
    result2 = gwcs_2d_shift_scale_quantity.world_to_pixel(skycoord)
    assert_allclose(result1, (x, y))
    assert_allclose(result2, (x, y))


def test_world_to_array_index_quantity(
    gwcs_2d_shift_scale, gwcs_2d_shift_scale_quantity
):
    skycoord = gwcs_2d_shift_scale.pixel_to_world(x, y)
    result0 = gwcs_2d_shift_scale.world_to_pixel(skycoord)
    result1 = gwcs_2d_shift_scale.world_to_array_index(skycoord)
    result2 = gwcs_2d_shift_scale_quantity.world_to_array_index(skycoord)
    assert_allclose(result0, (x, y))
    assert_allclose(result1, (y, x))
    assert_allclose(result2, (y, x))


@pytest.fixture(params=[0, 1])
def sky_ra_dec(request, gwcs_2d_spatial_shift):
    ref_frame = gwcs_2d_spatial_shift.output_frame.reference_frame
    ra, dec = 2, 4
    if request.param == 0:
        sky = coord.SkyCoord(ra * u.deg, dec * u.deg, frame=ref_frame)
    else:
        ra = np.ones((3, 4)) * ra
        dec = np.ones((3, 4)) * dec
        sky = coord.SkyCoord(ra * u.deg, dec * u.deg, frame=ref_frame)
    return sky, ra, dec


def test_world_to_pixel(gwcs_2d_spatial_shift, sky_ra_dec):
    wcsobj = gwcs_2d_spatial_shift
    sky, ra, dec = sky_ra_dec
    assert_allclose(wcsobj.world_to_pixel(sky), wcsobj.invert(ra, dec))


def test_world_to_array_index(gwcs_simple_imaging, sky_ra_dec):
    wcsobj = gwcs_simple_imaging
    sky, ra, dec = sky_ra_dec

    assert_allclose(
        wcsobj.world_to_array_index(sky),
        wcsobj.invert(ra, dec)[::-1],
    )


def test_world_to_pixel_values(gwcs_2d_spatial_shift, sky_ra_dec):
    wcsobj = gwcs_2d_spatial_shift
    sky, ra, dec = sky_ra_dec

    assert_allclose(wcsobj.world_to_pixel_values(ra, dec), wcsobj.invert(ra, dec))


def test_world_to_array_index_values(gwcs_simple_imaging, sky_ra_dec):
    wcsobj = gwcs_simple_imaging
    sky, ra, dec = sky_ra_dec

    assert_allclose(
        wcsobj.world_to_array_index_values(ra, dec),
        wcsobj.invert(ra, dec)[::-1],
    )


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
        1, "SPATIAL", (0,), unit=(u.deg,), name="sepframe"
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


def test_high_level_objects_to_values_units(gwcs_3d_identity_units):
    from astropy.wcs.wcsapi.high_level_api import high_level_objects_to_values

    wcs = gwcs_3d_identity_units
    world = wcs.pixel_to_world(1, 1, 1)

    values = high_level_objects_to_values(*world, low_level_wcs=wcs)

    expected_values = [
        world[0].spherical.lon.to_value(wcs.output_frame.unit[0]),
        world[0].spherical.lon.to_value(wcs.output_frame.unit[1]),
        world[1].to_value(wcs.output_frame.unit[2]),
    ]

    assert not any(isinstance(o, u.Quantity) for o in values)
    np.testing.assert_allclose(values, expected_values)


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


def test_no_input_frame(gwcs_simple_2d):
    """Test running the API on the WCS with no input frame."""
    assert (np.array([2]), np.array([-1])) == gwcs_simple_2d.world_to_pixel_values(
        np.array([3]), np.array([1])
    )
    assert (np.array([4]), np.array([3])) == gwcs_simple_2d.pixel_to_world_values(
        np.array([3]), np.array([1])
    )
