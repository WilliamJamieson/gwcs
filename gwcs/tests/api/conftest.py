import pytest

fixture_names = (
    "gwcs_2d_spatial_shift",
    "gwcs_2d_spatial_reordered",
    "gwcs_2d_quantity_shift",
    "gwcs_1d_freq",
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
    "gwcs_with_frames_strings",
    "gwcs_high_level_pixel",
)


@pytest.fixture(params=fixture_names)
def fixture_name(request):
    return request.param


@pytest.fixture
def wcs_object(request, fixture_name):
    return request.getfixturevalue(fixture_name)
