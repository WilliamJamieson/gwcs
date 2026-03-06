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


@pytest.fixture
def pixels(fixture_name):
    match fixture_name:
        case "gwcs_1d_freq" | "gwcs_1d_freq_quantity" | "gwcs_stokes_lookup":
            return (1,)
        case (
            "gwcs_2d_spatial_shift"
            | "gwcs_2d_spatial_reordered"
            | "gwcs_2d_quantity_shift"
            | "gwcs_2d_shift_scale"
            | "gwcs_2d_shift_scale_quantity"
            | "gwcs_simple_2d"
            | "gwcs_empty_output_2d"
            | "gwcs_simple_imaging"
            | "gwcs_high_level_pixel"
        ):
            return 1, 2
        case "gwcs_3d_spatial_wave" | "gwcs_3d_identity_units":
            return 1, 2, 3
        case "gwcs_3d_galactic_spectral":
            return 10, 20, 30
        case "gwcs_4d_identity_units" | "gwcs_with_frames_strings":
            return 1, 2, 3, 4
        case _:
            msg = f"Unknown fixture name: {fixture_name}"
            raise ValueError(msg)


@pytest.fixture
def world(fixture_name):  # noqa: PLR0911
    match fixture_name:
        case "gwcs_1d_freq" | "gwcs_stokes_lookup":
            return (2,)
        case "gwcs_1d_freq_quantity":
            return (1,)
        case (
            "gwcs_2d_spatial_shift"
            | "gwcs_2d_quantity_shift"
            | "gwcs_simple_2d"
            | "gwcs_empty_output_2d"
            | "gwcs_high_level_pixel"
        ):
            return 2, 4
        case "gwcs_2d_spatial_reordered":
            return 4, 2
        case "gwcs_2d_shift_scale" | "gwcs_2d_shift_scale_quantity":
            return 10, 40
        case "gwcs_simple_imaging":
            return 5.525098, -72.051902
        case "gwcs_3d_spatial_wave":
            return 2, 4, 6
        case "gwcs_3d_identity_units":
            return 1, 2, 3
        case "gwcs_3d_galactic_spectral":
            return 79.80904918923017, 10.5, 205.79129497471095
        case "gwcs_4d_identity_units":
            return 2.777778e-04, 5.555556e-04, 3, 4
        case "gwcs_with_frames_strings":
            return 2, 3, 0
        case _:
            msg = f"Unknown fixture name: {fixture_name}"
            raise ValueError(msg)
