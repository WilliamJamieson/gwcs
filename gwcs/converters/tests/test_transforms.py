# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from asdf_astropy.testing.helpers import assert_model_roundtrip
from astropy import units as u
from astropy.modeling import Model
from astropy.modeling.models import Identity, Pix2Sky_Gnomonic

from gwcs import fitswcs, geometry
from gwcs import spectroscopy as sp
from gwcs.converters.geometry import (
    DirectionCosinesConverter,
    SphericalCartesianConverter,
)
from gwcs.converters.spectroscopy import GratingEquationConverter

sell_glass = sp.SellmeierGlass(
    B_coef=[0.58339748, 0.46085267, 3.8915394],
    C_coef=[0.00252643, 0.010078333, 1200.556],
)
sell_zemax = sp.SellmeierZemax(
    65,
    35,
    0,
    0,
    [0.58339748, 0.46085267, 3.8915394],
    [0.00252643, 0.010078333, 1200.556],
    [-2.66e-05, 0.0, 0.0],
)
snell = sp.Snell3D()
todircos = geometry.ToDirectionCosines()
fromdircos = geometry.FromDirectionCosines()
tocart = geometry.SphericalToCartesian()
tospher = geometry.CartesianToSpherical()
tan = Pix2Sky_Gnomonic()
fwcs = fitswcs.FITSImagingWCSTransform(tan)


transforms = [
    todircos,
    fromdircos,
    tospher,
    tocart,
    snell,
    sell_glass,
    sell_zemax,
    sell_zemax & todircos | snell & Identity(1) | fromdircos,
    sell_glass & todircos | snell & Identity(1) | fromdircos,
    sp.WavelengthFromGratingEquation(50000, -1),
    sp.AnglesFromGratingEquation3D(20000, 1),
    sp.WavelengthFromGratingEquation(15000 * 1 / u.m, -1),
    fwcs,
]


@pytest.mark.parametrize(("model"), transforms)
def test_transforms(tmp_path, model):
    assert_model_roundtrip(model, tmp_path, version="1.6.0")


class TestFromYamlError:
    """Test the from_yaml_tree_transform gives an error when necessary"""

    @pytest.mark.parametrize("transform_type", [None, "foo"])
    def test_direction_cosines(self, transform_type):
        """Test for the DirectionCosines"""

        node = {} if transform_type is None else {"transform_type": transform_type}

        with pytest.raises(TypeError, match=r"Unknown transform_type.*"):
            DirectionCosinesConverter().from_yaml_tree_transform(node, None, None)

    @pytest.mark.parametrize("transform_type", [None, "foo"])
    def test_spherical_cartesian(self, transform_type):
        """Test for the SpericalCartesian"""
        node = {"wrap_lon_at": 42}

        if transform_type is not None:
            node["transform_type"] = transform_type

        with pytest.raises(TypeError, match=r"Unknown transform_type.*"):
            SphericalCartesianConverter().from_yaml_tree_transform(node, None, None)

    @pytest.mark.parametrize("output", [None, "foo"])
    def test_grating_equation(self, output):
        """Test for the GratingEquation"""

        node = {"groove_density": 5000, "order": -1}
        if output is not None:
            node["output"] = output

        with pytest.raises(ValueError, match=r"Can't create a GratingEquation.*"):
            GratingEquationConverter().from_yaml_tree_transform(node, None, None)


class TestToYamlError:
    """Test the to_yaml_tree_transform gives an error when necessary"""

    @pytest.mark.parametrize("model", [Identity(1), Pix2Sky_Gnomonic()])
    def test_direction_cosines(self, model):
        """Test that the converter gives an error when the wrong model type is passed"""

        with pytest.raises(TypeError, match=r"Model of type .*"):
            DirectionCosinesConverter().to_yaml_tree_transform(model, None, None)

    @pytest.mark.parametrize("model", [Identity(1), Pix2Sky_Gnomonic()])
    def test_spherical_cartesian(self, model):
        """Test that the converter gives an error when the wrong model type is passed"""

        with pytest.raises(TypeError, match=r"Model of type .*"):
            SphericalCartesianConverter().to_yaml_tree_transform(model, None, None)

    def test_grating_equation(self):
        """Test that the converter gives an error when the wrong model type is passed"""

        class BadGrating(Model):
            groove_density = 1 * u.dimensionless_unscaled
            spectral_order = 1 * u.dimensionless_unscaled

            def evaluate(self, *args, **kwargs):
                return 1  # pragma: no cover

        model = BadGrating()

        with pytest.raises(TypeError, match=r"Can't serialize an instance of.*"):
            GratingEquationConverter().to_yaml_tree_transform(model, None, None)
