"""
This file contains a set of pytest fixtures which are different gwcses for testing.
"""

import pytest

from gwcs import examples, geometry

# We import all the example fixtures here so that they are available to all tests
from .example_fixtures import *  # noqa: F403


@pytest.fixture
def sellmeier_glass():
    return examples.sellmeier_glass()


@pytest.fixture
def sellmeier_zemax():
    return examples.sellmeier_zemax()


@pytest.fixture
def spher_to_cart():
    return geometry.SphericalToCartesian()


@pytest.fixture
def cart_to_spher():
    return geometry.CartesianToSpherical()
