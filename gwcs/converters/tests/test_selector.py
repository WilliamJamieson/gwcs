# Licensed under a 3-clause BSD style license - see LICENSE.rst
import asdf
import numpy as np
import pytest
from asdf_astropy.testing.helpers import assert_model_equal
from astropy.modeling import Model
from astropy.modeling.models import Const1D, Mapping, Polynomial2D, Scale, Shift
from numpy.testing import assert_array_equal

from gwcs.converters.selector import LabelMapperConverter
from gwcs.selector import (
    LabelMapper,
    LabelMapperArray,
    LabelMapperDict,
    LabelMapperRange,
    RegionsSelector,
    _LabelMapper,
)
from gwcs.tests.test_region import create_scalar_mapper


def _assert_mapper_equal(a, b):
    assert type(a) is type(b)

    if isinstance(a, LabelMapper):
        assert_model_equal(a.mapper, b.mapper)
        assert_model_equal(a, b)
    elif isinstance(a.mapper, dict):
        assert a.mapper.__class__ == b.mapper.__class__  # nosec
        assert np.isin(list(a.mapper), list(b.mapper)).all()  # nosec
        for k in a.mapper:
            assert a.mapper[k].__class__ == b.mapper[k].__class__  # nosec
            assert all(a.mapper[k].parameters == b.mapper[k].parameters)  # nosec
        assert a.inputs == b.inputs  # nosec
        assert a.inputs_mapping.mapping == b.inputs_mapping.mapping  # nosec
    else:
        assert_array_equal(a.mapper, b.mapper)


def _assert_selector_equal(a, b):
    assert type(a) is type(b)
    _assert_mapper_equal(a.label_mapper, b.label_mapper)
    assert_array_equal(a.inputs, b.inputs)
    assert_array_equal(a.outputs, b.outputs)
    assert_array_equal(a.selector.keys(), b.selector.keys())
    for key in a.selector:
        assert_array_equal(a.selector[key].parameters, b.selector[key].parameters)
    assert_array_equal(a.undefined_transform_value, b.undefined_transform_value)


def assert_selector_roundtrip(s, tmp_path, lazy_load=True, version=None):
    """
    Assert that a selector can be written to an ASDF file and read back
    in without losing any of its essential properties.
    """
    path = tmp_path / "test.asdf"

    with asdf.AsdfFile({"selector": s}, version=version) as af:
        af.write_to(path)

    with asdf.open(path, lazy_load=lazy_load) as af:
        rs = af["selector"]
        match rs := af["selector"]:
            case RegionsSelector():
                return _assert_selector_equal(s, rs)
            case _LabelMapper():
                return _assert_mapper_equal(s, rs)

        msg = "Unknown selector type"  # pragma: no cover
        raise TypeError(msg)  # pragma: nocover


@pytest.mark.parametrize("lazy_load", [True, False])
class TestSelectorConverter:
    def test_regions_selector(self, tmp_path, lazy_load):
        m1 = Mapping([0, 1, 1]) | Shift(1) & Shift(2) & Shift(3)
        m2 = Mapping([0, 1, 1]) | Scale(2) & Scale(3) & Scale(3)
        sel = {1: m1, 2: m2}
        a = np.zeros((5, 6), dtype=np.int32)
        a[:, 1:3] = 1
        a[:, 4:5] = 2
        mask = LabelMapperArray(a)
        rs = RegionsSelector(
            inputs=("x", "y"),
            outputs=("ra", "dec", "lam"),
            selector=sel,
            label_mapper=mask,
        )
        assert_selector_roundtrip(rs, tmp_path, lazy_load=lazy_load)

    def test_LabelMapperArray_str(self, tmp_path, lazy_load):
        a = np.array(
            [
                ["label1", "", "label2"],
                ["label1", "", ""],
                ["label1", "label2", "label2"],
            ]
        )
        mask = LabelMapperArray(a)
        assert_selector_roundtrip(mask, tmp_path, lazy_load=lazy_load)

    def test_labelMapperArray_int(self, tmp_path, lazy_load):
        a = np.array([[1, 0, 2], [1, 0, 0], [1, 2, 2]])
        mask = LabelMapperArray(a)
        assert_selector_roundtrip(mask, tmp_path, lazy_load=lazy_load)

    def test_labelMapperArray_non_default_inputs(self, tmp_path, lazy_load):
        a = np.array([[1, 0, 2], [1, 0, 0], [1, 2, 2]])
        mask = LabelMapperArray(
            a, inputs_mapping=Mapping((0, 1), n_inputs=3), inputs=("x", "y", "order")
        )
        assert_selector_roundtrip(mask, tmp_path, lazy_load=lazy_load)

    def test_LabelMapperDict(self, tmp_path, lazy_load):
        d_mapper = create_scalar_mapper()
        sel = LabelMapperDict(
            ("x", "y"), d_mapper, inputs_mapping=Mapping((0,), n_inputs=2), atol=1e-3
        )
        assert_selector_roundtrip(sel, tmp_path, lazy_load=lazy_load)

    def test_LabelMapperRange(self, tmp_path, lazy_load):
        m = []
        for i in np.arange(9) * 0.1:
            c0_0, c1_0, c0_1, c1_1 = np.ones((4,)) * i
            m.append(Polynomial2D(2, c0_0=c0_0, c1_0=c1_0, c0_1=c0_1, c1_1=c1_1))
        keys = np.array(
            [
                [4.88, 5.64],
                [5.75, 6.5],
                [6.67, 7.47],
                [7.7, 8.63],
                [8.83, 9.96],
                [10.19, 11.49],
                [11.77, 13.28],
                [13.33, 15.34],
                [15.56, 18.09],
            ]
        )
        rmapper = {}
        for k, v in zip(keys, m, strict=False):
            rmapper[tuple(k)] = v
        sel = LabelMapperRange(
            ("x", "y"), rmapper, inputs_mapping=Mapping((0,), n_inputs=2)
        )
        assert_selector_roundtrip(sel, tmp_path, lazy_load=lazy_load)

    def test_no_inputs(self, lazy_load):
        """
        This having no inputs listed is valid under the schemas but models create
        them by default so we directly simulate the node information
        """
        a = np.array(
            [
                ["label1", "", "label2"],
                ["label1", "", ""],
                ["label1", "label2", "label2"],
            ]
        )
        node = {"mapper": a}
        mapper_array = LabelMapperConverter().from_yaml_tree_transform(node, None, None)
        assert isinstance(mapper_array, LabelMapperArray)
        assert (mapper_array.mapper == a).all()
        assert mapper_array.inputs == ("x", "y")
        assert mapper_array.no_label == ""
        assert mapper_array.inputs_mapping is None

    def test_bad_inputs_mapping(self, lazy_load):
        """
        Have a bad inputs mapping, we again simulate this by acting on the node
        directly
        """
        node = {"inputs_mapping": "foo"}
        with pytest.raises(TypeError, match=r"inputs_mapping must be an.*"):
            LabelMapperConverter().from_yaml_tree_transform(node, None, None)

    @pytest.mark.parametrize(
        "mapper",
        [
            np.array(["label1", "label2"]),
            np.array(
                [
                    [["label1", "label2"], ["label3", "label4"]],
                    [["label5", "label6"], ["label7", "label8"]],
                ]
            ),
        ],
    )
    def test_not_2D(self, tmp_path, lazy_load, mapper):
        """
        Have a mapping that is not 2D
        """
        assert mapper.ndim != 2
        sel = LabelMapperArray(mapper)
        with pytest.raises(NotImplementedError, match=r"GWCS currently only.*"):
            assert_selector_roundtrip(sel, tmp_path, lazy_load=lazy_load)

    def test_LabelMapper(self, tmp_path, lazy_load):
        """Test round trip the LabelMapper"""
        transform = Const1D(12.3)
        sel = LabelMapper(inputs=("x", "y"), mapper=transform, inputs_mapping=(1,))

        assert_selector_roundtrip(sel, tmp_path, lazy_load=lazy_load)

    def test_bad_model(self, lazy_load):
        class BadMapper(Model):
            no_label = 42
            inputs_mapping = None

            def evaluate(*args, **kwargs):
                return 1  # pragma: no cover

        sel = BadMapper()
        with pytest.raises(TypeError, match=r"Unrecognized type.*"):
            LabelMapperConverter().to_yaml_tree_transform(sel, None, None)
