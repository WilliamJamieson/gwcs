# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.modeling.models import Gaussian1D, Gaussian2D
from gwcs.bounding_box import BoundingBox, _BaseModelArgument, ModelArgument, CompoundBoundingBox

import numpy as np
import pytest
import unittest.mock as mk


class TestBoundingBox:
    def test_validate(self):
        pass


class TestModelArgument:
    def test_create(self):
        argument = ModelArgument('test', 1, True)
        assert isinstance(argument, tuple)
        assert isinstance(argument, _BaseModelArgument)
        assert argument.name == 'test' == argument[0]
        assert argument.index == 1 == argument[1]
        assert argument.remove == True == argument[2]
        assert argument == ('test', 1, True)

    def test__get_index(self):
        argument = ModelArgument('test', 1, True)

        # name is None
        assert argument._get_index(None, mk.MagicMock()) is None

        # model is None
        assert argument._get_index(mk.MagicMock(), None) is None
        assert argument._get_index(mk.MagicMock()) is None

        # neither is None
        model = mk.MagicMock()
        model.inputs = ['test', 'name']
        assert argument._get_index('name', model) == 1

        # Error
        with pytest.raises(ValueError) as err:
            argument._get_index('other', model)
        assert str(err.value) == \
            "other is not an input of your model inputs: ['test', 'name']."

    def test__get_name(self):
        argument = ModelArgument('test', 1, True)

        # index is None
        assert argument._get_name(None, mk.MagicMock()) is None

        # model is None
        assert argument._get_name(mk.MagicMock(), None) is None
        assert argument._get_name(mk.MagicMock()) is None

        # neither is None
        model = mk.MagicMock()
        model.inputs = ['test', 'name']
        assert argument._get_name(1, model) == 'name'

        # Error
        with pytest.raises(IndexError) as err:
            argument._get_name(3, model)
        assert str(err.value) == \
            "There is nothing of index: 3 in your model inputs: ['test', 'name']."

    def test_validate(self):
        model = mk.MagicMock()
        model.inputs = [mk.MagicMock(), 'name']

        # Success with full data
        assert ModelArgument.validate(model, 'name', 1, True) == ('name', 1, True)
        assert ModelArgument.validate(model, 'name', 1) == ('name', 1, False)

        # Fail with full data
        with pytest.raises(ValueError) as err:
            ModelArgument.validate(model, 'name', 2)
        assert str(err.value) == \
            "Index should be 1, but was given 2."

        with mk.patch.object(ModelArgument, '_get_name', autospec=True,
                             return_value='test') as mkGet:
            with pytest.raises(ValueError) as err:
                ModelArgument.validate(model, 'name', 1)
            assert str(err.value) == \
                "Name should be test, but was given name."
            assert mkGet.call_args_list == [mk.call(1, model)]

        # Success with missing inputs
        assert ModelArgument.validate(model, 'name')  == ('name', 1, False)
        assert ModelArgument.validate(model, index=1) == ('name', 1, False)
        assert ModelArgument.validate(name='name', index=1)  == ('name', 1, False)

        # Fail with missing inputs
        with pytest.raises(ValueError) as err:
            ModelArgument.validate(model)
        assert str(err.value) == \
            "Enough information must be given so that both name and index can be determined."

        with pytest.raises(ValueError) as err:
            ModelArgument.validate(name='name')
        assert str(err.value) == \
            "Enough information must be given so that both name and index can be determined."

        with pytest.raises(ValueError) as err:
            ModelArgument.validate(index=1)
        assert str(err.value) == \
            "Enough information must be given so that both name and index can be determined."

        with pytest.raises(ValueError) as err:
            ModelArgument.validate()
        assert str(err.value) == \
            "Enough information must be given so that both name and index can be determined."

    def test_get_slice(self):
        argument = ModelArgument('test', 1, True)
        value_arg = mk.MagicMock()
        value_kwarg = mk.MagicMock()
        args = (mk.MagicMock(), value_arg)
        kwargs = {'test': value_kwarg, 'other': mk.MagicMock()}

        # Test get as kwarg


class TestCompoundBoundingBox:
    def test___init__(self):
        bbox = {1: (-1, 0), 2: (0, 1)}
        bounding_box = CompoundBoundingBox(bbox)

        assert bounding_box == bbox
        assert bounding_box._model is None

        bounding_box = CompoundBoundingBox(bbox, Gaussian1D())
        assert bounding_box == bbox
        assert (bounding_box._model.parameters == Gaussian1D().parameters).all()
