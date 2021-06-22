# Licensed under a 3-clause BSD style license - see LICENSE.rst

from collections import UserDict, namedtuple
from astropy.modeling.core import Model
from astropy.modeling.utils import _BoundingBox
from astropy.utils import isiterable

import numpy as np
from typing import List, Dict, Any, Callable


class BoundingBox(_BoundingBox):
    @classmethod
    def _validate(cls, model, bounding_box,  nd):
        if nd == 1:
            MESSAGE = f"""Bounding box for {model.__class__.__name__} model must be a sequence
            of length 2 consisting of a lower and upper bound, or a 1-tuple
            containing such a sequence as its sole element."""

            try:
                valid_shape = np.shape(bounding_box) in ((2,), (1, 2))
            except TypeError:
                # np.shape does not work with lists of Quantities
                valid_shape = np.shape([b.to_value() for b in bounding_box]) in ((2,), (1, 2))
            except ValueError:
                raise ValueError(MESSAGE)

            if not isiterable(bounding_box) or not valid_shape:
                raise ValueError(MESSAGE)

            if len(bounding_box) == 1:
                return cls((tuple(bounding_box[0]),))
            return cls(tuple(bounding_box))
        else:
            MESSAGE = f"""Bounding box for {model.__class__.__name__} model must be a sequence
            of length {model.n_inputs} (the number of model inputs) consisting of pairs of
            lower and upper bounds for those inputs on which to evaluate the model."""

            try:
                valid_shape = all([len(i) == 2 for i in bounding_box])
            except TypeError:
                valid_shape = False
            if len(bounding_box) != nd:
                valid_shape = False

            if not isiterable(bounding_box) or not valid_shape:
                raise ValueError(MESSAGE)

            return cls(tuple(bounds) for bounds in bounding_box)

    @classmethod
    def validate(cls, model, bounding_box, slice_args=None):
        """
        Validate a given bounding box sequence against the given model (which
        may be either a subclass of `~astropy.modeling.Model` or an instance
        thereof, so long as the ``.inputs`` attribute is defined.

        Currently this just checks that the bounding_box is either a 2-tuple
        of lower and upper bounds for 1-D models, or an N-tuple of 2-tuples
        for N-D models.

        This also returns a normalized version of the bounding_box input to
        ensure it is always an N-tuple (even for the 1-D case).
        """

        if isinstance(bounding_box, dict) or isinstance(bounding_box, CompoundBoundingBox):
            return CompoundBoundingBox.validate(model, bounding_box, slice_args=slice_args)

        nd = model.n_inputs
        if slice_args is not None:
            # Get list of removed if possible
            try:
                slice_args = slice_args.removed
            except AttributeError:
                pass

            # Get the number of args to remove
            try:
                length = len(slice_args)
            except TypeError:
                length = 1

            nd -= length

        return cls._validate(model, bounding_box, nd)


_BaseModelArgument = namedtuple('_BaseModelArgument', "name remove index")


class ModelArgument(_BaseModelArgument):
    @staticmethod
    def _get_index(name: str, model=None):
        if name is None:
            return None

        if model is None:
            return None
        else:
            if name in model.inputs:
                return model.inputs.index(name)
            else:
                raise ValueError(f'{name} is not an input of your model inputs: {model.inputs}.')

    @classmethod
    def validate(cls, model=None, name=None, remove=False, index=None):
        valid_index = cls._get_index(name, model)
        if valid_index is None:
            valid_index = index
        else:
            if index is not None and valid_index != index:
                raise IndexError(f"Index should be {valid_index}, but was given {index}.")

        if name is None or valid_index is None:
            raise ValueError("Enough information must be given so that both name and index can be determined.")
        else:
            return cls(name, remove, valid_index)

    def get_slice(self, **kwargs):
        if self.name in kwargs:
            return kwargs[self.name]
        else:
            raise ValueError(f"Cannot find a valid input corresponding to {self.name} in: {kwargs}.")

    @staticmethod
    def _removed_bounding_box():
        return BoundingBox((-np.inf, np.inf))

    def _add_bounding_box(self, bounding_box: BoundingBox) -> BoundingBox:
        if bounding_box.dimension == 1:
            new_bounding_box = [bounding_box]
        else:
            new_bounding_box = list(bounding_box)

        new_bounding_box.insert(self.index, self._removed_bounding_box())

        return BoundingBox(new_bounding_box)

    def add_bounding_box(self, bounding_box: BoundingBox) -> BoundingBox:
        if self.remove:
            return self._add_bounding_box(bounding_box)
        else:
            return bounding_box


class ModelArguments(object):
    def __init__(self, arguments=List[ModelArgument]):
        self._arguments = arguments

    @property
    def arguments(self) -> List[ModelArgument]:
        return self._arguments

    @property
    def names(self) -> List[str]:
        return [arg.name for arg in self._arguments]

    @property
    def indices(self) -> Dict[str, int]:
        return {arg.name: arg.index for arg in self._arguments}

    @property
    def sorted(self) -> 'ModelArguments':
        return ModelArguments(sorted(self._arguments, key=lambda x: x.index))

    @property
    def removed(self):
        return [arg for arg in self._arguments if arg.remove]

    @staticmethod
    def _validate_argument(model, arg):
        if isinstance(arg, list) or isinstance(arg, tuple):
            return ModelArgument.validate(model, *arg)
        else:
            return ModelArgument.validate(model, name=arg)

    @classmethod
    def validate(cls, model=None, arguments=None):
        if arguments is None:
            arguments = []

        if isinstance(arguments, ModelArguments):
            arguments = arguments.arguments

        valid_arguments = [cls._validate_argument(model, arg) for arg in arguments]

        return cls(valid_arguments)

    def __eq__(self, value):
        if isinstance(value, ModelArguments):
            return self.arguments == value.arguments
        else:
            return False

    def get_slice(self, **kwargs) -> tuple:
        slice_tuple = tuple([arg.get_slice(**kwargs) for arg in self._arguments])

        if len(slice_tuple) == 1:
            return slice_tuple[0]
        else:
            return slice_tuple

    def add_bounding_box(self, bounding_box: BoundingBox) -> BoundingBox:
        for argument in self._arguments:
            bounding_box = argument.add_bounding_box(bounding_box)
        return bounding_box


class CompoundBoundingBox(UserDict):
    def __init__(self, bounding_box: Dict[Any, BoundingBox],
                 model: Model=None, slice_args: ModelArguments=None,
                 create_slice: Callable=None):
        super().__init__(bounding_box)
        self._model = model
        self._slice_args = ModelArguments.validate(model, slice_args)
        self._create_slice = create_slice

    @property
    def slice_args(self):
        return self._slice_args

    @property
    def slice_names(self):
        return self._slice_args.names

    @property
    def slice_indicies(self):
        return self._slice_args.indices

    @classmethod
    def validate(cls, model, bounding_box, slice_args=None):
        if not isinstance(model, Model):
            model = None

        if isinstance(bounding_box, CompoundBoundingBox) and slice_args is None:
            slice_args = ModelArguments.validate(model,
                                                 bounding_box.slice_args)
        else:
            slice_args = ModelArguments.validate(model, slice_args)

        new_box = cls({}, model, slice_args)

        for slice_index, slice_box in bounding_box.items():
            new_box[slice_index] = BoundingBox.validate(model, slice_box, slice_args)

        return new_box

    def _get_slice(self, slice_index) -> BoundingBox:
        if slice_index in self:
            bbox = self[slice_index]
        elif self._create_slice is not None:
            bbox = self._create_slice(self._model, slice_index)
            self[slice_index] = bbox
        else:
            raise RuntimeError(f"No bounding_box is defined for slice: {slice_index}!")

        return bbox

    def _get_bounding_box(self, **kwargs) -> BoundingBox:
        slice_index = self._slice_args.get_slice(**kwargs)

        return self._get_slice(slice_index)

    def _add_bounding_box(self, bounding_box: BoundingBox):
        return self._slice_args.sorted.add_bounding_box(bounding_box)

    def get_bounding_box(self, **kwargs):
        return self._add_bounding_box(self._get_bounding_box(**kwargs))
