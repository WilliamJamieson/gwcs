# Licensed under a 3-clause BSD style license - see LICENSE.rst

from collections import UserDict, namedtuple
from astropy.modeling.core import Model
from astropy.modeling.utils import _BoundingBox
from astropy.utils import isiterable

import numpy as np
from typing import List, Dict


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
            try:
                length = len(slice_args)
            except TypeError:
                length = 1

            nd -= length

        return cls._validate(model, bounding_box, nd)


_BaseModelArgument = namedtuple('_BaseModelArgument', "name index remove")


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

    @staticmethod
    def _get_name(index: int, model=None):
        if index is None:
            return None

        if model is None:
            return None
        else:
            try:
                return model.inputs[index]
            except IndexError:
                raise IndexError(f'There is nothing of index: {index} in your model inputs: {model.inputs}.')

    @classmethod
    def validate(cls, model=None, name=None, index=None, remove=False):
        valid_index = cls._get_index(name, model)
        if valid_index is None:
            valid_index = index
        else:
            if index is not None and valid_index != index:
                raise ValueError(f"Index should be {valid_index}, but was given {index}.")

        valid_name = cls._get_name(index, model)
        if valid_name is None:
            valid_name = name
        else:
            if name is not None and valid_name != name:
                raise ValueError(f"Name should be {valid_name}, but was given {name}.")

        if valid_name is None or valid_index is None:
            raise ValueError("Enough information must be given so that both name and index can be determined.")
        else:
            return cls(valid_name, valid_index, remove)

    def get_slice(self, *args, **kwargs):
        args = list(args)
        if self.name in kwargs:
            value = kwargs[self.name]
            del kwargs[self.name]
        elif len(args) - 1 < self.index:
            value = args.pop(self.index)
        else:
            raise ValueError(f"Cannot find a valid argument corriponding to {self.name} or index {self.index}")

        return value, tuple(args), kwargs

    def add_bounding_box(self, bounding_box: BoundingBox) -> BoundingBox:
        if self.remove:
            new_bounding_box = []
            for index, value in enumerate(bounding_box):
                if index == self.index:
                    new_bounding_box.append(BoundingBox(-np.inf, np.inf))
                new_bounding_box.append(value)
            return BoundingBox(new_bounding_box)
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
        return ModelArguments(sorted(self, key=lambda x: x.index))

    @classmethod
    def validate(cls, model=None, arguments=None):
        if arguments is None:
            arguments = []

        if isinstance(arguments, ModelArguments):
            arguments = arguments.arguments

        valid_arguments = [ModelArgument.validate(model, *arg) for arg in arguments]
        return cls(valid_arguments)

    def __len__(self):
        return len(self._arguments)

    def get_arguments(self, *args, **kwargs) -> tuple:
        return tuple([arg.get_argument(*args, **kwargs) for arg in self._arguments])

    def add_bounding_box(self, bounding_box: BoundingBox) -> BoundingBox:
        for argument in self._arguments:
            bounding_box = argument.add_bounding_box(bounding_box)
        return bounding_box


class CompoundBoundingBox(UserDict):
    def __init__(self, bounding_box: Dict[str, BoundingBox],
                 model: Model=None, slice_args: ModelArguments=None):
        super().__init__(bounding_box)
        self._model = model

        if slice_args is None:
            slice_args = ModelArguments([])
        self._slice_args = ModelArguments.validate(self._model, slice_args)

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

    def _get_bounding_box(self, *args, **kwargs):
        slice_index = self._slice_args.get_arguments(*args, **kwargs)

        if slice_index in self:
            return self[slice_index]
        else:
            raise RuntimeError(f"No bounding_box is defined for slice: {slice_index}!")

    def _add_bounding_box(self, bounding_box: BoundingBox):
        return self._slice_args.sorted.add_bounding_box(bounding_box)

    def get_bounding_box(self, *args, **kwargs):
        return self._add_bounding_box(self._get_bounding_box(*args, **kwargs))


# class ComplexBoundingBox(UserDict):
#     def __init__(self, bounding_box,  model=None, slice_arg=None, slice_index=None, remove_slice_arg=False):
#         super().__init__(bounding_box)
#         self._model = model

#         if self._model is None:
#             self._slice_arg = slice_arg
#         else:
#             self.set_slice_arg(slice_arg)

#         self._remove_slice_arg = remove_slice_arg

#     @property
#     def slice_arg(self):
#         return self._slice_arg

#     @property
#     def remove_slice_arg(self):
#         return self._remove_slice_arg

#     def _get_arg_index(self, slice_arg):
#         if np.issubdtype(type(slice_arg), np.integer):
#             arg_index = slice_arg
#         else:
#             if slice_arg in self._model.inputs:
#                 arg_index = self._model.inputs.index(slice_arg)
#             else:
#                 raise ValueError(f'{slice_arg} is not an input of of your model inputs {self._model.inputs}')

#         if arg_index < self._model.n_inputs:
#             return arg_index
#         else:
#             raise ValueError(f'{arg_index} is out of model argument bounds')

#     @classmethod
#     def validate(cls, model, bounding_box, slice_arg=None, remove_slice_arg=None):
#         if isinstance(bounding_box, CompoundBoundingBox) and slice_arg is None:
#             slice_arg = bounding_box.slice_arg

#         if remove_slice_arg is None:
#             if isinstance(bounding_box, CompoundBoundingBox):
#                 remove_slice_arg = bounding_box.remove_slice_arg
#             else:
#                 remove_slice_arg = False

#         new_box = cls({}, model, slice_arg, remove_slice_arg)

#         if not remove_slice_arg:
#             slice_arg = None

#         for slice_index, slice_box in bounding_box.items():
#             new_box[slice_index] = BoundingBox.validate(model, slice_box, slice_arg)

#         return new_box

#     def set_slice_arg(self, slice_arg):
#         if slice_arg is None:
#             self._slice_arg = slice_arg
#         elif isinstance(slice_arg, tuple):
#             self._slice_arg = tuple([self._get_arg_index(arg) for arg in slice_arg])
#         else:
#             self._slice_arg = self._get_arg_index(slice_arg)

#     def _get_slice_index(self, inputs, slice_arg):
#         if inputs is None:
#             raise RuntimeError('Inputs must not be None in order to lookup slice')

#         slice_index = inputs[self._get_arg_index(slice_arg)]
#         if isinstance(slice_index, np.ndarray):
#             slice_index = slice_index.item()

#         return slice_index

#     def get_bounding_box(self, inputs=None, slice_index=True):
#         if isinstance(slice_index, bool) and slice:
#             if self._slice_arg is None:
#                 return None
#             else:
#                 if isinstance(self._slice_arg, tuple):
#                     slice_index = tuple([self._get_slice_index(inputs, slice_arg)
#                                          for slice_arg in self._slice_arg])
#                 else:
#                     slice_index = self._get_slice_index(inputs, self._slice_arg)

#         if slice_index in self:
#             return self[slice_index]
#         else:
#             raise RuntimeError(f"No bounding_box is defined for slice: {slice_index}!")

#     def reverse(self, axes_ind):
#         # bbox = {slice_index: slice_box.reverse(axes_ind)
#         #         for slice_index, slice_box in self.items()}
#         bbox = {}
#         for slice_index, slice_box in self.items():
#             bbox[slice_index] = [slice_box[ind] for ind in axes_ind][::-1]

#         return CompoundBoundingBox(bbox, self._model, self._slice_arg, self._remove_slice_arg)

#     def py_order(self, axes_order):
#         bbox = {}
#         for slice_index, slice_box in self.items():
#             bbox[slice_index] = tuple(slice_box[::-1][i] for i in axes_order)

#         return CompoundBoundingBox(bbox, self._model, self._slice_arg, self._remove_slice_arg)
