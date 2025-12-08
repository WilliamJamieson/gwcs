from abc import abstractmethod

from astropy.modeling import Model


class NumericalInverseModel(Model):
    """
    A model that computes the numerical inverse of a given model.

    Parameters
    ----------
    model : `~astropy.modeling.Model`
        The model for which to compute the numerical inverse.
    """

    fittable = False

    def __init__(self, model: Model):
        super().__init__()
        self.model = model

    @property
    def inverse(self):
        return self.model

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """The evaluation method for the numerical inverse algorithm"""
