"""
Create a generic class representing an algorithm which can be applied to a dataset.
"""
from typing import List, Type, Callable, Dict
import alglab.dataset


class Algorithm(object):

    def __init__(self,
                 name: str,
                 implementation: Callable,
                 return_type: Type = None,
                 parameter_names: List[str] = None,
                 dataset_class: Type[alglab.dataset.Dataset] = alglab.dataset.NoDataset):
        """Create an algorithm definition. The implementation should be a python method which takes
        a dataset as a positional argument (if dataset_class is not NoDataset) and
        the parameters as keyword arguments. The implementation should return an object of type
        return_type.
        """
        self.implementation = implementation
        self.parameter_names = parameter_names if parameter_names is not None else []
        self.return_type = return_type
        self.dataset_class = dataset_class
        self.name = name

    def run(self, dataset: alglab.dataset.Dataset, params: Dict):
        if not isinstance(dataset, self.dataset_class):
            raise TypeError("Provided dataset type must match dataset_class expected by the implementation.")

        for param in params.keys():
            if param not in self.parameter_names:
                raise ValueError("Unexpected parameter name.")

        if self.dataset_class is not alglab.dataset.NoDataset:
            yield from self.implementation(dataset, **params)
        else:
            yield from self.implementation(**params)

    def __repr__(self):
        return self.name
