"""Classes and methods related to running experiments on algorithms and datasets."""
import time

from typing import Dict, Type, List, Callable, Iterable, Union
import pandas as pd
import itertools
from collections import OrderedDict

import alglab.algorithm
import alglab.dataset
import alglab.results
import alglab.evaluation


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def resolve_parameters(fixed_parameters, varying_parameters):
    """
    Given a dictionary of fixed parameters and a dictionary of varying parameters, resolve the varying parameters
    which are defined by function.
    """
    resolved_parameters = {}
    dynamic_parameters = []
    for param_name, value in varying_parameters.items():
        if not callable(value):
            # This is a 'static' parameter
            resolved_parameters[param_name] = value
        else:
            # This is a dynamically defined parameter
            dynamic_parameters.append(param_name)

    # Resolve the dynamic parameters
    for param_name in dynamic_parameters:
        resolved_parameters[param_name] = varying_parameters[param_name](fixed_parameters | resolved_parameters)

    return resolved_parameters


class Experiment(object):

    def __init__(self, alg: alglab.algorithm.Algorithm, dataset: alglab.dataset.Dataset, params,
                 evaluators: List[alglab.evaluation.Evaluator] = None):
        """An experiment is a single instance of running an algorithm on a dataset with a set of parameters.
        The running time of the algorithm is measured by default. In addition to this, the evaluation_functions
        variable should contain a dictionary of methods which will be applied to the result of the algorithm.
        """
        self.dynamic = isinstance(dataset, alglab.dataset.DynamicDataset)
        self.alg = alg
        self.dataset = dataset
        self.params = params
        self.evaluators = evaluators

    def run(self):
        """Run the experiment."""
        # We always measure the running time of the experiment.
        experiment_gen = self.alg.run(self.dataset, self.params)

        experiment_ended = False
        total_running_time = 0
        iter = 0
        while not experiment_ended:
            try:
                start_time = time.time()
                alg_output = next(experiment_gen)
                #print(f"Iteration {iter} completed: {alg_output}")   
                #(alg_output.shape)     
                end_time = time.time()

                total_running_time += end_time - start_time
                result = {'iter': iter,
                          'iter_running_time_s': end_time - start_time,
                          'total_running_time_s': total_running_time}

                # Apply the evaluation functions
                if isinstance(self.dataset, alglab.dataset.DynamicDataset):
                    self.dataset.set_iteration(iter)
                if self.evaluators:
                    for evaluator in self.evaluators:
                        result[evaluator.name] = evaluator.apply(self.dataset, alg_output)
                #print(f"Iteration {iter} results: {result}")
                yield result

                iter += 1
            except StopIteration:
                experiment_ended = True


class ExperimentalSuite(object):

    def __init__(self,
                 algorithms: List[alglab.algorithm.Algorithm],
                 dataset: Union[Type[alglab.dataset.Dataset], alglab.dataset.Dataset],
                 results_filename: str,
                 num_runs: int = 1,
                 alg_fixed_params: Dict[str, Dict] = None,
                 alg_varying_params:  Dict[str, Dict] = None,
                 dataset_fixed_params: Dict = None,
                 dataset_varying_params: Dict = None,
                 evaluators: List[alglab.evaluation.Evaluator] = None):
        """Run a suite of experiments while varying some parameters.

        Varying parameter dictionaries should have parameter names as keys and the values should be an iterable containing:
            - values to be used directly; or
            - functions, taking fixed and statically defined variable parameters and returning a parameter value
        """
        self.num_runs = num_runs

        if num_runs < 1:
            raise ValueError('num_runs must be greater than or equal to 1')

        self.algorithms = algorithms
        self.algorithm_names = [alg.name for alg in self.algorithms]

        # Automatically populate the parameter dictionaries
        if alg_fixed_params is None:
            alg_fixed_params = {}
        if alg_varying_params is None:
            alg_varying_params = {}
        if dataset_fixed_params is None:
            dataset_fixed_params = {}
        if dataset_varying_params is None:
            dataset_varying_params = {}

        for alg in self.algorithms:
            alg_name = alg.name

            # Check that every algorithm has an entry in the params dictionary
            if alg_name not in alg_fixed_params:
                alg_fixed_params[alg_name] = {}
            if alg_name not in alg_varying_params:
                alg_varying_params[alg_name] = {}

            # Check that the parameters exist for the algorithm
            for param in alg_fixed_params[alg_name]:
                if param not in alg.parameter_names:
                    raise ValueError(f"Parameter {param} not configured for {alg_name} algorithm.")

            # Convert the parameter iterables to lists
            for param_name in alg_varying_params[alg_name].keys():
                alg_varying_params[alg_name][param_name] = list(alg_varying_params[alg_name][param_name])

        #  Convert parameter iterables to lists
        for param_name in dataset_varying_params.keys():
            dataset_varying_params[param_name] = list(dataset_varying_params[param_name])

        self.alg_fixed_params = alg_fixed_params
        self.alg_varying_params = alg_varying_params
        self.dataset = None
        self.dataset_class = None
        if isinstance(dataset, alglab.dataset.Dataset):
            self.dataset = dataset
        else:
            self.dataset_class = dataset
        self.dataset_fixed_params = dataset_fixed_params
        self.dataset_varying_params = dataset_varying_params
        self.evaluators = evaluators
        self.results_filename = results_filename

        self.results_columns = self.get_results_df_columns()

        # Compute the total number of experiments to run
        num_datasets = 1
        for param_name, values in self.dataset_varying_params.items():
            num_datasets *= len(values)
        self.num_experiments = 0
        for alg_name in self.algorithm_names:
            num_experiments_this_alg = 1
            for param_name, values in self.alg_varying_params[alg_name].items():
                num_experiments_this_alg *= len(values)
            self.num_experiments += num_experiments_this_alg * num_datasets
        self.num_trials = self.num_experiments * self.num_runs

        self.results = None

    def get_results_df_columns(self):
        """Create a list of all the columns in the results file and dataframe."""
        columns = ['trial_id', 'experiment_id', 'run_id', 'algorithm']
        for param_name in self.dataset_fixed_params.keys():
            columns.append(param_name)
        for param_name in self.dataset_varying_params.keys():
            columns.append(param_name)
        for alg_name in self.algorithm_names:
            for param_name in self.alg_fixed_params[alg_name].keys():
                columns.append(param_name)
            for param_name in self.alg_varying_params[alg_name].keys():
                columns.append(param_name)
        columns.append('iter')
        columns.append('iter_running_time_s')
        columns.append('total_running_time_s')
        for evaluator in self.evaluators:
            columns.append(evaluator.name)
        return list(OrderedDict.fromkeys(columns))

    def run_all(self, append_results=False) -> alglab.results.Results:
        """Run all the experiments in this suite."""

        # If we are appending the results, make sure that the header of the results file already matches the
        # header we would have written.
        if append_results:
            existing_results = alglab.results.Results(self.results_filename)
            if existing_results.column_names() != self.results_columns:
                raise ValueError("Cannot append results file: column names do not match.")
            true_trial_number = existing_results.results_df.iloc[-1]["trial_id"] + 1
            base_experiment_number = existing_results.results_df.iloc[-1]["experiment_id"] + 1
        else:
            true_trial_number = 1
            base_experiment_number = 1

        reported_trial_number = 1

        file_access_string = 'a' if append_results else 'w'

        with open(self.results_filename, file_access_string) as results_file:
            # Write the header line of the results file
            if not append_results:
                results_file.write(", ".join(self.results_columns))
                results_file.write("\n")

            for run in range(1, self.num_runs + 1):
                experiment_number = base_experiment_number
                for dataset_params in product_dict(**self.dataset_varying_params):
                    resolved_varying_dataset_params = resolve_parameters(self.dataset_fixed_params, dataset_params)
                    full_dataset_params = self.dataset_fixed_params | resolved_varying_dataset_params
                    dataset = self.dataset
                    if self.dataset is None:
                        dataset = self.dataset_class(**full_dataset_params)

                    for alg in self.algorithms:
                        alg_name = alg.name
                        for alg_params in product_dict(**self.alg_varying_params[alg_name]):
                            resolved_varying_alg_params = resolve_parameters(full_dataset_params | self.alg_fixed_params[alg_name], alg_params)
                            full_alg_params = self.alg_fixed_params[alg_name] | resolved_varying_alg_params
                            print(f"Trial {reported_trial_number} / {self.num_trials}: {alg_name} on {dataset} with parameters {full_alg_params}")
                            this_experiment = Experiment(alg, dataset, full_alg_params, self.evaluators)

                            for result in this_experiment.run():
                                this_result = result | full_dataset_params | full_alg_params | \
                                              {'algorithm': alg_name, 'trial_id': true_trial_number,
                                               'experiment_id': experiment_number, 'run_id': run}
                                results_file.write(", ".join([str(this_result[col]) if col in this_result else '' for col in self.results_columns]))
                                results_file.write("\n")
                                results_file.flush()

                            true_trial_number += 1
                            reported_trial_number += 1
                            experiment_number += 1

        # Create a dataframe from the results
        self.results = alglab.results.Results(self.results_filename)
        return self.results
