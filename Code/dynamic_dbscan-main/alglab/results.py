"""Classes and method for processing experimental results."""
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import numpy as np

plt.rcParams.update({
    "text.usetex": False,
})

class Results(object):

    def __init__(self, results_filename):
        """Load results from a csv file."""
        self.results_df = pd.read_csv(results_filename, skipinitialspace=True)
        self.algorithm_names = self.results_df['algorithm'].unique()
        self.num_runs = self.results_df['run_id'].max()
        self.stats_df = self.create_averaged_stats()

    def create_averaged_stats(self):
        """
        Create averages and error bars from multiple runs.
        """
        alg_dfs = {alg_name: self.results_df.loc[self.results_df['algorithm'] == alg_name] for alg_name in
                  self.algorithm_names}
        stats_df = pd.DataFrame()
        for alg_name, alg_df in alg_dfs.items():
            mean_df = alg_df.groupby(['experiment_id', 'iter']).mean(numeric_only=True)
            mean_df = mean_df.drop(['trial_id', 'run_id'], axis=1).add_prefix('_mean_')
            sem_df = alg_df.groupby(['experiment_id', 'iter']).sem(numeric_only=True)
            sem_df = sem_df.drop(['trial_id', 'run_id'], axis=1).add_prefix('_sem_')
            sem_df['algorithm'] = alg_name
            sem_df = sem_df.join(mean_df)
            stats_df = pd.concat([stats_df, sem_df])
        stats_df.reset_index(inplace=True)
        return stats_df

    def column_names(self) -> List[str]:
        return self.results_df.columns.values.tolist()

    def line_plot(self, x_col, y_col, filename=None,
                  ignore_algorithms=None,
                  fixed_parameters=None,
                  x_range=None,
                  x_label=None,
                  y_label=None,
                  algorithm_names=None,
                  show_legend=True,):
        """Plot one column of the dataframe against another."""
        if ignore_algorithms is None:
            ignore_algorithms = []
        if fixed_parameters is None:
            fixed_parameters = {}
        if algorithm_names is None:
            algorithm_names = {}

        fig, ax = plt.subplots(figsize=(4, 3))
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(False)

        if x_col not in ['iter']:
            x_col = f"_mean_{x_col}"

        filtered_stats_df = self.stats_df
        for param, val in fixed_parameters.items():
            filtered_stats_df = filtered_stats_df[filtered_stats_df[f"_mean_{param}"] == val]

        for alg_name in self.algorithm_names:
            if alg_name not in ignore_algorithms:
                this_alg_name = algorithm_names[alg_name] if alg_name in algorithm_names else alg_name
                this_alg_results = filtered_stats_df[(filtered_stats_df['algorithm'] == alg_name)]
                plt.plot(this_alg_results[x_col],
                         this_alg_results[f"_mean_{y_col}"],
                         linewidth=3,
                         label=this_alg_name)
                if self.num_runs > 1:
                    plt.fill_between(this_alg_results[x_col],
                                     this_alg_results[f"_mean_{y_col}"] - this_alg_results[f"_sem_{y_col}"],
                                     this_alg_results[f"_mean_{y_col}"] + this_alg_results[f"_sem_{y_col}"],
                                     alpha=0.2)

        # Set the limits of the plot if specified
        if x_range is not None:
            ax.set_xlim(x_range[0], x_range[1])

        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)

        if show_legend:
            plt.legend()
        if filename:
            plt.savefig(filename, format="pdf", bbox_inches="tight")
        #plt.show(block=False)

    def bar_plot(self, x_col, y_col, x_vals, filename=None,
                 ignore_algorithms=None):
        if ignore_algorithms is None:
            ignore_algorithms = []

        x = np.arange(len(x_vals))  # the label locations
        width = 0.75 / (len(self.algorithm_names) - len(ignore_algorithms))  # the width of the bars

        fig, ax = plt.subplots(layout='constrained')

        multiplier = 0
        for alg_name in self.algorithm_names:
            if alg_name not in ignore_algorithms:
                offset = width * multiplier
                alg_results = self.stats_df[(self.stats_df['algorithm'] == alg_name)]
                y_vals = []
                for x_val in x_vals:
                    y_vals.append(alg_results[alg_results[f"_mean_{x_col}"] == x_val][f"_mean_{y_col}"].values[0])
                rects = ax.bar(x + offset, y_vals, width, label=alg_name)
                ax.bar_label(rects, padding=3)
                multiplier += 1

        plt.legend()
        if filename:
            plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.show()

    def get_average_stat(self, stat, fixed_parameters=None):
        if fixed_parameters is None:
            fixed_parameters = {}

        output = {}

        filtered_stats_df = self.stats_df
        for param, val in fixed_parameters.items():
            filtered_stats_df = filtered_stats_df[filtered_stats_df[f"_mean_{param}"] == val]

        for alg_name in self.algorithm_names:
            this_alg_results = filtered_stats_df[(filtered_stats_df['algorithm'] == alg_name)]
            mean_stat = this_alg_results[f"_mean_{stat}"].mean()
            output[alg_name] = mean_stat

        return output

    def get_max_stat(self, stat, fixed_parameters=None):
        if fixed_parameters is None:
            fixed_parameters = {}

        output = {}

        filtered_stats_df = self.stats_df
        for param, val in fixed_parameters.items():
            filtered_stats_df = filtered_stats_df[filtered_stats_df[f"_mean_{param}"] == val]

        for alg_name in self.algorithm_names:
            this_alg_results = filtered_stats_df[(filtered_stats_df['algorithm'] == alg_name)]
            max_stat = this_alg_results[f"_mean_{stat}"].max()
            output[alg_name] = max_stat

        return output

    def get_final_stat(self, stat, fixed_parameters=None):
        if fixed_parameters is None:
            fixed_parameters = {}

        output = {}

        filtered_stats_df = self.stats_df
        for param, val in fixed_parameters.items():
            filtered_stats_df = filtered_stats_df[filtered_stats_df[f"_mean_{param}"] == val]

        for alg_name in self.algorithm_names:
            this_alg_results = filtered_stats_df[(filtered_stats_df['algorithm'] == alg_name)]
            final_stat = this_alg_results.iloc[-1][f"_mean_{stat}"]
            final_err = this_alg_results.iloc[-1][f"_sem_{stat}"]
            output[alg_name] = (final_stat, final_err)

        return output


