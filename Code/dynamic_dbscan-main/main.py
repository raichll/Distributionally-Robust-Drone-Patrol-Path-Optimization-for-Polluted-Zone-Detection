"""Main entry point to our experimental code."""
import argparse
import experiments


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument('command', type=str, choices=['plot', 'run'])
    parser.add_argument('experiment', type=str,
                        choices=['fig2', 'letters', 'mnist', 'fashion',
                                 'blobs', 'kdd', 'covertype','truck'])
    return parser.parse_args()



def main():
    args = parse_args()

    if args.command == 'plot':
        experiments.plot_results()
    else:
        if args.experiment == "letters":
            experiments.letters_experiment()
        elif args.experiment == "mnist":
            experiments.mnist_experiment()
        elif args.experiment == "fashion":
            experiments.fashion_mnist_experiment()
        elif args.experiment == "blobs":
            experiments.blobs_experiment()
        elif args.experiment == "kdd":
            experiments.kdd_experiment()
        elif args.experiment == "covertype":
            experiments.covertype_experiment()
        elif args.experiment == "truck":
            experiments.truck_experiment()
        else:
            print("Invalid experiment specified.")

import sys

sys.argv = [
    'main.py',   # 脚本名
    'run',              # <-- command 参数
    'truck'  # <-- experiment 参数
]


if __name__ == "__main__":
    main()
