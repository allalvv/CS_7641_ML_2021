
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import argparse
import warnings
from experiments import run_experiments, run_experiments_best
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--all", help="Run all experiments", action='store_true')
    g.add_argument("-e", help="Run specific experiment: dt, knn, ann, svm, ada")
    g.add_argument("--best", help="Run best experiment results for dt, knn, ann, svm, ada: best", action='store_false')
    args = parser.parse_args()
    return args


def run():

    args = parse_args()
    if args.all:
        print("Running all experiments")
        run_experiments(all=True)
    else:
        if not args.best:
            print("Running best experiment results")
            run_experiments_best()
        else:
            if args.e.lower() not in ['dt', 'knn', 'ann', 'svm', 'ada']:
                raise ValueError("Invalid experiment, please select from following: ada, dt, knn, nn, svm")
            else:
                print("Running model: ", args.e.lower())
                run_experiments(experiment=args.e)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    verbose = True

    run()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
