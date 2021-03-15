import argparse
from datetime import datetime
import logging

import random as rand
import numpy as np

from data import loader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(experiment_detals, experiment, timing_key, verbose, timings):
    t = datetime.now()
    for details in experiment_detals:
        logger.info("Running {} experiment: {}".format(timing_key, details.ds_readable_name))
        exp = experiment(details, verbose=verbose)
        exp.perform()
    t_d = datetime.now() - t
    timings[timing_key] = t_d.seconds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for Randomized Optimization')
    parser.add_argument('--dump_data', action='store_true', help='Build train/test/validate splits '
                                                                 'and save to the data folder '
                                                                 '(should only need to be done once)')
    args = parser.parse_args()
    verbose = True
    seed = 1
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1, dtype='uint64')
        logger.info("Using seed {}".format(seed))
        np.random.seed(seed)
        rand.seed(seed)

    logger.info("Loading data")
    logger.info("----------")

    ds1_details = {
        'data': loader.Covid_19_ICU_Admission(verbose=verbose, seed=seed),
        'name': 'Covid_19_ICU_Admission',
        'readable_name': 'Covid-19_ICU_Admission'}
    ds2_details = {
        'data': loader.CreditCardApproval(verbose=verbose, seed=seed),
        'name': 'CreditCardApproval',
        'readable_name': 'Credit Card Approval'}

    datasets = [
        ds1_details,
        ds2_details
    ]

    experiment_details = []
    for ds in datasets:
        data = ds['data']
        data.load_and_process()
        if args.dump_data:
            data.dump_test_train_val(test_size=0.2, random_state=seed)
