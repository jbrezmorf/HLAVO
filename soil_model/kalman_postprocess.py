import os
import sys
import yaml
import argparse
import json
import numpy as np
import scipy as sc
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from parflow.tools import settings
from soil_model.parflow_model import ToyProblem
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints
from soil_model.evapotranspiration_fce import ET0
from soil_model.auxiliary_functions import sqrt_func, set_nested_attr, get_nested_attr, add_noise
from soil_model.data.load_data import load_data
from soil_model.kalman import KalmanFilter



def postprocess_kalman_results(work_dir):
    # Serialize the Kalman filter object

    with open(os.path.join(work_dir, "auxiliary_data.json"), 'r') as f:
        auxiliary_data = json.load(f)
    with open(os.path.join(work_dir, "model_config.json"), 'r') as f:
        model_config = json.load(f)
    with open(os.path.join(work_dir, "kalman_config.json"), 'r') as f:
        kalman_config = json.load(f)

    noisy_measurements = np.load(os.path.join(work_dir, "noisy_measurements.npy"))
    pred_loc_measurements = np.load(os.path.join(work_dir, "pred_loc_measurements.npy"))

    pred_model_params = np.load(os.path.join(work_dir, "pred_model_params.npy"))

    noisy_measurements_to_test = np.load(os.path.join(work_dir, "noisy_measurements_to_test.npy"))
    test_pred_loc_measurements = np.load(os.path.join(work_dir, "test_pred_loc_measurements.npy"))

    pred_state_data_iter = np.load(os.path.join(work_dir, "pred_state_data_iter.npy"))
    ukf_p_var_iter = np.load(os.path.join(work_dir, "ukf_p_var_iter.npy"))
    ukf_last_P = np.load(os.path.join(work_dir, "ukf_last_P.npy"))

    times = np.load(os.path.join(work_dir, "times.npy"))

    pred_loc_measurements_variances = np.load(os.path.join(work_dir, "pred_loc_measurements_variances.npy"))
    test_pred_loc_measurements_variances = np.load(os.path.join(work_dir, "test_pred_loc_measurements_variances.npy"))

    ukf = KalmanFilter.deserialize_kalman_filter(os.path.join(work_dir, "kalman_filter.pkl"), model_config, kalman_config, auxiliary_data, num_state_params=ukf_last_P.shape[0])

    plot_results(pred_loc_measurements, test_pred_loc_measurements,
                 noisy_measurements_to_test, pred_model_params,  noisy_measurements, ukf_p_var_iter, times, kalman_config)


    KalmanFilter.plot_heatmap(cov_matrix=ukf_last_P)


def plot_results(pred_loc_measurements, test_pred_loc_measurements, noisy_measurements_to_test, pred_model_params,
                 noisy_measurements, ukf_p_var_iter, times, kalman_config):

    measurements = []
    measurements_to_test = []
    pred_loc_measurements_variances = []
    test_pred_loc_measurements_variances = []

    measurements_data_name = kalman_config["measurements_data_name"]

    print("pred_loc_measurements ", pred_loc_measurements)

    KalmanFilter.plot_measurements(times, measurements, noisy_measurements, pred_loc_measurements,
                                   pred_loc_measurements_variances, measurements_data_name, title_prefix="train_")
    KalmanFilter.plot_measurements(times, measurements_to_test, noisy_measurements_to_test, test_pred_loc_measurements,
                                   test_pred_loc_measurements_variances, measurements_data_name, title_prefix="test_")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir', help='Path to work dir')
    parser.add_argument('config_file', help='Path to configuration file')
    args = parser.parse_args(sys.argv[1:])

    work_dir = args.work_dir

    postprocess_kalman_results(work_dir=work_dir)
