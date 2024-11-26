import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
#from sklearn.linear_model import LinearRegression


def load_data(data_dir, n_samples):

    #data_dir = "/home/martin/Documents/HLAVO/soil_model/24_10_01_full_saturation/"

    pr2_data_file = os.path.join(data_dir, "pr2_data_filtered.csv")
    odyssey_data_file = os.path.join(data_dir, "odyssey_data_filtered.csv")

    # Load the CSV file into a pandas DataFrame
    pr2_data = pd.read_csv(pr2_data_file)
    odyssey_data = pd.read_csv(odyssey_data_file)

    min_idx = 264
    max_idx = 1992

    x_range = np.arange(min_idx, max_idx)

    # Define the window size (e.g., 3-point moving average)
    window_size = 50

    odyssey_0 = odyssey_data["odyssey_0"][min_idx:max_idx]
    odyssey_1 = odyssey_data["odyssey_1"][min_idx:max_idx]
    odyssey_2 = odyssey_data["odyssey_2"][min_idx:max_idx]
    odyssey_3 = odyssey_data["odyssey_3"][min_idx:max_idx]
    odyssey_4 = odyssey_data["odyssey_4"][min_idx:max_idx]

    pr2_0 = pr2_data["SoilMoistMin_0"][min_idx:max_idx]
    pr2_1 = pr2_data["SoilMoistMin_1"][min_idx:max_idx]
    pr2_2 = pr2_data["SoilMoistMin_2"][min_idx:max_idx]
    pr2_3 = pr2_data["SoilMoistMin_3"][min_idx:max_idx]
    pr2_4 = pr2_data["SoilMoistMin_4"][min_idx:max_idx]
    pr2_5 = pr2_data["SoilMoistMin_5"][min_idx:max_idx]

    # Apply the moving average filter
    pr2_0_smoothed = pr2_0.rolling(window=window_size, min_periods=1).mean()
    pr2_1_smoothed = pr2_1.rolling(window=window_size, min_periods=1).mean()
    pr2_2_smoothed = pr2_2.rolling(window=window_size, min_periods=1).mean()
    pr2_5_smoothed = pr2_5.rolling(window=window_size, min_periods=1).mean()
    pr2_4_smoothed = pr2_4.rolling(window=window_size, min_periods=1).mean()
    pr2_3_smoothed = pr2_3.rolling(window=window_size, min_periods=1).mean()

    # One value per hour
    pr2_0_smoothed = pr2_0_smoothed[::12]
    pr2_1_smoothed = pr2_1_smoothed[::12]
    pr2_2_smoothed = pr2_2_smoothed[::12]
    pr2_3_smoothed = pr2_3_smoothed[::12]
    pr2_4_smoothed = pr2_4_smoothed[::12]
    pr2_5_smoothed = pr2_5_smoothed[::12]

    #train_data = np.array([pr2_0_smoothed, pr2_1_smoothed, pr2_2_smoothed, pr2_4_smoothed, pr2_5_smoothed]).T
    train_data = np.array([pr2_5_smoothed[:n_samples]]).T
    test_data = np.array([pr2_3_smoothed[:n_samples]]).T

    return train_data, test_data


