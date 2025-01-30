import sys
from pathlib import Path
import yaml
import argparse
import json
import numpy as np
import scipy as sc
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from joblib import Memory
memory = Memory(location='cache_dir', verbose=10)

from kalman_result import KalmanResults
from parflow_model import ToyProblem
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints
#from soil_model.evapotranspiration_fce import ET0
from auxiliary_functions import sqrt_func, add_noise
from data.load_data import load_data
from kalman_state import StateStructure
from scipy import linalg

######
# Unscented Kalman Filter for Parflow model
# see __main__ for details
######




def get_space_indices(grid_dz, mes_locations):
    return [int(mes_loc / grid_dz) for mes_loc in mes_locations]



class KalmanFilter:

    @staticmethod
    def from_config(workdir, config_path, verbose=False):
        with config_path.open("r") as f:
            config_dict = yaml.safe_load(f)
        return KalmanFilter(config_dict, workdir, verbose)


    def __init__(self, config, workdir, verbose=False):
        self.work_dir = Path(workdir)
        self.verbose = verbose
        np.random.seed(config["seed"])

        self.kalman_config = config["kalman_config"]

        self.model_config = config["model_config"]
        self.model = self._make_model()
        self.space_indices_train = get_space_indices(self.model.get_space_step(), self.kalman_config["mes_locations_train"])
        self.space_indices_test = get_space_indices(self.model.get_space_step(), self.kalman_config["mes_locations_test"])

        state_params = self.kalman_config["state_params"]
        state_params["train_meas"] = {"z_pos": self.kalman_config["mes_locations_train"]}
        state_params["test_meas"] = {"z_pos": self.kalman_config["mes_locations_test"]}

        self.state_struc = StateStructure(len(self.model.data_z), self.kalman_config["state_params"])
        # JB TODO: process all items of the main kalman config dict in the constructor or move all into run method
        # the class could also be replaced by a function
        # if "static_params" not in self.model_config:
        #     self.model_config["static_params"] = {}
        #
        # if "params" not in self.model_config:
        #     self.model_config["params"] = {}

        # if len(self.model_config["evapotranspiration_params"]['names']) != len(self.model_config["evapotranspiration_params"]['values']):
        #     raise ValueError("Evapotranspiration_params: The number of names and values do not match!")

        precipitation_list = []
        for (hours, precipitation) in self.model_config['rain_periods']:
            precipitation_list.extend([precipitation] * hours)
        self.model_config["precipitation_list"] = precipitation_list

        self.results = KalmanResults(workdir, self.model.data_z, self.state_struc, config['postprocess'])
        pass

    def _make_model(self):
        if self.model_config["model_class_name"] == "ToyProblem":
            model_class = ToyProblem
        else:
            raise NotImplemented("Import desired class")
        return model_class(self.model_config, workdir=self.work_dir / "output-toy")

    def plot_pressure(self):
        model = self._make_model()
        et_per_time = 0 #ET0(**dict(zip(self.model_config['evapotranspiration_params']["names"], self.model_config['evapotranspiration_params']["values"]))) / 1000 / 24  # mm/day to m/sec
        #model._run.Patch.top.BCPressure.alltime.Value = self.model_config["precipitation_list"][0] + et_per_time
        model.set_dynamic_params(self.model_config["params"]["names"], self.model_config["params"]["values"])

        model.run(init_pressure=None, precipitation_value=self.model_config["precipitation_list"][0] + et_per_time,
                  stop_time=self.model_config['rain_periods'][0][0])

        # model.save_pressure("pressure.png")
        #model.save_pressure("pressure.png")

    def run(self):
        #############################
        ### Generate measurements ###
        #############################
        if "measurements_dir" in self.kalman_config:
            noisy_measurements, noisy_measurements_to_test = load_data(data_dir=self.kalman_config["measurements_dir"], n_samples=len(self.model_config["precipitation_list"]))

            # Why to call model for real data?
            self.model.run(init_pressure=None, stop_time=1)

            sample_variance = np.var(noisy_measurements, axis=0)
            measurement_noise_covariance = np.diag(sample_variance)

            measurements_to_test = []
            measurements = []
        else:
            measurements, noisy_measurements, measurements_to_test, noisy_measurements_to_test, \
            state_data_iters = self.generate_measurements(self.kalman_config["measurements_data_name"])

            residuals = noisy_measurements - measurements
            measurement_noise_covariance = np.cov(residuals, rowvar=False)

        #self.model_config["grid_dz"] = self.model.get_space_step()
        # self.additional_data_len = len(self.kalman_config["mes_locations_train"]) + \
        #                            len(self.kalman_config["mes_locations_test"]) + \
        #                            len(KalmanFilter.get_nonzero_std_params(self.model_config["params"])) #+ \
        #                            #len(self.kalman_config["mes_locations_train_slope_intercept"])

        # if "flux_eps" in self.model_config:
        #     self.additional_data_len += 1

        self.results.ref_states = np.array(state_data_iters)
        #self.results.measuremnt_in = noisy_measurements
        # added during UKF loop

        print("state data iters ", np.array(state_data_iters).shape)
        self.results.plot_pressure(self.model, state_data_iters)

        #######################################
        ### Unscented Kalman filter setting ###
        ### - Sigma points
        ### - initital state covariance
        ### - UKF metrices
        ########################################
        ukf = self.set_kalman_filter(measurement_noise_covariance)

        #######################################
        ### Kalman filter run ###
        ### For each measurement (time step) ukf.update() and ukf.predict() are called
        ########################################
        self.run_kalman_filter(ukf, noisy_measurements)

        return self.results



    def model_run(self, kalman_step, pressure, params):
        flux = self.model_config["precipitation_list"][kalman_step]
        self.model.run(init_pressure=pressure, precipitation_value=flux,
                  state_params=params, start_time=kalman_step, stop_time=kalman_step+1)
        #new_pressure = self.model.get_data(current_time=1 / self.model._run.TimeStep.Value, data_name="pressure")
        new_pressure = self.model.get_data(current_time=kalman_step+1, data_name="pressure")
        return new_pressure

    def model_iteration(self, kalman_step, data_name, pressure, params):
        # et_per_time = ET0(n=14, T=20, u2=10, Tmax=27, Tmin=12, RHmax=0.55, RHmin=0.35,
        #                   month=6) / 1000 / 24  # mm/day to m/sec
        et_per_time = 0 #ET0(**dict(zip(self.model_config['evapotranspiration_params']["names"], self.model_config['evapotranspiration_params']["values"]))) / 1000 / 24  # mm/day to m/hour

        new_pressure = self.model_run(kalman_step, pressure, params)
        new_saturation = self.model.get_data(current_time=kalman_step, data_name="saturation")
        measurement = self.get_measurement(kalman_step+1, self.kalman_config["mes_locations_train"], data_name)
        measurement_to_test = self.get_measurement(kalman_step+1, self.kalman_config["mes_locations_test"], data_name)
        return measurement, measurement_to_test, new_pressure, new_saturation

    def generate_measurements(self, data_name):
        measurements = []
        measurements_to_test = []
        state_data_iters = []

        ###################
        ##   Model runs  ##
        ###################
        # Loop through time steps

        pressure_vec = self.model.make_linear_pressure(self.model_config)
        ref_params = self.state_struc.compose_ref_dict()
        ref_params['pressure_field'] = pressure_vec
        # JB TODO: finish GField.ref to simplify the following lines
        state_vec = self.state_struc.encode_state(ref_params)
        params_vec = state_vec[len(pressure_vec):]

        for i in range(0, len(self.model_config["precipitation_list"])):
            measurement_train, measurement_test, pressure_vec, sat_vec \
                = self.model_iteration(i, data_name, pressure_vec, ref_params)
            self.results.ref_saturation.append(sat_vec)
            measurements.append(measurement_train)
            measurements_to_test.append(measurement_test)

            if self.verbose:
                print("i: {}, data_pressure: {} ".format(i, pressure_vec))
            ref_params['pressure_field'] = pressure_vec
            ref_params['train_meas'] = measurement_train
            ref_params['test_meas'] = measurement_test
            iter_state = self.state_struc.encode_state(ref_params)
            state_data_iters.append(iter_state)

        noisy_measurements = KalmanFilter.add_noise_to_measurements(
                                    np.array(measurements),
                                    level=self.kalman_config["measurements_noise_level"], distr_type=self.kalman_config["noise_distr_type"])
        noisy_measurements_to_test = KalmanFilter.add_noise_to_measurements(
                                     np.array(measurements_to_test),
                                     level=self.kalman_config["measurements_noise_level"], distr_type=self.kalman_config["noise_distr_type"])

        measurements = np.array(measurements)
        measurements_to_test = np.array(measurements_to_test)
        noisy_measurements = np.array(noisy_measurements)
        noisy_measurements_to_test = np.array(noisy_measurements_to_test)

        # residuals = noisy_measurements - measurements
        # # print("residuals ", residuals)
        # # residuals = residuals * 2
        # measurement_noise_covariance = np.cov(residuals, rowvar=False)
        # print("measurement noise covariance ", measurement_noise_covariance)
        # # print("10 percent cov ", np.cov(noisy_measurements*0.1, rowvar=False))
        # # exit()
        # # pde.plot_kymograph(storage)
        return measurements, noisy_measurements, measurements_to_test, noisy_measurements_to_test, state_data_iters

    def get_measurement(self, kalman_step, mes_locations, data_name):
        #data.time= 0
        space_step = self.model.get_space_step()
        # times = self.model.get_times()
        # measurements = np.zeros((len(np.unique(times)), len(mes_locations)))
        space_indices = [int(mes_loc / space_step) for mes_loc in mes_locations]

        #current_time = 0

        if data_name == "saturation":
            data_to_measure = self.model.get_data(current_time=kalman_step, data_name="saturation")
        elif data_name == "pressure":
            data_to_measure = self.model.get_data(current_time=kalman_step, data_name="pressure")

        #    print("data to measure ", data_to_measure)

        measurement = np.flip(np.squeeze(data_to_measure))[space_indices]
        return measurement

    @staticmethod
    def add_noise_to_measurements(measurements, level=0.1, distr_type="uniform"):
        noisy_measurements = np.zeros(measurements.shape)
        for i in range(measurements.shape[1]):
            noisy_measurements[:, i] = add_noise(measurements[:, i], noise_level=level, distr_type=distr_type)
        return noisy_measurements


    #####################
    ### Kalman filter ###
    #####################
    def state_transition_function(self, state_vec, dt, time_step):
        print("dt: ", dt, "time step: ", time_step)
        state = self.state_struc.decode_state(state_vec)
        pressure_data = state["pressure_field"]
        #model_params_data = state_dict["model_params"]
        #model_params_dict = state_decode(self.kalman_config["params"], state_data)
        #pressure_data = state_data[0:-len_additional_data]  # Extract saturation from state vector
        model_params_data = []
        flux_eps_std = None

        # dynamic_model_params = KalmanFilter.get_nonzero_std_params(model_config["params"])
        # if len(dynamic_model_params) > 0:
        #     if "flux_eps" in model_config:
        #         model_params_data = state_vec[-len(dynamic_model_params) - 1:-1]
        #         flux_eps_std = state_vec[-1]
        #     else:
        #         model_params_data = state_vec[-len(dynamic_model_params):]

        #pressure_data = pressure_data.reshape(pressure_data.shape[0], 1, 1)

        #model = self.model_class(model_config, workdir=os.path.join(kalman_config["work_dir"], "output-toy"))

        et_per_time = 0 #ET0(**dict(zip(model_config['evapotranspiration_params']["names"],
                   #model_config['evapotranspiration_params']["values"]))) / 1000 / 24

        # if "Patch.top.BCPressure.alltime.Value" not in model_config["params"]:
        #     if flux_eps_std is not None:
        #         et_per_time += np.random.normal(model_config["flux_eps"][0], model_config["flux_eps"][1]**2)

        #model.set_raining(value=model_config["precipitation_list"][time_step] + et_per_time)
        #et_per_time += np.random.normal(0, 0.0001 ** 2)
        #model._run.Patch.top.BCPressure.alltime.Value = model_config["precipitation_list"][time_step] + et_per_time


        #model_params_new_values = list(np.array(list(dynamic_model_params.values()))[: 0])

        #if flux_eps_std is not None:
        #    model_params_new_values.append(0)  # eps value in next state

        percipitation = self.model_config["precipitation_list"][time_step]
        self.model.run(pressure_data, percipitation, state, start_time=time_step, stop_time=time_step+1)
        state["pressure_field"] = self.model.get_data(current_time=1 / self.model._run.TimeStep.Value, data_name="pressure")
        #data_pressure = model.get_data(current_time=1/model._run.TimeStep.Value, data_name="pressure")
        data_saturation = self.model.get_data(current_time=1 / self.model._run.TimeStep.Value, data_name="saturation")



        #next_state_init = list(KalmanFilter.squeeze_to_last(data_pressure))

        state["train_meas"] =  KalmanFilter.squeeze_to_last(np.flip(np.squeeze(data_saturation))
                                                            [self.space_indices_train])
        state["test_meas"] = KalmanFilter.squeeze_to_last(np.flip(KalmanFilter.squeeze_to_last(data_saturation))
                                                          [self.space_indices_test])
        #
        # if len(model_params_new_values) == 0:
        #     model_params_new_values = model_params_data

        # if len(model_params_new_values) > 0:
        #     next_state_init.extend(model_params_new_values)
        #n_nodes = len(model.nodes_z)
        #new_state_vec = np.concatenate([state_vec[:n_nodes], state_train, state_test, state_vec[n_nodes:]])
        new_state_vec = self.state_struc.encode_state(state)

        return new_state_vec

    @staticmethod
    def get_nonzero_std_params(model_params):
        print("model params ", model_params)
        # Filter out items with nonzero std
        filtered_data = {key: value for key, value in model_params.items() if value[1] != 0}
        return filtered_data

    def measurement_function(self, state_vec, space_indices_type=None):
        #if "mes_locations_train_slope_intercept" in kalman_config:
        #    slope_intercept = state_data[-len(kalman_config["mes_locations_train_slope_intercept"]):]
        #    state_data = state_data[:-len(kalman_config["mes_locations_train_slope_intercept"])]

        state = self.state_struc.decode_state(state_vec)
        # # JB TODO: get rid of
        # n_nodes = state_dict["pressure_field"].size()
        # len_dynamic_params = len(state_dict) - 1
        # if len_dynamic_params > 0:
        #     additional_data = state_data[-len_additional_data:-len_dynamic_params]
        # else:
        #     additional_data = state_data[-len_additional_data:]
        #
        # len_space_indices_train = len(get_space_indices(model_config["grid_dz"], kalman_config["mes_locations_train"]))
        # len_space_indices_test = len(get_space_indices(model_config["grid_dz"], kalman_config["mes_locations_test"]))
        #
        if space_indices_type is None:
            space_indices_type = "train"
        if space_indices_type == "train":
            measurements = state['train_meas']
            #print("train measurements ", measurements)
        elif space_indices_type == "test":
            measurements = state['test_meas']
            print("test measurements ", measurements)
        return measurements

    @staticmethod
    def squeeze_to_last(arr):
        squeezed = np.squeeze(arr)  # Remove all singleton dimensions
        return np.reshape(squeezed, (-1,))  # Reshape to 1D

    @staticmethod
    def get_sigma_points_obj(sigma_points_params, num_state_params):
        return MerweScaledSigmaPoints(n=num_state_params, sqrt_method=sqrt_func, **sigma_points_params, )


    # def get_fx_hx_function(self, model_config, kalman_config):
    #     return KalmanFilter.state_transition_function_wrapper(self.state_struc, self.model, model_config, kalman_config),\
    #            KalmanFilter.measurement_function_wrapper(self.state_struc, model_config, kalman_config),

    def set_kalman_filter(self,  measurement_noise_covariance):
        num_state_params = self.state_struc.size()
        dim_z = len(self.kalman_config["mes_locations_train"])  # Number of measurement inputs

        sigma_points_params = self.kalman_config["sigma_points_params"]

        #sigma_points = JulierSigmaPoints(n=n, kappa=1)
        sigma_points = KalmanFilter.get_sigma_points_obj(sigma_points_params, num_state_params)
        #sigma_points = MerweScaledSigmaPoints(n=num_state_params, alpha=sigma_points_params["alpha"], beta=sigma_points_params["beta"], kappa=sigma_points_params["kappa"], sqrt_method=sqrt_func)

        # Initialize the UKF filter
        time_step = 1 # one hour time step

        #fx_func, hx_func = self.get_fx_hx_function(self.model_config, self.kalman_config)

        ukf = UnscentedKalmanFilter(dim_x=num_state_params, dim_z=dim_z, dt=time_step,
                                    fx=self.state_transition_function, #KalmanFilter.state_transition_function_wrapper(len_additional_data=self.additional_data_len, model_config=self.model_config, kalman_config=self.kalman_config),
                                    hx=self.measurement_function, #KalmanFilter.measurement_function_wrapper(len_additional_data=self.additional_data_len, model_config=self.model_config, kalman_config=self.kalman_config),
                                    points=sigma_points)


        Q_state = self.state_struc.compose_Q()
        # n_nodes = len(self.model.nodes_z)
        # Q_pressure = Q_state[:n_nodes, :n_nodes]
        # Q_params = Q_state[n_nodes:, n_nodes:]
        # n_measure = num_state_params - len(Q_state)
        # Q_measure = np.zeros((n_measure, n_measure))
        ukf.Q = Q_state
        # len_dynamic_params = len(KalmanFilter.get_nonzero_std_params(self.model_config["params"]))
        # if "Q_model_params_var" in self.kalman_config and len_dynamic_params > 0:
        #     ukf.Q[:-len_dynamic_params] = ukf.Q[:-len_dynamic_params] * self.kalman_config["Q_var"]
        #     ukf.Q[-len_dynamic_params:] = ukf.Q[-len_dynamic_params:] * self.kalman_config["Q_model_params_var"]
        # else:
        #     ukf.Q = ukf.Q * self.kalman_config["Q_var"] #5e-5 # 5e-8

        print("ukf.Q.shape ", ukf.Q.shape)
        print("ukf.Q ", ukf.Q)

        #print("Q_discrete_white_noise(dim=1, dt=1.0, var=1e-6, block_size=num_locations)  ", Q_discrete_white_noise(dim=1, dt=1.0, var=5e-8, block_size=num_state_params))
        #ukf.Q = Q_discrete_white_noise(dim=1, dt=1.0, var=5e-8, block_size=num_state_params)  # Process noise covariance
        #print("Q discrete noise shape ", ukf.Q.shape)
        ukf.R = measurement_noise_covariance #* 1e6
        # print("R.shape ", ukf.R.shape)
        # exit()
        print("R measurement_noise_covariance ", measurement_noise_covariance)

        # best setting so far initial_covariance = np.eye(n) * 1e4, ukf.Q = np.ones(num_locations) * 1e-7
        #data.time = 0


        #print("space indices train ", space_indices_train)

        #initial_state_data = np.array([np.squeeze(data.pressure), np.squeeze(data.saturation[space_indices])]).flatten()#np.squeeze(data.pressure)
        #initial_state_data = list(np.squeeze(data.pressure))

        data_pressure = self.model.get_data(current_time=0, data_name="pressure")
        data_saturation = self.model.get_data(current_time=0, data_name="saturation")



        #print("data.saturation ", data.saturation)
        #print("data.saturation[space_indices_train] ", data.saturation[space_indices_train])
        #print("KalmanFilter.squeeze_to_last(data.saturation[space_indices_train]) ", KalmanFilter.squeeze_to_last(data.saturation[space_indices_train]))
        #print("np.squeeze(data.saturation[space_indices_train]) ", np.squeeze(data.saturation[space_indices_train], axis=tuple(range(data.saturation[space_indices_train].ndim - 1))))
        #print("list(np.squeeze(data.saturation[space_indices_train])) ", list(np.squeeze(data.saturation[space_indices_train], axis=tuple(range(data.saturation[space_indices_train].ndim - 1)))))


        # model_dynamic_params_mean_std = np.array(list(KalmanFilter.get_nonzero_std_params(self.model_config["params"]).values()))

        #noisy_model_params_values = []
        # params_std = []
        # if len(model_dynamic_params_mean_std) > 0:
        #     noisy_model_params_values = add_noise(model_dynamic_params_mean_std[:, 0], distr_type=self.kalman_config["noise_distr_type"],
        #                                           noise_level=self.kalman_config["model_params_noise_level"], std=list(model_dynamic_params_mean_std[:, 1]))
        #     params_std = model_dynamic_params_mean_std[:, 1]
        #print("noisy model params values ", noisy_model_params_values)
        # initial_state_std = np.ones(num_state_params)

        # if "pressure_saturation_data_std" in self.kalman_config:
        #     initial_state_std[:len(initial_state_data)] = self.kalman_config["pressure_saturation_data_std"]
        # else:
        #     np.abs(np.array(initial_state_data) * self.kalman_config["measurements_noise_level"])

        #initial_state_data.extend(noisy_model_params_values)

        # if "flux_eps" in self.model_config:
        #     params_std.append(0)
        #
        # var_models_params_std = np.array(params_std)
        # if len(var_models_params_std) > 0:
        #     initial_state_std[-len(params_std):] = var_models_params_std
        #
        # initial_state_covariance = np.zeros((num_state_params, num_state_params))
        # np.fill_diagonal(initial_state_covariance, np.array(initial_state_std)**2)

        init_mean, init_cov = self.state_struc.compose_init_state(self.model.data_z)
        init_state = self.state_struc.decode_state(init_mean)
        init_state["pressure_field"] =  add_noise(np.squeeze(data_pressure),
                                            noise_level=self.kalman_config["pressure_saturation_data_noise_level"],
                                            distr_type=self.kalman_config["noise_distr_type"])
        init_state["train_meas"] = add_noise(np.squeeze(data_saturation[self.space_indices_train]),
                                            noise_level=self.kalman_config["pressure_saturation_data_noise_level"],
                                            distr_type=self.kalman_config["noise_distr_type"])
        init_state["test_meas"] = add_noise(np.squeeze(data_saturation[self.space_indices_test]),
                                            noise_level=self.kalman_config["pressure_saturation_data_noise_level"],
                                            distr_type=self.kalman_config["noise_distr_type"])

        #cov_pressure = init_cov[:n_nodes, :n_nodes]
        #cov_params = init_cov[n_nodes:, n_nodes:]
        #cov_measure = np.eye(n_measure) * 1e-8
        #initial_cov = linalg.block_diag(cov_pressure, cov_measure, cov_params)
        # JB TODO: use init_mean, implement random choice of ref using init distr

        #
        # print("initial state std ", initial_state_std)
        # print("initial state std ", initial_state_std ** 2)

        #initial_state_covariance[-1] = 1e
        #ukf.Q[-1] = 5e-8

        # print("initial state data ", initial_state_data)
        # print("initial state covariance ", initial_state_covariance.shape)
        # print("initital state std ", initial_state_std)

        # if "flux_eps" in self.model_config:
        #     initial_state_data.append(self.model_config["flux_eps"][1])  # eps std to state

        #print("initial state data ", initial_state_data.shape)


        ukf.x = self.state_struc.encode_state(init_state) #initial_state_data #(state.data[int(0.3/0.05)], state.data[int(0.6/0.05)])#state  # Initial state vector
        ukf.P = init_cov  # Initial state covariance matrix

        # print("initital state data ", initial_state_data)
        # print("initial state covariance ", initial_state_covariance)


        return ukf

    def run_kalman_filter(self, ukf, noisy_measurements):
        pred_loc_measurements = []
        test_pred_loc_measurements = []
        pred_model_params = []
        pred_state_iter = []
        ukf_p_var_iter = []

        model_dynamic_params = KalmanFilter.get_nonzero_std_params(self.model_config["params"])
        print("noisy_meaesurements ", noisy_measurements)
        # Loop through measurements at each time step
        for time_step, measurement in enumerate(noisy_measurements):
            ukf.predict(time_step=time_step)
            ukf.update(measurement)
            print("sum ukf.P ", np.sum(ukf.P))
            print("Estimated State:", ukf.x)
            self.results.times.append(time_step)
            self.results.ukf_x.append(ukf.x)
            self.results.ukf_P.append(ukf.P)
            self.results.measuremnt_in.append(measurement)


            pred_state_iter.append(ukf.x)
            ukf_p_var_iter.append(np.diag(ukf.P))

            if len(model_dynamic_params) > 0:
                model_params_data = ukf.x[-len(model_dynamic_params):]
                pred_model_params.append(model_params_data)

            est_loc_measurements = self.measurement_function(ukf.x, space_indices_type="train")
            print("noisy measurement ", measurement)
            print("est loc measurements ", est_loc_measurements)

            test_est_loc_measurements =self.measurement_function(ukf.x, space_indices_type="test")

            pred_loc_measurements.append(est_loc_measurements)
            test_pred_loc_measurements.append(test_est_loc_measurements)

            #self.results.ukf_train_meas.append(est_loc_measurements)
            #self.results.ukf_test_meas.append(test_est_loc_measurements)

        pred_loc_measurements = np.array(pred_loc_measurements)
        test_pred_loc_measurements = np.array(test_pred_loc_measurements)
        ukf_p_var_iter = np.array(ukf_p_var_iter)

        ukf_last_p = ukf.P
        return self.results
        #return pred_loc_measurements, test_pred_loc_measurements, pred_model_params, pred_state_iter, ukf_p_var_iter, ukf_last_p



    def postprocess_data(self, state_data_iters, pred_state_data_iter):
        iter_mse_pressure_data = []
        iter_mse_train_measurements = []
        iter_mse_test_measurements = []

        iter_mse_model_config_data = {}

        print("len additional data ", self.additional_data_len)

        for state_data, pred_state_data  in zip(state_data_iters, pred_state_data_iter):
            print("len state data ", len(state_data))
            print("len pred state data ", len(pred_state_data))
            len(self.kalman_config["mes_locations_train"]) + len(self.kalman_config["mes_locations_test"]) + len(
                self.model_config["params"]["names"])

            if "flux_eps" in self.model_config:
                pred_state_data = pred_state_data[:-1]

            pressure_data = state_data[:-self.additional_data_len]
            pred_pressure_data = pred_state_data[:-self.additional_data_len]
            print("len pressure data ", len(pressure_data))

            print("pressure data ", pressure_data)
            print("pred pressure data ", pred_pressure_data)

            print("len pressure data ", len(pressure_data))
            print("len pred pressure data ", len(pred_pressure_data))

            iter_mse_pressure_data.append(np.linalg.norm(pressure_data - pred_pressure_data))

            train_measurements = state_data[-self.additional_data_len: -self.additional_data_len + len(self.kalman_config["mes_locations_train"])]
            pred_train_measurements = pred_state_data[-self.additional_data_len: -self.additional_data_len + len(
                self.kalman_config["mes_locations_train"])]

            iter_mse_train_measurements.append(np.linalg.norm(train_measurements - pred_train_measurements))

            if self.additional_data_len == len(self.kalman_config["mes_locations_train"]) + len(self.kalman_config["mes_locations_test"]):
                test_measurements = state_data[-self.additional_data_len + len(self.kalman_config["mes_locations_train"]):]
                pred_test_measurements = pred_state_data[ -self.additional_data_len + len(self.kalman_config["mes_locations_train"]):]
            else:
                test_measurements = state_data[
                                    -self.additional_data_len + len(self.kalman_config["mes_locations_train"]):
                                    -self.additional_data_len + len(self.kalman_config["mes_locations_train"])
                                    + len(self.kalman_config["mes_locations_test"])]

                pred_test_measurements = pred_state_data[
                                         -self.additional_data_len + len(self.kalman_config["mes_locations_train"]):
                                         -self.additional_data_len + len(self.kalman_config["mes_locations_train"])
                                         + len(self.kalman_config["mes_locations_test"])]

            iter_mse_test_measurements.append(np.linalg.norm(test_measurements - pred_test_measurements))

            if len(self.model_config["params"]["names"]) > 0:
                for idx, param_name in enumerate(self.model_config["params"]["names"]):
                    l2_norm = np.linalg.norm(state_data[-len(self.model_config["params"]["names"]) + idx] - pred_state_data[-len(self.model_config["params"]["names"]) + idx])

                    iter_mse_model_config_data.setdefault(param_name, []).append(l2_norm)


        print("iter_mse_pressure_data ", iter_mse_pressure_data)
        print("iter_mse_train_measurements ", iter_mse_train_measurements)
        print("iter_mse_test_measurements ", iter_mse_test_measurements)
        print("iter_mse_model_config_data ", iter_mse_model_config_data)



    # JB: How is serialization supposed to work?
    # What is point to deserialize only UKF but not KalmanFilter?
    # get_fx_hx_function - is called with wrong parameters anyway
    #
    # # Serialize only necessary attributes
    # @staticmethod
    # def serialize_kalman_filter(kf, filename):
    #     # Collect serializable attributes
    #     data = {
    #         'K': kf.K,
    #         'P': kf.P,
    #         'R': kf.R,
    #         'Q': kf.Q,
    #         'dim_x': kf._dim_x,
    #         'dim_z': kf._dim_z,
    #         'dt': kf._dt,
    #     }
    #     with filename.open('wb') as f:
    #         pickle.dump(data, f)
    #
    # @staticmethod
    # def deserialize_kalman_filter(filename, model_config, kalman_config, auxiliary_data, num_state_params):
    #     # Load data from the file
    #     with open(filename, 'rb') as f:
    #         data = pickle.load(f)
    #
    #     fx_func, hx_func = KalmanFilter.get_fx_hx_function(model_config, kalman_config, auxiliary_data["additional_data_len"])
    #
    #     sigma_points = KalmanFilter.get_sigma_points_obj(kalman_config["sigma_points_params"], num_state_params)
    #
    #     # Instantiate a new Kalman Filter object
    #     kf = UnscentedKalmanFilter(dim_x=data["dim_x"], dim_z=data["dim_z"], dt=data["dt"], fx=fx_func, hx=hx_func, points=sigma_points)
    #
    #     # Assign the attributes from the serialized data
    #     for attr, value in data.items():
    #         setattr(kf, attr, value)
    #     return kf




        # settings.set_working_directory(cwd)

#@memory.cache
def run_kalman(workdir, cfg_file):
    kalman_filter = KalmanFilter.from_config(Path(workdir), Path(cfg_file).resolve(), verbose=False)
    return kalman_filter.run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir', help='Path to work dir')
    parser.add_argument('config_file', help='Path to configuration file')
    args = parser.parse_args(sys.argv[1:])
    results = run_kalman(Path(args.work_dir), Path(args.config_file).resolve())
    results.postprocess()

if __name__ == "__main__":
    import cProfile
    import pstats
    main()
    # pr = cProfile.Profile()
    # pr.enable()

    # Configure ParFlow executable paths if needed
    #os.environ['PARFLOW_HOME'] = '/opt/parflow_install'
    #os.environ['PATH'] += ':/opt/parflow_install/bin'

    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('cumtime')
    # ps.print_stats(50)

