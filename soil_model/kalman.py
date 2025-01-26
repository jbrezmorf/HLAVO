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
from soil_model.parflow_model import ToyProblem
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints
from soil_model.evapotranspiration_fce import ET0
from soil_model.auxiliary_functions import sqrt_func, add_noise
from soil_model.data.load_data import load_data


######
# Unscented Kalman Filter for Parflow model
# see __main__ for details
######


def get_space_indices(grid_dz, mes_locations):
    return [int(mes_loc / grid_dz) for mes_loc in mes_locations]


class KalmanFilter:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('work_dir', help='Path to work dir')
        parser.add_argument('config_file', help='Path to configuration file')
        args = parser.parse_args(sys.argv[1:])

        self.verbose = False
        self.work_dir = args.work_dir

        os.chdir(self.work_dir)

        config_file_path = os.path.abspath(args.config_file)
        config = KalmanFilter.load_config(config_file_path)

        self.model_config = config["model_config"]
        self.kalman_config = config["kalman_config"]

        if self.model_config["model_class_name"] == "ToyProblem":
            self.model_class = ToyProblem
        else:
            raise NotImplemented("Import desired class")

        self.kalman_config["work_dir"] = self.work_dir

        if "static_params" not in self.model_config:
            self.model_config["static_params"] = {}

        if "params" not in self.model_config:
            self.model_config["params"] = {}

        if len(self.model_config["evapotranspiration_params"]['names']) != len(self.model_config["evapotranspiration_params"]['values']):
            raise ValueError("Evapotranspiration_params: The number of names and values do not match!")

        precipitation_list = []
        for (hours, precipitation) in self.model_config['rain_periods']:
            precipitation_list.extend([precipitation] * hours)
        self.model_config["precipitation_list"] = precipitation_list

        np.random.seed(config["seed"])

    def plot_pressure(self):
        model = self.model_class(self.model_config, workdir=os.path.join(self.work_dir, "output-toy"))

        et_per_time = 0 #ET0(**dict(zip(self.model_config['evapotranspiration_params']["names"], self.model_config['evapotranspiration_params']["values"]))) / 1000 / 24  # mm/day to m/sec
        #model._run.Patch.top.BCPressure.alltime.Value = self.model_config["precipitation_list"][0] + et_per_time
        iter_values = []
        iter_values = model.set_dynamic_params(self.model_config["params"]["names"], self.model_config["params"]["values"])

        model.run(init_pressure=None, precipitation_value=self.model_config["precipitation_list"][0] + et_per_time, stop_time=self.model_config['rain_periods'][0][0])

        # model.save_pressure("pressure.png")
        #model.save_pressure("pressure.png")

    def run(self):
        #############################
        ### Generate measurements ###
        #############################
        model = self.model_class(self.model_config, workdir=os.path.join(self.work_dir, "output-toy"))
        if "measurements_dir" in self.kalman_config:
            noisy_measurements, noisy_measurements_to_test = load_data(data_dir=self.kalman_config["measurements_dir"], n_samples=len(self.model_config["precipitation_list"]))

            model.run(init_pressure=None, stop_time=1)

            sample_variance = np.var(noisy_measurements, axis=0)
            measurement_noise_covariance = np.diag(sample_variance)

            measurements_to_test = []
            measurements = []
        else:
            model, measurements, noisy_measurements, measurements_to_test, noisy_measurements_to_test, \
            state_data_iters, last_time_step_data_pressure = self.generate_measurements(model, self.kalman_config["measurements_data_name"])

            residuals = noisy_measurements - measurements
            measurement_noise_covariance = np.cov(residuals, rowvar=False)

        self.model_config["grid_dz"] = model.get_space_step()
        self.additional_data_len = len(self.kalman_config["mes_locations_train"]) + \
                                   len(self.kalman_config["mes_locations_test"]) + \
                                   len(KalmanFilter.get_nonzero_std_params(self.model_config["params"])) #+ \
                                   #len(self.kalman_config["mes_locations_train_slope_intercept"])

        if "flux_eps" in self.model_config:
            self.additional_data_len += 1

        if self.kalman_config["plot"]:
            print("state data iters ", np.array(state_data_iters).shape)
            KalmanFilter.plot_pressure_from_state_data(model, state_data_iters, self.additional_data_len)


        space_indices_train = [int(mes_loc / model.get_space_step()) for mes_loc in self.kalman_config["mes_locations_train"]]
        space_indices_test = [int(mes_loc / model.get_space_step()) for mes_loc in self.kalman_config["mes_locations_test"]]

        #######################################
        ### Unscented Kalman filter setting ###
        ### - Sigma points
        ### - initital state covariance
        ### - UKF metrices
        ########################################
        print("state_data_iters ", state_data_iters)
        ukf = self.set_kalman_filter(model, measurement_noise_covariance, state_length=np.array(state_data_iters).shape[1])

        #######################################
        ### Kalman filter run ###
        ### For each measurement (time step) ukf.update() and ukf.predict() are called
        ########################################
        pred_loc_measurements, test_pred_loc_measurements, pred_model_params, pred_state_data_iter,\
        ukf_p_var_iter, ukf_last_P = self.run_kalman_filter(ukf, noisy_measurements, space_indices_train, space_indices_test)


        ##############################
        ### Results postprocessing ###
        ##############################
        # Serialize the Kalman filter object
        KalmanFilter.serialize_kalman_filter(ukf, os.path.join(self.work_dir, "kalman_filter.pkl"))

        auxiliary_data = {}
        auxiliary_data["additional_data_len"] = self.additional_data_len

        with open(os.path.join(self.work_dir, "auxiliary_data.json"), 'w') as f:
            json.dump(auxiliary_data, f)
        with open(os.path.join(self.work_dir, "model_config.json"), 'w') as f:
            json.dump(self.model_config, f)
        with open(os.path.join(self.work_dir, "kalman_config.json"), 'w') as f:
            json.dump(self.kalman_config, f)

        np.save(os.path.join(self.work_dir, "noisy_measurements"), noisy_measurements)
        np.save(os.path.join(self.work_dir, "pred_loc_measurements"), pred_loc_measurements)
        np.save(os.path.join(self.work_dir, "pred_model_params"), pred_model_params)
        np.save(os.path.join(self.work_dir, "noisy_measurements_to_test"), noisy_measurements_to_test)
        np.save(os.path.join(self.work_dir, "test_pred_loc_measurements"), test_pred_loc_measurements)
        np.save(os.path.join(self.work_dir, "pred_state_data_iter"), pred_state_data_iter)
        np.save(os.path.join(self.work_dir, "ukf_p_var_iter"), ukf_p_var_iter)
        np.save(os.path.join(self.work_dir, "ukf_last_P"), ukf_last_P)

        self.plot_results(pred_loc_measurements, test_pred_loc_measurements, measurements_to_test,
                     noisy_measurements_to_test, pred_model_params, measurements, noisy_measurements, self.kalman_config["measurements_data_name"], ukf_p_var_iter)

        if self.kalman_config["plot"]:
            self.plot_heatmap(cov_matrix=ukf_last_P)

        #self.postprocess_data(state_data_iters, pred_state_data_iter)

    def model_iteration(self, model, flux_bc_pressure, data_name, data_pressure=None):
        # et_per_time = ET0(n=14, T=20, u2=10, Tmax=27, Tmin=12, RHmax=0.55, RHmin=0.35,
        #                   month=6) / 1000 / 24  # mm/day to m/sec
        et_per_time = 0 #ET0(**dict(zip(self.model_config['evapotranspiration_params']["names"], self.model_config['evapotranspiration_params']["values"]))) / 1000 / 24  # mm/day to m/hour

        model.run(init_pressure=data_pressure, precipitation_value=flux_bc_pressure + et_per_time, model_params=self.model_config["params"], stop_time=1)

        data_pressure = model.get_data(current_time=1 / model._run.TimeStep.Value, data_name="pressure")

        measurement, last_time_step_data_pressure = self.get_measurements(model, space_step=model.get_space_step(),
                                       mes_locations=self.kalman_config["mes_locations_train"], data_name=data_name)

        measurement_to_test, last_time_step_data_pressure = self.get_measurements(model, space_step=model.get_space_step(),
                                               mes_locations=self.kalman_config["mes_locations_test"], data_name=data_name)

        iter_values = np.array(list(KalmanFilter.get_nonzero_std_params(self.model_config["params"]).values()))

        if len(iter_values) > 0:
            iter_values = iter_values[:, 0] #list(self.model_config["params"].values())

        iter_state = list(np.squeeze(data_pressure))
        iter_state.extend(list(np.squeeze(measurement[0])))
        iter_state.extend(list(np.squeeze(measurement_to_test[0])))
        iter_state.extend(iter_values)

        return model, measurement[0], measurement_to_test[0], iter_values, last_time_step_data_pressure

    def generate_measurements(self, model, data_name):
        measurements = []
        measurements_to_test = []
        state_data_iters = []

        ###################
        ##   Model runs  ##
        ###################
        # Loop through time steps
        for i in range(0, len(self.model_config["precipitation_list"])):
            if i == 0:
                data_pressure = None

            model, measurement_train, measurement_test, iter_values, last_time_step_data_pressure = self.model_iteration(model, self.model_config["precipitation_list"][i], data_name, data_pressure=data_pressure)
            measurements.append(measurement_train)
            measurements_to_test.append(measurement_test)

            data_pressure = last_time_step_data_pressure[1]

            if self.verbose:
                print("i: {}, data_pressure: {} ".format(i, data_pressure))

            iter_state = list(np.squeeze(data_pressure))
            iter_state.extend(list(np.squeeze(measurement_train)))
            iter_state.extend(list(np.squeeze(measurement_test)))
            iter_state.extend(iter_values)

            state_data_iters.append(iter_state)

        noisy_measurements = KalmanFilter.add_noise_to_measurements(np.array(measurements), level=self.kalman_config["measurements_noise_level"], distr_type=self.kalman_config["noise_distr_type"])
        noisy_measurements_to_test = KalmanFilter.add_noise_to_measurements(np.array(measurements_to_test), level=self.kalman_config["measurements_noise_level"], distr_type=self.kalman_config["noise_distr_type"])

        measurements = np.array(measurements)
        measurements_to_test = np.array(measurements_to_test)
        noisy_measurements = np.array(noisy_measurements)
        noisy_measurements_to_test = np.array(noisy_measurements_to_test)

        times = np.arange(1, len(measurements) + 1, 1)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        axes.scatter(times, measurements[:, 0], marker="o", label="measurements")
        axes.scatter(times, noisy_measurements[:, 0], marker='x', label="noisy measurements")
        axes.set_xlabel("time")
        axes.set_ylabel(data_name)
        fig.savefig("L2_coarse_L1_fine_samples.pdf")
        fig.legend()
        plt.show()

        if measurements.shape[1] > 1:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            axes.scatter(times, measurements[:, 1], marker="o", label="measurements")
            axes.scatter(times, noisy_measurements[:, 1], marker='x', label="noisy measurements")
            axes.set_xlabel("time")
            axes.set_ylabel(data_name)
            fig.legend()
            plt.show()

        # residuals = noisy_measurements - measurements
        # # print("residuals ", residuals)
        # # residuals = residuals * 2
        # measurement_noise_covariance = np.cov(residuals, rowvar=False)
        # print("measurement noise covariance ", measurement_noise_covariance)
        # # print("10 percent cov ", np.cov(noisy_measurements*0.1, rowvar=False))
        # # exit()
        # # pde.plot_kymograph(storage)
        return model, measurements, noisy_measurements, measurements_to_test, noisy_measurements_to_test, state_data_iters, last_time_step_data_pressure

    def get_measurements(self, model, space_step, mes_locations=None, data_name="pressure"):
        #data.time= 0
        times = model.get_times()
        measurements = np.zeros((len(np.unique(times)), len(mes_locations)))
        space_indices = [int(mes_loc / space_step) for mes_loc in mes_locations]

        current_time = 0
        for data_t in times:
            if data_name == "saturation":
                data_to_measure = model.get_data(current_time=current_time, data_name="saturation")
            elif data_name == "pressure":
                data_to_measure = model.get_data(current_time=current_time, data_name="pressure")

            print("data to measure ", data_to_measure)

            measurements[data_t, :] = np.flip(np.squeeze(data_to_measure))[space_indices]
            current_time += 1

        return measurements, (current_time-1, model.get_data(current_time=current_time-1, data_name="pressure"))

    @staticmethod
    def add_noise_to_measurements(measurements, level=0.1, distr_type="uniform"):
        noisy_measurements = np.zeros(measurements.shape)
        for i in range(measurements.shape[1]):
            noisy_measurements[:, i] = add_noise(measurements[:, i], noise_level=level, distr_type=distr_type)
        return noisy_measurements


    #####################
    ### Kalman filter ###
    #####################
    @staticmethod
    def state_transition_function_wrapper(len_additional_data, model, model_config, kalman_config):
        def state_transition_function(state_data, dt, time_step):
            pressure_data = state_data[0:-len_additional_data]  # Extract saturation from state vector
            model_params_data = []
            flux_eps_std = None

            dynamic_model_params = KalmanFilter.get_nonzero_std_params(model_config["params"])
            if len(dynamic_model_params) > 0:
                if "flux_eps" in model_config:
                    model_params_data = state_data[-len(dynamic_model_params)-1:-1]
                    flux_eps_std = state_data[-1]
                else:
                    model_params_data = state_data[-len(dynamic_model_params):]

            pressure_data = pressure_data.reshape(pressure_data.shape[0], 1, 1)

            #model = self.model_class(model_config, workdir=os.path.join(kalman_config["work_dir"], "output-toy"))

            et_per_time = 0 #ET0(**dict(zip(model_config['evapotranspiration_params']["names"],
                       #model_config['evapotranspiration_params']["values"]))) / 1000 / 24

            # if "Patch.top.BCPressure.alltime.Value" not in model_config["params"]:
            #     if flux_eps_std is not None:
            #         et_per_time += np.random.normal(model_config["flux_eps"][0], model_config["flux_eps"][1]**2)

            #model.set_raining(value=model_config["precipitation_list"][time_step] + et_per_time)
            #et_per_time += np.random.normal(0, 0.0001 ** 2)
            #model._run.Patch.top.BCPressure.alltime.Value = model_config["precipitation_list"][time_step] + et_per_time

            space_indices_train = get_space_indices(model_config["grid_dz"], kalman_config["mes_locations_train"])
            space_indices_test = get_space_indices(model_config["grid_dz"], kalman_config["mes_locations_test"])

            model_params_new_values = list(np.array(list(dynamic_model_params.values()))[: 0])

            if flux_eps_std is not None:
                model_params_new_values.append(0)  # eps value in next state

            model.run(init_pressure=pressure_data, precipitation_value=model_config["precipitation_list"][time_step] + et_per_time, model_params=model_config["params"], stop_time=1)

            data_pressure = model.get_data(current_time=1/model._run.TimeStep.Value, data_name="pressure")
            data_saturation = model.get_data(current_time=1 / model._run.TimeStep.Value, data_name="saturation")

            saturation_data_loc_train = np.flip(np.squeeze(data_saturation))[space_indices_train]

            next_state_init = list(KalmanFilter.squeeze_to_last(data_pressure))
            next_state_init.extend(list(KalmanFilter.squeeze_to_last(saturation_data_loc_train)))

            if len(space_indices_test) > 0:
                saturation_data_loc_test = np.flip(KalmanFilter.squeeze_to_last(data_saturation))[space_indices_test]
                next_state_init.extend(list(KalmanFilter.squeeze_to_last(saturation_data_loc_test)))

            if len(model_params_new_values) == 0:
                model_params_new_values = model_params_data

            if len(model_params_new_values) > 0:
                next_state_init.extend(model_params_new_values)

            return np.array(next_state_init)
        return state_transition_function

    @staticmethod
    def get_nonzero_std_params(model_params):
        print("model params ", model_params)
        # Filter out items with nonzero std
        filtered_data = {key: value for key, value in model_params.items() if value[1] != 0}
        return filtered_data

    @staticmethod
    def measurement_function_wrapper(len_additional_data, model_config, kalman_config):
        def measurement_function(state_data, space_indices_type=None):
            if "mes_locations_train_slope_intercept" in kalman_config:
                slope_intercept = state_data[-len(kalman_config["mes_locations_train_slope_intercept"]):]
                state_data = state_data[:-len(kalman_config["mes_locations_train_slope_intercept"])]

            len_dynamic_params = len(KalmanFilter.get_nonzero_std_params(model_config["params"]))
            if len_dynamic_params > 0:
                additional_data = state_data[-len_additional_data:-len_dynamic_params]
            else:
                additional_data = state_data[-len_additional_data:]

            len_space_indices_train = len(get_space_indices(model_config["grid_dz"], kalman_config["mes_locations_train"]))
            len_space_indices_test = len(get_space_indices(model_config["grid_dz"], kalman_config["mes_locations_test"]))

            if space_indices_type is None:
                space_indices_type = "train"

            if space_indices_type == "train":
                measurements = additional_data[:len_space_indices_train]
                #print("train measurements ", measurements)
            elif space_indices_type == "test":
                measurements = additional_data[len_space_indices_train:len_space_indices_train+len_space_indices_test]
                print("test measurements ", measurements)
            return measurements
        return measurement_function

    @staticmethod
    def squeeze_to_last(arr):
        squeezed = np.squeeze(arr)  # Remove all singleton dimensions
        return np.reshape(squeezed, (-1,))  # Reshape to 1D

    @staticmethod
    def get_sigma_points_obj(sigma_points_params, num_state_params):
        return MerweScaledSigmaPoints(n=num_state_params, alpha=sigma_points_params["alpha"], beta=sigma_points_params["beta"], kappa=sigma_points_params["kappa"], sqrt_method=sqrt_func)

    @staticmethod
    def get_fx_hx_function(model, model_config, kalman_config, additional_data_len):
        return KalmanFilter.state_transition_function_wrapper(len_additional_data=additional_data_len, model=model, model_config=model_config, kalman_config=kalman_config),\
               KalmanFilter.measurement_function_wrapper(len_additional_data=additional_data_len, model_config=model_config, kalman_config=kalman_config),

    def set_kalman_filter(self, model, measurement_noise_covariance, state_length):
        num_state_params = state_length #last_time_step_data_pressure[1].shape[0] + self.additional_data_len# pressure + saturation + model parameters
        dim_z = len(self.kalman_config["mes_locations_train"])  # Number of measurement inputs

        sigma_points_params = self.kalman_config["sigma_points_params"]

        #sigma_points = JulierSigmaPoints(n=n, kappa=1)
        sigma_points = KalmanFilter.get_sigma_points_obj(sigma_points_params, num_state_params)
        #sigma_points = MerweScaledSigmaPoints(n=num_state_params, alpha=sigma_points_params["alpha"], beta=sigma_points_params["beta"], kappa=sigma_points_params["kappa"], sqrt_method=sqrt_func)

        # Initialize the UKF filter
        time_step = 1 # one hour time step

        fx_func, hx_func = KalmanFilter.get_fx_hx_function(model, self.model_config, self.kalman_config, self.additional_data_len)

        ukf = UnscentedKalmanFilter(dim_x=num_state_params, dim_z=dim_z, dt=time_step,
                                    fx=fx_func, #KalmanFilter.state_transition_function_wrapper(len_additional_data=self.additional_data_len, model_config=self.model_config, kalman_config=self.kalman_config),
                                    hx=hx_func, #KalmanFilter.measurement_function_wrapper(len_additional_data=self.additional_data_len, model_config=self.model_config, kalman_config=self.kalman_config),
                                    points=sigma_points)

        ukf.Q = np.ones(num_state_params)

        len_dynamic_params = len(KalmanFilter.get_nonzero_std_params(self.model_config["params"]))
        if "Q_model_params_var" in self.kalman_config and len_dynamic_params > 0:
            ukf.Q[:-len_dynamic_params] = ukf.Q[:-len_dynamic_params] * self.kalman_config["Q_var"]
            ukf.Q[-len_dynamic_params:] = ukf.Q[-len_dynamic_params:] * self.kalman_config["Q_model_params_var"]
        else:
            ukf.Q = ukf.Q * self.kalman_config["Q_var"] #5e-5 # 5e-8

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

        space_indices_train = get_space_indices(self.model_config["grid_dz"], self.kalman_config["mes_locations_train"])
        space_indices_test = get_space_indices(self.model_config["grid_dz"], self.kalman_config["mes_locations_test"])

        print("space indices train ", space_indices_train)

        #initial_state_data = np.array([np.squeeze(data.pressure), np.squeeze(data.saturation[space_indices])]).flatten()#np.squeeze(data.pressure)
        #initial_state_data = list(np.squeeze(data.pressure))

        data_pressure = model.get_data(current_time=0, data_name="pressure")
        data_saturation = model.get_data(current_time=0, data_name="saturation")

        initial_state_data = list(add_noise(np.squeeze(data_pressure), noise_level=self.kalman_config["pressure_saturation_data_noise_level"], distr_type=self.kalman_config["noise_distr_type"]))

        saturation_data = []

        #print("data.saturation ", data.saturation)
        #print("data.saturation[space_indices_train] ", data.saturation[space_indices_train])
        #print("KalmanFilter.squeeze_to_last(data.saturation[space_indices_train]) ", KalmanFilter.squeeze_to_last(data.saturation[space_indices_train]))
        #print("np.squeeze(data.saturation[space_indices_train]) ", np.squeeze(data.saturation[space_indices_train], axis=tuple(range(data.saturation[space_indices_train].ndim - 1))))
        #print("list(np.squeeze(data.saturation[space_indices_train])) ", list(np.squeeze(data.saturation[space_indices_train], axis=tuple(range(data.saturation[space_indices_train].ndim - 1)))))

        saturation_data.extend(list(KalmanFilter.squeeze_to_last(data_saturation[space_indices_train])))
        if len(space_indices_test) > 0:
            #print("tuple(range(data.saturation[space_indices_test].ndim - 1))) ", tuple(range(data.saturation[space_indices_test].ndim - 1)))
            saturation_data.extend(list(KalmanFilter.squeeze_to_last(data_saturation[space_indices_test])))

        saturation_data = list(
            add_noise(np.array(saturation_data), noise_level=self.kalman_config["pressure_saturation_data_noise_level"], distr_type=self.kalman_config["noise_distr_type"]))

        initial_state_data.extend(saturation_data)

        model_dynamic_params_mean_std = np.array(list(KalmanFilter.get_nonzero_std_params(self.model_config["params"]).values()))

        noisy_model_params_values = []
        params_std = []
        if len(model_dynamic_params_mean_std) > 0:
            noisy_model_params_values = add_noise(model_dynamic_params_mean_std[:, 0], distr_type=self.kalman_config["noise_distr_type"],
                                                  noise_level=self.kalman_config["model_params_noise_level"], std=list(model_dynamic_params_mean_std[:, 1]))
            params_std = model_dynamic_params_mean_std[:, 1]
        print("noisy model params values ", noisy_model_params_values)
        initial_state_std = np.ones(num_state_params)

        if "pressure_saturation_data_std" in self.kalman_config:
            initial_state_std[:len(initial_state_data)] = self.kalman_config["pressure_saturation_data_std"]
        else:
            np.abs(np.array(initial_state_data) * self.kalman_config["measurements_noise_level"])

        initial_state_data.extend(noisy_model_params_values)

        if "flux_eps" in self.model_config:
            params_std.append(0)

        var_models_params_std = np.array(params_std)
        if len(var_models_params_std) > 0:
            initial_state_std[-len(params_std):] = var_models_params_std

        initial_state_covariance = np.zeros((num_state_params, num_state_params))
        np.fill_diagonal(initial_state_covariance, np.array(initial_state_std)**2)

        # print("initial state std ", initial_state_std)
        # print("initial state std ", initial_state_std ** 2)

        #initial_state_covariance[-1] = 1e
        #ukf.Q[-1] = 5e-8

        # print("initial state data ", initial_state_data)
        # print("initial state covariance ", initial_state_covariance.shape)
        # print("initital state std ", initial_state_std)

        if "flux_eps" in self.model_config:
            initial_state_data.append(self.model_config["flux_eps"][1])  # eps std to state

        initial_state_data = np.array(initial_state_data)

        #print("initial state data ", initial_state_data.shape)


        ukf.x = initial_state_data #initial_state_data #(state.data[int(0.3/0.05)], state.data[int(0.6/0.05)])#state  # Initial state vector
        ukf.P = initial_state_covariance  # Initial state covariance matrix

        # print("initital state data ", initial_state_data)
        # print("initial state covariance ", initial_state_covariance)


        return ukf

    def run_kalman_filter(self, ukf, noisy_measurements, space_indices_train, space_indices_test):
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

            pred_state_iter.append(ukf.x)
            ukf_p_var_iter.append(np.diag(ukf.P))

            if len(model_dynamic_params) > 0:
                model_params_data = ukf.x[-len(model_dynamic_params):]
                pred_model_params.append(model_params_data)

            est_loc_measurements = KalmanFilter.measurement_function_wrapper(len_additional_data=self.additional_data_len,
                                                                             model_config=self.model_config,
                                                                             kalman_config=self.kalman_config)(ukf.x, space_indices_type="train")
            print("noisy measurement ", measurement)
            print("est loc measurements ", est_loc_measurements)

            test_est_loc_measurements =KalmanFilter.measurement_function_wrapper(len_additional_data=self.additional_data_len,
                                                                             model_config=self.model_config,
                                                                             kalman_config=self.kalman_config)(ukf.x, space_indices_type="test")

            pred_loc_measurements.append(est_loc_measurements)
            test_pred_loc_measurements.append(test_est_loc_measurements)

        pred_loc_measurements = np.array(pred_loc_measurements)
        test_pred_loc_measurements = np.array(test_pred_loc_measurements)
        ukf_p_var_iter = np.array(ukf_p_var_iter)

        ukf_last_p = ukf.P

        return pred_loc_measurements, test_pred_loc_measurements, pred_model_params, pred_state_iter, ukf_p_var_iter, ukf_last_p

    @staticmethod
    def plot_model_params(pred_model_params, times, variances, model_config):

        model_dynamic_params = KalmanFilter.get_nonzero_std_params(model_config["params"])
        print("model dynamic params ", model_dynamic_params)

        for idx, (param_name, mean_value_std) in enumerate(model_dynamic_params.items()):
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            print("pred_model_params[:, {}]shape ".format(idx), pred_model_params[:, idx].shape)
            axes.hlines(y=mean_value_std[0], xmin=0, xmax=pred_model_params.shape[0], linewidth=2, color='r')
            #axes.scatter(times, pred_model_params[:, idx], marker="o", label="predictions")
            print("variances[:, idx] ", variances[:, idx])
            axes.errorbar(times, pred_model_params[:, idx], yerr=np.sqrt(variances[:, idx]), fmt='o', capsize=5)# label='Data with variance')

            #axes.set_xlabel("param_name")
            axes.set_ylabel(param_name)
            fig.legend()
            fig.savefig("model_param_{}.pdf".format(param_name))
            plt.show()

    @staticmethod
    def plot_heatmap(cov_matrix):
        # Generate a heatmap using seaborn
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))

        #print("cov matrix ", cov_matrix)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        #print("eigenvalues ", eigenvalues)

        if np.any(eigenvalues < 0):
            print("Warning: Covariance matrix is not positive semi-definite!")

        #print("np.diag(cov_matrix) ", np.diag(cov_matrix))

        # Step 1: Get the standard deviations from the diagonal of the covariance matrix
        std_devs = np.sqrt(np.diag(cov_matrix))

        diag_matrix = np.diag(std_devs)

        #np.linalg.inv(diagonal_matrix) @ cov_matrix @ np.linalg.inv(diagonal_matrix)

        #print("std devs ", std_devs)

        # Step 2: Create a correlation matrix by normalizing the covariance matrix
        #correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)

        correlation_matrix = np.linalg.inv(diag_matrix) @ cov_matrix @ np.linalg.inv(diag_matrix)

        #print("np.outer(std_devs, std_devs) ", np.outer(std_devs, std_devs))

        #off_diagonal_elements = cov_matrix[np.triu_indices_from(cov_matrix, k=1)]

        # Find the maximum off-diagonal value
        #max_off_diagonal = np.max(np.abs(off_diagonal_elements))

        np.fill_diagonal(correlation_matrix, 0)

        # Get the indices where values are greater than 1
        indices = np.argwhere(correlation_matrix > 1)

        print("\nIndices where values are greater than 1:")
        for idx in indices:
            print(f"Row {idx[0]}, Column {idx[1]}: Value = {correlation_matrix[idx[0], idx[1]]}")

        correlation_matrix = np.clip(correlation_matrix, -1, 1)

        # sns.heatmap(cov_matrix, cbar=True, cmap='coolwarm', annot=False, ax=axes)
        #
        # # Add title and labels
        # axes.set_title('cov_matrix Matrix Heatmap')
        # axes.set_xlabel('Variables')
        # axes.set_ylabel('Variables')
        #
        # fig.savefig("heatmap.pdf")
        # plt.show()
        #
        # print("correlation matrix ", correlation_matrix)

        sns.heatmap(correlation_matrix, cbar=True, cmap='coolwarm', annot=False, ax=axes)

        # Add title and labels
        axes.set_title('Correlation Matrix Heatmap')
        axes.set_xlabel('Variables')
        axes.set_ylabel('Variables')

        fig.savefig("heatmap.pdf")
        plt.show()

        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
        #
        # sns.clustermap(cov_matrix, cmap='coolwarm')
        #
        # # Add title and labels
        # axes.set_title('Covariance Matrix Heatmap')
        # axes.set_xlabel('Variables')
        # axes.set_ylabel('Variables')
        #
        # fig.savefig("clustermap.pdf")
        # plt.show()
    @staticmethod
    def plot_measurements(times, measurements, noisy_measurements, pred_loc_measurements, pred_loc_measurements_variances, measurements_data_name, title_prefix):
        n_measurements = noisy_measurements.shape[1]

        import matplotlib
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        matplotlib.rcParams.update({'font.size': 26})

        print("n measurements ", n_measurements)

        for i in range(n_measurements):
            print("np.sqrt(pred_loc_measurements_variances[:, i]) ", np.sqrt(pred_loc_measurements_variances[:, i]))
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            #fig, axes = plt.subplots(1, 1)
            #axes.scatter(times, pred_loc_measurements[:, i], marker="o", label="predictions")
            axes.errorbar(times, pred_loc_measurements[:, i], ms=5, yerr=np.sqrt(pred_loc_measurements_variances[:, i]), fmt='o', capsize=5,
                          label='UKF predictions')
            if len(measurements) > 0:
                axes.scatter(times, measurements[:, i], s=15, marker='x', label="measurements")

            axes.scatter(times, noisy_measurements[:, i], s=15, marker='x', label="noisy measurements")
            axes.set_xlabel("time[h]")
            axes.set_ylabel(measurements_data_name)
            fig.legend()
            fig.savefig(title_prefix + "time_{}_loc_{}.pdf".format(measurements_data_name, i))
            plt.show()

    def plot_results(self, pred_loc_measurements, test_pred_loc_measurements, measurements_to_test, noisy_measurements_to_test,  pred_model_params, measurements, noisy_measurements, measurements_data_name="pressure", ukf_p_var_iter=None):
        print("state_loc_measurements ", pred_loc_measurements)
        print("noisy_measurements ", noisy_measurements)

        print("ukf_p_var_iter.shape ", ukf_p_var_iter.shape)
        print("pred model params shape ", np.array(pred_model_params).shape)

        model_params_variances = None
        pred_loc_measurements_variances = []
        test_pred_loc_measurements_variances = []
        if ukf_p_var_iter is not None:
            if len(pred_model_params)> 0:
                model_params_variances = ukf_p_var_iter[:, -len(pred_model_params[0]):]
                print("model params variances ", model_params_variances)

            pred_loc_measurements_variances = ukf_p_var_iter[:, -self.additional_data_len: -self.additional_data_len + len(
                self.kalman_config["mes_locations_train"])]

            if self.additional_data_len == len(self.kalman_config["mes_locations_train"]) + len(
                    self.kalman_config["mes_locations_test"]):
                test_pred_loc_measurements_variances = ukf_p_var_iter[:,
                                    -self.additional_data_len + len(self.kalman_config["mes_locations_train"]):]
            else:
                test_pred_loc_measurements_variances = ukf_p_var_iter[:,
                                    -self.additional_data_len + len(self.kalman_config["mes_locations_train"]):
                                    -self.additional_data_len + len(self.kalman_config["mes_locations_train"])
                                    + len(self.kalman_config["mes_locations_test"])]

            print("pred_loc_measurements_variances shape ", pred_loc_measurements_variances.shape)
            print("test_pred_loc_measurements_variances shape ", test_pred_loc_measurements_variances.shape)


        times = np.arange(1, pred_loc_measurements.shape[0] + 1, 1)

        print("times.shape ", times.shape)
        print("xs[:, 0].shape ", pred_loc_measurements[:, 0].shape)

        np.save(os.path.join(self.work_dir, "times"), times)
        np.save(os.path.join(self.work_dir, "model_params_variances"), model_params_variances)
        np.save(os.path.join(self.work_dir, "pred_loc_measurements_variances"), pred_loc_measurements_variances)
        np.save(os.path.join(self.work_dir, "test_pred_loc_measurements_variances"), test_pred_loc_measurements_variances)


        # plt.scatter(times, pred_loc_measurements[:, 0], marker="o", label="predictions")
        # plt.scatter(times, measurements[:, 0], marker='x',  label="measurements")
        # plt.scatter(times, noisy_measurements[:, 0], marker='x',  label="noisy measurements")
        # plt.legend()
        # plt.show()

        #######
        # Plot model params data
        ######
        print("pred model params ", pred_model_params)
        print("pred_model_params shape ", np.array(pred_model_params).shape)

        if len(pred_model_params) > 0 and self.kalman_config["plot"]:
            KalmanFilter.plot_model_params(np.array(pred_model_params), times, model_params_variances, self.model_config)

        if self.kalman_config["plot"]:
            KalmanFilter.plot_measurements(times, measurements, noisy_measurements, pred_loc_measurements,
                                   pred_loc_measurements_variances, measurements_data_name,  title_prefix="train_")
            KalmanFilter.plot_measurements(times, measurements_to_test, noisy_measurements_to_test, test_pred_loc_measurements,
                                   test_pred_loc_measurements_variances, measurements_data_name, title_prefix="test_")

        if len(measurements) > 0:
            for i in range(measurements.shape[1]):
                print("MSE predictions vs measurements loc {}: {} ".format(i, np.mean((measurements[:, 0] - pred_loc_measurements[:, 0])**2)))
                print("MSE noisy measurements vs measurements loc {}: {}".format(i, np.mean((measurements[:, 0] - noisy_measurements[:, 0])**2)))

        #print("MSE predictions vs measurements ", np.mean((measurements - pred_loc_measurements)**2))
        #print("MSE noisy measurements vs measurements ", np.mean((measurements - noisy_measurements)**2))

        # print("Var noisy measurements ", np.var(pred_loc_measurements[:, 1]))
        # print("Var predictions ", np.var(noisy_measurements[:, 1]))

        if len(measurements_to_test) > 0:
            for i in range(measurements_to_test.shape[1]):
                print("TEST MSE predictions vs measurements loc loc {}: {} ".format(i, np.mean((measurements_to_test[:, 0] - test_pred_loc_measurements[:, 0])**2)))
                print("TEST MSE noisy measurements vs measurements loc {}: {} ".format(i, np.mean((measurements_to_test[:, 0] - noisy_measurements_to_test[:, 0])**2)))

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

    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return config_dict

    # Serialize only necessary attributes
    @staticmethod
    def serialize_kalman_filter(kf, filename):
        # Collect serializable attributes
        data = {
            'K': kf.K,
            'P': kf.P,
            'R': kf.R,
            'Q': kf.Q,
            'dim_x': kf._dim_x,
            'dim_z': kf._dim_z,
            'dt': kf._dt,
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def deserialize_kalman_filter(filename, model_config, kalman_config, auxiliary_data, num_state_params):
        # Load data from the file
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        fx_func, hx_func = KalmanFilter.get_fx_hx_function(model_config, kalman_config, auxiliary_data["additional_data_len"])

        sigma_points = KalmanFilter.get_sigma_points_obj(kalman_config["sigma_points_params"], num_state_params)

        # Instantiate a new Kalman Filter object
        kf = UnscentedKalmanFilter(dim_x=data["dim_x"], dim_z=data["dim_z"], dt=data["dt"], fx=fx_func, hx=hx_func, points=sigma_points)

        # Assign the attributes from the serialized data
        for attr, value in data.items():
            setattr(kf, attr, value)
        return kf

    @staticmethod
    def plot_pressure_from_state_data(model, state_data_iter, additional_data_len):
        state_data_iter = np.array(state_data_iter)
        pressure = state_data_iter[:, :-additional_data_len]

        print("PRESSURE ", pressure.shape)

        model.plot_pressure(pressure)



        # settings.set_working_directory(cwd)


if __name__ == "__main__":
    import cProfile
    import pstats

    # pr = cProfile.Profile()
    # pr.enable()

    # Configure ParFlow executable paths if needed
    #os.environ['PARFLOW_HOME'] = '/opt/parflow_install'
    #os.environ['PATH'] += ':/opt/parflow_install/bin'
    kalman_filter = KalmanFilter()
    kalman_filter.run()

    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('cumtime')
    # ps.print_stats(50)



