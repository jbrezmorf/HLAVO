import os
import sys
import yaml
import argparse
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from parflow.tools import settings
from soil_model.parflow_model import ToyProblem
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints
from soil_model.evapotranspiration_fce import ET0
from soil_model.auxiliary_functions import sqrt_func, set_nested_attr, get_nested_attr, add_noise


######
# Unscented Kalman Filter for Parflow model
# see __main__ for details
######


def get_space_indices(grid_dz, mes_locations):
    #if type == "train":
    return [int(mes_loc / grid_dz) for mes_loc in mes_locations]
    #elif type == "test":
    #    return [int(mes_loc / model._run.ComputationalGrid.DZ) for mes_loc in mes_locations_to_test]


class KalmanFilter:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('config_file', help='Path to configuration file')
        args = parser.parse_args(sys.argv[1:])
        config_file_path = os.path.abspath(args.config_file)

        config = KalmanFilter.load_config(config_file_path)

        self.model_config = config["model_config"]
        self.kalman_config = config["kalman_config"]

        if len(self.model_config["params"]['names']) != len(self.model_config["params"]['values']):
             raise ValueError("The number of names and values do not match!")
        if len(self.model_config["params"]['names']) != len(self.model_config["params"]['std']):
            raise ValueError("The number of names and stds do not match!")

        np.random.seed(config["seed"])

    def run(self):

        #############################
        ### Generate measurements ###
        #############################
        model, data, measurements, noisy_measurements, measurements_to_test, noisy_measurements_to_test, state_data_iters = self.generate_measurements(
            self.kalman_config["measurements_data_name"])
        residuals = noisy_measurements - measurements
        measurement_noise_covariance = np.cov(residuals, rowvar=False)
        print("measurement noise covariance ", measurement_noise_covariance)
        # print("10 percent cov ", np.cov(noisy_measurements*0.1, rowvar=False))
        # exit()
        # pde.plot_kymograph(storage)

        self.model_config["grid_dz"] = model._run.ComputationalGrid.DZ

        self.additional_data_len = len(self.kalman_config["mes_locations_train"]) + len(self.kalman_config["mes_locations_test"]) + len(self.model_config["params"]["names"])

        print("state data iters ", state_data_iters)

        space_indices_train = [int(mes_loc / model._run.ComputationalGrid.DZ) for mes_loc in self.kalman_config["mes_locations_train"]]
        space_indices_test = [int(mes_loc / model._run.ComputationalGrid.DZ) for mes_loc in self.kalman_config["mes_locations_test"]]

        #######################################
        ### Unscented Kalman filter setting ###
        ### - Sigma points
        ### - initital state covariance
        ### - UKF metrices
        ########################################
        ukf = self.set_kalman_filter(data, measurement_noise_covariance)

        #######################################
        ### Kalman filter run ###
        ### For each measurement (time step) ukf.update() and ukf.predict() are called
        ########################################
        pred_loc_measurements, test_pred_loc_measurements, pred_model_params, pred_state_data_iter = self.run_kalman_filter(ukf,
                                                                                                 noisy_measurements,
                                                                                                 space_indices_train,
                                                                                                 space_indices_test)

        ##############################
        ### Results postprocessing ###
        ##############################
        self.plot_results(pred_loc_measurements, test_pred_loc_measurements, measurements_to_test,
                     noisy_measurements_to_test, pred_model_params, measurements, noisy_measurements, self.kalman_config["measurements_data_name"])


        self.postprocess_data(state_data_iters, pred_state_data_iter)



    def model_iteration(self, flux_bc_pressure, data_name, data_pressure=None):
        et_per_time = ET0(n=14, T=20, u2=10, Tmax=27, Tmin=12, RHmax=0.55, RHmin=0.35,
                          month=6) / 1000 / 24  # mm/day to m/sec

        model = ToyProblem(workdir="output-toy")
        model.setup_config()
        if data_pressure is not None:
            model.set_init_pressure(init_p=data_pressure)
        else:
            model.set_init_pressure()
        model._run.TimingInfo.StopTime = 1
        #print("flux_bcpressure_per_time[0] + et_per_time ", flux_bcpressure_per_time[0] + et_per_time)
        model._run.Patch.top.BCPressure.alltime.Value = flux_bc_pressure + et_per_time
        iter_values = []
        for params, value in zip(self.model_config["params"]["names"], self.model_config["params"]["values"]):
            if params == "Patch.top.BCPressure.alltime.Value":
                continue
            set_nested_attr(model._run, params, value)
            iter_values.append(value)
        model.run()

        settings.set_working_directory(model._workdir)

        data = model._run.data_accessor
        data.time = 1 / model._run.TimeStep.Value
        data_pressure = data.pressure

        measurement = self.get_measurements(model._run.data_accessor, space_step=model._run.ComputationalGrid.DZ,
                                       mes_locations=self.kalman_config["mes_locations_train"], data_name=data_name)

        measurement_to_test = self.get_measurements(model._run.data_accessor, space_step=model._run.ComputationalGrid.DZ,
                                               mes_locations=self.kalman_config["mes_locations_test"], data_name=data_name)

        iter_state = list(np.squeeze(data_pressure))
        iter_state.extend(list(np.squeeze(measurement[0])))
        iter_state.extend(list(np.squeeze(measurement_to_test[0])))
        iter_state.extend(iter_values)

        return model, data, measurement[0], measurement_to_test[0], iter_values

    def generate_measurements(self, data_name):
        #flux_bcpressure_per_time = [-2e-2] * 24 + [0] * 16 + [-2e-2] * 32
        #flux_bcpressure_per_time = [-1.3889 * 10 ** -6] * 48 + [0] * 24 + [-1.3889 * 10 ** -6] * 36

        #flux_bcpressure_per_time = [-2e-3] * 24 + [0] * 4 + [-2e-3] * 6
        flux_bcpressure_per_time = [-2e-2] * 12

        # Calculate ET0 - see soil_model.et for details. Input parameters:
        # n = daylight hours [-]
        # T = mean daily air temperature at 2 m height [°C]
        # u2 = wind speed at 2 m height [m/s]
        # month = month number of actual day [1-12]
        # Tmax, Tmin = 10 day or monthly average of maximal/minimal daily temperatures [°C]
        # RHmax, RHmin = 10 day or monthly average of maximal/minimal relative humidity [0-1]
        et_per_time = ET0(n=14, T=20, u2=10, Tmax=27, Tmin=12, RHmax=0.55, RHmin=0.35, month=6) / 1000 / 24  # mm/day to m/sec

        print("et per time ", et_per_time)
        print("flux_bcpressure_per_time ", flux_bcpressure_per_time)

        measurements = []
        measurements_to_test = []
        state_data_iters = []

        ###################
        ##   Model runs  ##
        ###################
        # Loop through time steps
        for i in range(0, len(flux_bcpressure_per_time)):
            if i == 0:
                data_pressure = None
            model, data, measurement_train, measurement_test, iter_values = self.model_iteration(flux_bcpressure_per_time[i], data_name, data_pressure=data_pressure)
            measurements.append(measurement_train)
            measurements_to_test.append(measurement_test)

            data_pressure = data.pressure

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
        return model, data, measurements, noisy_measurements, measurements_to_test, noisy_measurements_to_test, state_data_iters


    def get_measurements(self, data, space_step, mes_locations=None, data_name="pressure"):
        data.time= 0
        measurements = np.zeros((len(np.unique(data.times)), len(mes_locations)))
        space_indices = [int(mes_loc / space_step) for mes_loc in mes_locations]

        print("space indices ", space_indices)
        # print("data.times ", data.times)

        for data_t in data.times:
            if data_name == "saturation":
                data_to_measure = data.saturation
            elif data_name == "pressure":
                data_to_measure = data.pressure
            measurements[data_t, :] = np.flip(np.squeeze(data_to_measure))[space_indices]
            data.time += 1

        #print("pressure_measurements ", pressure_measurements)
        return measurements
        #print("field ", field.data[int(mes_1_loc/space_step)])


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
    def state_transition_function_wrapper(len_additional_data, model_config, kalman_config):
        def state_transition_function(state_data, dt):
            #flux_bcpressure_per_time = [-2e-2] * 24 + [0] * 16 + [-2e-2] * 32
            et_per_time = 0

            print("state data shape ", state_data)
            pressure_data = state_data[0:-len_additional_data]  # Extract saturation from state vector
            print("pressure data shape ", pressure_data.shape)

            model_params_data = []
            if len(model_config["params"]["names"]) > 0:
                model_params_data = state_data[-len(model_config["params"]["names"]):]

            print("model_params_data ", model_params_data)

            pressure_data = pressure_data.reshape(pressure_data.shape[0], 1, 1)

            model = ToyProblem(workdir="output-toy")
            model.setup_config()
            model.set_init_pressure(init_p=pressure_data)
            model._run.TimingInfo.StopTime = 1
            #model._run.Patch.top.BCPressure.alltime.Value = flux_bcpressure_per_time[0] + et_per_time

            et_per_time = ET0(n=14, T=20, u2=10, Tmax=27, Tmin=12, RHmax=0.55, RHmin=0.35, month=6) / 1000 / 24

            space_indices_train = get_space_indices(model_config["grid_dz"], kalman_config["mes_locations_train"])
            space_indices_test = get_space_indices(model_config["grid_dz"], kalman_config["mes_locations_test"])

            if len(model_config["params"]["names"]) > 0:
                for params, value in zip(model_config["params"]["names"], model_params_data):
                    print("params: {}, value: {}".format(params, value))

                    if params == "Patch.top.BCPressure.alltime.Value":
                        value += et_per_time
                    set_nested_attr(model._run, params, value)

                    if params == "Geom.domain.Saturation.Alpha":
                        set_nested_attr(model._run, "Geom.domain.RelPerm.Alpha", value)

                    if params == "Geom.domain.Saturation.N":
                        set_nested_attr(model._run, "Geom.domain.RelPerm.N", value)


            model.run()

            #cwd = settings.get_working_directory()
            settings.set_working_directory(model._workdir)

            data = model._run.data_accessor
            #print("data.times ", data.times)

            data.time= 1/model._run.TimeStep.Value
            #@TODO: use fixed stop time instead of time step counting

            saturation_data_loc_train = np.flip(np.squeeze(data.saturation))[space_indices_train]

            #saturation_data_loc = data.saturation[space_indices]

            next_state_init = list(np.squeeze(data.pressure))
            next_state_init.extend(list(np.squeeze(saturation_data_loc_train)))

            if len(space_indices_test) > 0:
                saturation_data_loc_test = np.flip(np.squeeze(data.saturation))[space_indices_test]
                next_state_init.extend(list(np.squeeze(saturation_data_loc_test)))

            model_params_new_values = model_params_data

            if len(model_params_new_values) > 0:
                next_state_init.extend(model_params_new_values)

            next_state_init = np.array(next_state_init)

            #next_state_init = np.array([np.squeeze(data.pressure), np.squeeze(saturation_data_loc)]).flatten()
            print("next state init ",  next_state_init)
            print("next state inint shape ", next_state_init.shape)

            return next_state_init
        return state_transition_function


    @staticmethod
    def measurement_function_wrapper(len_additional_data, model_config, kalman_config):
        @staticmethod
        def measurement_function(pressure_data, space_indices_type=None):
            if len(model_config["params"]["names"]) > 0:
                additional_data = pressure_data[-len_additional_data:-len(model_config["params"]["names"])]
            else:
                additional_data = pressure_data[-len_additional_data:]

            len_space_indices_train = len(get_space_indices(model_config["grid_dz"], kalman_config["mes_locations_train"]))
            len_space_indices_test = len(get_space_indices(model_config["grid_dz"], kalman_config["mes_locations_test"]))

            if space_indices_type is None:
                space_indices_type = "train"

            if space_indices_type == "train":
                measurements = additional_data[:len_space_indices_train]
                print("train measurements ", measurements)
            elif space_indices_type == "test":
                measurements = additional_data[len_space_indices_train:len_space_indices_train+len_space_indices_test]
                print("test measurements ", measurements)
            return measurements
        return measurement_function


    def set_kalman_filter(self, data, measurement_noise_covariance):
        num_state_params = data.pressure.shape[0] + self.additional_data_len# pressure + saturation + model parameters
        dim_z = len(self.kalman_config["mes_locations_train"])  # Number of measurement inputs

        #sigma_points = JulierSigmaPoints(n=n, kappa=1)
        sigma_points = MerweScaledSigmaPoints(n=num_state_params, alpha=1e-2, beta=2.0, kappa=1, sqrt_method=sqrt_func)
        # Initialize the UKF filter
        time_step = 1 # one hour time step
        ukf = UnscentedKalmanFilter(dim_x=num_state_params, dim_z=dim_z, dt=time_step,
                                    fx=KalmanFilter.state_transition_function_wrapper(len_additional_data=self.additional_data_len, model_config=self.model_config, kalman_config=self.kalman_config),
                                    hx=KalmanFilter.measurement_function_wrapper(len_additional_data=self.additional_data_len, model_config=self.model_config, kalman_config=self.kalman_config),
                                    points=sigma_points)

        ukf.Q = np.ones(num_state_params) * 5e-8 # 5e-8
        print("ukf.Q.shape ", ukf.Q.shape)
        print("ukf.Q ", ukf.Q)
        #ukf.Q = Q_discrete_white_noise(dim=1, dt=1.0, var=1e-6, block_size=num_locations)  # Process noise covariance
        ukf.R = measurement_noise_covariance #* 1e6
        # print("R.shape ", ukf.R.shape)
        # exit()

        # best setting so far initial_covariance = np.eye(n) * 1e4, ukf.Q = np.ones(num_locations) * 1e-7
        data.time = 0

        space_indices_train = get_space_indices(self.model_config["grid_dz"], self.kalman_config["mes_locations_train"])
        space_indices_test = get_space_indices(self.model_config["grid_dz"], self.kalman_config["mes_locations_test"])

        #initial_state_data = np.array([np.squeeze(data.pressure), np.squeeze(data.saturation[space_indices])]).flatten()#np.squeeze(data.pressure)
        #initial_state_data = list(np.squeeze(data.pressure))

        initial_state_data = list(add_noise(np.squeeze(data.pressure), noise_level=self.kalman_config["pressure_saturation_data_noise_level"], distr_type=self.kalman_config["noise_distr_type"]))

        saturation_data = []

        saturation_data.extend(list(np.squeeze(data.saturation[space_indices_train])))
        if len(space_indices_test) > 0:
            saturation_data.extend(list(np.squeeze(data.saturation[space_indices_test])))

        saturation_data = list(
            add_noise(np.array(saturation_data), noise_level=self.kalman_config["pressure_saturation_data_noise_level"], distr_type=self.kalman_config["noise_distr_type"]))

        initial_state_data.extend(saturation_data)

        noisy_model_params_values = add_noise(list(self.model_config["params"]["values"]), distr_type=self.kalman_config["noise_distr_type"], std=list(self.model_config["params"]["std"]))
        print("noisy model params values ", noisy_model_params_values)

        initial_state_std = np.ones(num_state_params)

        if "pressure_saturation_data_std" in self.kalman_config:
            initial_state_std[:len(initial_state_data)] = self.kalman_config["pressure_saturation_data_std"]
        else:
            np.abs(np.array(initial_state_data) * self.kalman_config["measurements_noise_level"])

        initial_state_data.extend(noisy_model_params_values)

        var_models_params_std = np.array(self.model_config["params"]["std"])
        initial_state_std[-len(noisy_model_params_values):] = var_models_params_std


        initial_state_covariance = np.zeros((num_state_params, num_state_params))
        np.fill_diagonal(initial_state_covariance, np.array(initial_state_std)**2)

        print("initial state data ", initial_state_data)
        print("initial state covariance ", initial_state_covariance)
        print("initital state std ", initial_state_std)


        initial_state_data = np.array(initial_state_data)

        ukf.x = initial_state_data #initial_state_data #(state.data[int(0.3/0.05)], state.data[int(0.6/0.05)])#state  # Initial state vector
        ukf.P = initial_state_covariance  # Initial state covariance matrix

        return ukf


    def run_kalman_filter(self, ukf, noisy_measurements, space_indices_train, space_indices_test):
        pred_loc_measurements = []
        test_pred_loc_measurements = []
        pred_model_params = []
        pred_state_iter = []
        # Assuming you have a measurement at each time step
        print("noisy_meaesurements ", noisy_measurements)
        for measurement in noisy_measurements:
            print("measurement ", measurement)
            ukf.predict()
            ukf.update(measurement)
            print("sum ukf.P ", np.sum(ukf.P))
            print("Estimated State:", ukf.x)

            pred_state_iter.append(ukf.x)

            if len(self.model_config["params"]["names"]) > 0:
                model_params_data = ukf.x[-len(self.model_config["params"]["names"]):]
                pred_model_params.append(model_params_data)

            est_loc_measurements = KalmanFilter.measurement_function_wrapper(len_additional_data=self.additional_data_len,
                                                                             model_config=self.model_config,
                                                                             kalman_config=self.kalman_config)(ukf.x, space_indices_type="train")
            print("est loc measurements ", est_loc_measurements)

            test_est_loc_measurements =KalmanFilter.measurement_function_wrapper(len_additional_data=self.additional_data_len,
                                                                             model_config=self.model_config,
                                                                             kalman_config=self.kalman_config)(ukf.x, space_indices_type="test")

            pred_loc_measurements.append(est_loc_measurements)
            test_pred_loc_measurements.append(test_est_loc_measurements)

        pred_loc_measurements = np.array(pred_loc_measurements)
        test_pred_loc_measurements = np.array(test_pred_loc_measurements)

        return pred_loc_measurements, test_pred_loc_measurements, pred_model_params, pred_state_iter

    def plot_model_params(self, pred_model_params, times):

        for idx, (param_name, param_init_value) in enumerate(zip(self.model_config["params"]["names"], self.model_config["params"]["names"])):
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            print("pred_model_params[:, {}]shape ".format(idx), pred_model_params[:, idx].shape)
            axes.hlines(y=param_init_value, xmin=0, xmax=pred_model_params.shape[0], linewidth=2, color='r')
            axes.scatter(times, pred_model_params[:, idx], marker="o", label="predictions")
            #axes.set_xlabel("param_name")
            axes.set_ylabel(param_name)
            fig.legend()
            fig.savefig("model_param_{}.pdf".format(param_name))
            plt.show()


    def plot_results(self, pred_loc_measurements, test_pred_loc_measurements, measurements_to_test, noisy_measurements_to_test,  pred_model_params, measurements, noisy_measurements, measurements_data_name="pressure",):
        print("state_loc_measurements ", pred_loc_measurements)
        print("noisy_measurements ", noisy_measurements)

        times = np.arange(1, pred_loc_measurements.shape[0] + 1, 1)

        print("times.shape ", times.shape)
        print("xs[:, 0].shape ", pred_loc_measurements[:, 0].shape)

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

        if len(pred_model_params) > 0:
            self.plot_model_params(np.array(pred_model_params), times)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        axes.scatter(times, pred_loc_measurements[:, 0], marker="o", label="predictions")
        axes.scatter(times, measurements[:, 0], marker='x',  label="measurements")
        axes.scatter(times, noisy_measurements[:, 0], marker='x',  label="noisy measurements")
        axes.set_xlabel("time")
        axes.set_ylabel(measurements_data_name)
        fig.legend()
        fig.savefig("time_{}_loc_0.pdf".format(measurements_data_name))
        plt.show()

        print("Var noisy measurements ", np.var(pred_loc_measurements[:, 0]))
        print("Var predictions ", np.var(pred_loc_measurements[:, 0]))

        if measurements.shape[1] > 1:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            axes.scatter(times, pred_loc_measurements[:, 1], marker="o", label="predictions")
            axes.scatter(times, measurements[:, 1], marker='x', label="measurements")
            axes.scatter(times, noisy_measurements[:, 1], marker='x', label="noisy measurements")
            axes.set_xlabel("time")
            axes.set_ylabel(measurements_data_name)
            fig.legend()
            fig.savefig("time_{}_loc_1.pdf".format(measurements_data_name))
            plt.show()
            # plt.scatter(times, pred_loc_measurements[:, 1], marker="o", label="predictions")
            # plt.scatter(times, measurements[:, 1], marker='x',  label="measurements")
            # plt.scatter(times, noisy_measurements[:, 1], marker='x',  label="noisy measurements")
            # plt.legend()
            # plt.show()

        for i, pos in enumerate(self.kalman_config["mes_locations_test"]):
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            axes.scatter(times, test_pred_loc_measurements[:, i], marker="o", label="predictions")
            axes.scatter(times, measurements_to_test[:, i], marker='x', label="measurements")
            axes.scatter(times, noisy_measurements_to_test[:, i], marker='x', label="noisy measurements")
            axes.set_xlabel("time")
            axes.set_ylabel(measurements_data_name)
            fig.legend()
            fig.savefig("test_time_{}_loc_{}.pdf".format(measurements_data_name, i))
            plt.show()

            # plt.scatter(times, test_pred_loc_measurements[:, i], marker="o", label="predictions")
            # plt.scatter(times, measurements_to_test[:, i], marker='x', label="measurements")
            # plt.scatter(times, noisy_measurements_to_test[:, i], marker='x', label="noisy measurements")
            # plt.legend()
            # plt.title("Test locations, pos: {}".format(pos))
            # plt.show()

        print("MSE predictions vs measurements loc 0 ", np.mean((measurements[:, 0] - pred_loc_measurements[:, 0])**2))
        print("MSE noisy measurements vs measurements loc 0", np.mean((measurements[:, 0] - noisy_measurements[:, 0])**2))

        print("MSE predictions vs measurements loc 1 ", np.mean((measurements[:, 1] - pred_loc_measurements[:, 1])**2))
        print("MSE noisy measurements vs measurements loc 1", np.mean((measurements[:, 1] - noisy_measurements[:, 1])**2))

        #print("MSE predictions vs measurements ", np.mean((measurements - pred_loc_measurements)**2))
        #print("MSE noisy measurements vs measurements ", np.mean((measurements - noisy_measurements)**2))

        print("Var noisy measurements ", np.var(pred_loc_measurements[:, 1]))
        print("Var predictions ", np.var(noisy_measurements[:, 1]))


        print("TEST MSE predictions vs measurements loc 0 ", np.mean((measurements_to_test[:, 0] - test_pred_loc_measurements[:, 0])**2))
        print("TEST MSE noisy measurements vs measurements loc 0", np.mean((measurements_to_test[:, 0] - noisy_measurements_to_test[:, 0])**2))

        print("TEST MSE predictions vs measurements loc 1 ", np.mean((measurements_to_test[:, 1] - test_pred_loc_measurements[:, 1])**2))
        print("TEST MSE noisy measurements vs measurements loc 1", np.mean((measurements_to_test[:, 1] - noisy_measurements_to_test[:, 1])**2))

    def postprocess_data(self, state_data_iters, pred_state_data_iter):

        for state_data, pred_state_data  in zip(state_data_iters, pred_state_data_iter):
            len(self.kalman_config["mes_locations_train"]) + len(self.kalman_config["mes_locations_test"]) + len(
                self.model_config["params"]["names"])

            pressure_data = state_data[:-self.additional_data_len]
            pred_pressure_data = pred_state_data[:-self.additional_data_len]

            train_measurements = state_data[-self.additional_data_len: -self.additional_data_len + len(self.kalman_config["mes_locations_train"])]
            pred_train_measurements = pred_state_data[-self.additional_data_len: -self.additional_data_len + len(
                self.kalman_config["mes_locations_train"])]

            test_measurements = state_data[-self.additional_data_len + len(self.kalman_config["mes_locations_train"]):
                                           -self.additional_data_len + len(self.kalman_config["mes_locations_train"])
                                           + len(self.kalman_config["mes_locations_test"])]

            pred_test_measurements = pred_state_data[-self.additional_data_len + len(self.kalman_config["mes_locations_train"]):
                                                     -self.additional_data_len + len(self.kalman_config["mes_locations_train"])
                                                     + len(self.kalman_config["mes_locations_test"])]


            #@TODO: compare model params


    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict


if __name__ == "__main__":
    import cProfile
    import pstats

    pr = cProfile.Profile()
    pr.enable()


    kalman_filter = KalmanFilter()
    kalman_filter.run()

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats(50)



