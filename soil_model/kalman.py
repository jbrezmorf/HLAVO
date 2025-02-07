import sys
from pathlib import Path
import yaml
import argparse
import numpy as np
#from joblib import Memory
#memory = Memory(location='cache_dir', verbose=10)

from kalman_result import KalmanResults
from parflow_model import ToyProblem
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints
#from soil_model.evapotranspiration_fce import ET0
from auxiliary_functions import sqrt_func, add_noise
from data.load_data import load_data
from kalman_state import StateStructure, MeasurementsStructure

######
# Unscented Kalman Filter for Parflow model
# see __main__ for details
######


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
        nodes_z = self.model.get_nodes_z()

        self.state_struc = StateStructure(len(nodes_z), self.kalman_config["state_params"])
        self.train_measurements_struc = MeasurementsStructure(nodes_z, self.kalman_config["train_measurements"])
        self.test_measurements_struc = MeasurementsStructure(nodes_z, self.kalman_config["test_measurements"])
        self.state_measurements = {}

        precipitation_list = []
        for (hours, precipitation) in self.model_config['rain_periods']:
            precipitation_list.extend([precipitation] * hours)
        self.model_config["precipitation_list"] = precipitation_list

        self.results = KalmanResults(workdir, nodes_z, self.state_struc, self.train_measurements_struc,
                                     self.test_measurements_struc, config['postprocess'])
        pass

    def _make_model(self):
        if self.model_config["model_class_name"] == "ToyProblem":
            model_class = ToyProblem
        else:
            raise NotImplemented("Import desired class")
        return model_class(self.model_config, workdir=self.work_dir / "output-toy")

    # def plot_pressure(self):
    #     """
    #     MS TODO: Not used can remove? See KalmanResults.plot_pressure
    #     """
    #     model = self._make_model()
    #     et_per_time = 0 #ET0(**dict(zip(self.model_config['evapotranspiration_params']["names"], self.model_config['evapotranspiration_params']["values"]))) / 1000 / 24  # mm/day to m/sec
    #     #model._run.Patch.top.BCPressure.alltime.Value = self.model_config["precipitation_list"][0] + et_per_time
    #     model.set_dynamic_params(self.model_config["params"]["names"], self.model_config["params"]["values"])
    #
    #     model.run(init_pressure=None, precipitation_value=self.model_config["precipitation_list"][0] + et_per_time,
    #               stop_time=self.model_config['rain_periods'][0][0])
    #
    #     # model.save_pressure("pressure.png")
    #     #model.save_pressure("pressure.png")

    def run(self):
        #############################
        ### Generate measurements ###
        #############################
        if "measurements_dir" in self.kalman_config:
            noisy_measurements, noisy_measurements_to_test = load_data(data_dir=self.kalman_config["measurements_dir"], n_samples=len(self.model_config["precipitation_list"]))

            # Why to call model for real data?
            # MS TODO: why this model.run ?
            self.model.run(init_pressure=None, stop_time=1)

            sample_variance = np.var(noisy_measurements, axis=0)
            measurement_noise_covariance = np.diag(sample_variance)
        else:
            measurements, noisy_measurements, measurements_to_test, noisy_measurements_to_test, \
            state_data_iters = self.generate_measurements()

            residuals = noisy_measurements - measurements
            measurement_noise_covariance = np.cov(residuals, rowvar=False)

        # if "flux_eps" in self.model_config:
        #     self.additional_data_len += 1

        self.results.ref_states = np.array(state_data_iters)
        self.results.train_measuremnts_exact = measurements
        self.results.test_measuremnts_exact = measurements_to_test

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
        new_pressure = self.model.get_data(current_time=kalman_step+1, data_name="pressure")
        return new_pressure

    def model_iteration(self, kalman_step, pressure, params):
        # et_per_time = ET0(n=14, T=20, u2=10, Tmax=27, Tmin=12, RHmax=0.55, RHmin=0.35,
        #                   month=6) / 1000 / 24  # mm/day to m/sec
        et_per_time = 0 #ET0(**dict(zip(self.model_config['evapotranspiration_params']["names"], self.model_config['evapotranspiration_params']["values"]))) / 1000 / 24  # mm/day to m/hour
        new_pressure = self.model_run(kalman_step, pressure, params)
        new_saturation = self.model.get_data(current_time=kalman_step, data_name="saturation")
        measurements_train = self.get_measurement(kalman_step + 1, self.train_measurements_struc)
        measurements_test = self.get_measurement(kalman_step + 1, self.test_measurements_struc)

        return measurements_train, measurements_test, new_pressure, new_saturation

    def generate_measurements(self):
        train_measurements = []
        test_measurements = []
        noisy_train_measurements = []
        noisy_test_measurements = []
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

        for i in range(0, len(self.model_config["precipitation_list"])):
            measurement_train, measurement_test, pressure_vec, sat_vec \
                = self.model_iteration(i, pressure_vec, ref_params)
            self.results.ref_saturation.append(sat_vec)

            train_measurements.append(self.train_measurements_struc.encode(measurement_train))
            test_measurements.append(self.test_measurements_struc.encode(measurement_test))

            noisy_train_measurements.append(self.train_measurements_struc.encode(measurement_train, noisy=True))
            noisy_test_measurements.append(self.test_measurements_struc.encode(measurement_test, noisy=True))

            if self.verbose:
                print("i: {}, data_pressure: {} ".format(i, pressure_vec))
            ref_params['pressure_field'] = pressure_vec

            iter_state = self.state_struc.encode_state(ref_params)
            state_data_iters.append(iter_state)

        train_measurements = np.array(train_measurements)
        test_measurements = np.array(test_measurements)
        noisy_train_measurements = np.array(noisy_train_measurements)
        noisy_test_measurements = np.array(noisy_test_measurements)

        return train_measurements, noisy_train_measurements, test_measurements, noisy_test_measurements, state_data_iters

    def get_measurement(self, kalman_step, measurements_struct):
        measurements_dict = {}
        for measurement_name, measure_obj in measurements_struct.items():
            data_to_measure = self.model.get_data(current_time=kalman_step, data_name=measurement_name)
            measurements_dict[measurement_name] = measure_obj.interp @ data_to_measure
        return measurements_dict

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
        flux_eps_std = None

        et_per_time = 0 #ET0(**dict(zip(model_config['evapotranspiration_params']["names"],
                   #model_config['evapotranspiration_params']["values"]))) / 1000 / 24

        percipitation = self.model_config["precipitation_list"][time_step]
        self.model.run(pressure_data, percipitation, state, start_time=time_step, stop_time=time_step+1)
        state["pressure_field"] = self.model.get_data(current_time=time_step+1, data_name="pressure")

        measurements_train = self.get_measurement(time_step + 1, self.train_measurements_struc)
        measurements_test = self.get_measurement(time_step + 1, self.test_measurements_struc)

        new_state_vec = self.state_struc.encode_state(state)
        self.state_measurements[tuple(new_state_vec)] = (measurements_train, measurements_test)
        return new_state_vec

    # @staticmethod
    # def get_nonzero_std_params(model_params):
    #     print("model params ", model_params)
    #     # Filter out items with nonzero std
    #     filtered_data = {key: value for key, value in model_params.items() if value[1] != 0}
    #     return filtered_data

    def measurement_function(self, state_vec, measurements_type="train"):
        measurements_train_dict, measurements_test_dict = self.state_measurements[tuple(state_vec)]

        if measurements_type == "train":
            return self.train_measurements_struc.encode(measurements_train_dict)
        elif measurements_type == "test":
            return self.test_measurements_struc.encode(measurements_test_dict)

    @staticmethod
    def get_sigma_points_obj(sigma_points_params, num_state_params):
        return MerweScaledSigmaPoints(n=num_state_params, sqrt_method=sqrt_func, **sigma_points_params, )

    def set_kalman_filter(self,  measurement_noise_covariance):
        num_state_params = self.state_struc.size()
        dim_z = measurement_noise_covariance.shape[0]   # Number of measurement inputs

        sigma_points_params = self.kalman_config["sigma_points_params"]

        #sigma_points = JulierSigmaPoints(n=n, kappa=1)
        sigma_points = KalmanFilter.get_sigma_points_obj(sigma_points_params, num_state_params)
        #sigma_points = MerweScaledSigmaPoints(n=num_state_params, alpha=sigma_points_params["alpha"], beta=sigma_points_params["beta"], kappa=sigma_points_params["kappa"], sqrt_method=sqrt_func)

        # Initialize the UKF filter
        time_step = 1 # one hour time step
        ukf = UnscentedKalmanFilter(dim_x=num_state_params, dim_z=dim_z, dt=time_step,
                                    fx=self.state_transition_function, #KalmanFilter.state_transition_function_wrapper(len_additional_data=self.additional_data_len, model_config=self.model_config, kalman_config=self.kalman_config),
                                    hx=self.measurement_function, #KalmanFilter.measurement_function_wrapper(len_additional_data=self.additional_data_len, model_config=self.model_config, kalman_config=self.kalman_config),
                                    points=sigma_points)


        Q_state = self.state_struc.compose_Q()
        ukf.Q = Q_state
        print("ukf.Q.shape ", ukf.Q.shape)
        print("ukf.Q ", ukf.Q)

        ukf.R = measurement_noise_covariance #* 1e6
        print("R measurement_noise_covariance ", measurement_noise_covariance)

        data_pressure = self.model.get_data(current_time=0, data_name="pressure")

        nodes_z = self.model.get_nodes_z()
        init_mean, init_cov = self.state_struc.compose_init_state(nodes_z)
        init_state = self.state_struc.decode_state(init_mean)
        init_state["pressure_field"] = add_noise(np.squeeze(data_pressure),
                                            noise_level=self.kalman_config["pressure_saturation_data_noise_level"],
                                            distr_type=self.kalman_config["noise_distr_type"])

        # JB TODO: use init_mean, implement random choice of ref using init distr
        ukf.x = self.state_struc.encode_state(init_state) #initial_state_data #(state.data[int(0.3/0.05)], state.data[int(0.6/0.05)])#state  # Initial state vector
        ukf.P = init_cov  # Initial state covariance matrix
        return ukf

    def run_kalman_filter(self, ukf, noisy_measurements):
        pred_loc_measurements = []
        test_pred_loc_measurements = []
        pred_model_params = []
        pred_state_iter = []
        ukf_p_var_iter = []

        #model_dynamic_params = KalmanFilter.get_nonzero_std_params(self.model_config["params"])
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
            self.results.measurement_in.append(measurement)

            measurements_train_dict, measurements_test_dict = self.state_measurements[list(self.state_measurements.keys())[-1]]
            self.results.ukf_train_meas.append(self.train_measurements_struc.encode(measurements_train_dict))
            self.results.ukf_test_meas.append(self.test_measurements_struc.encode(measurements_test_dict))

        return self.results


    #
    # def postprocess_data(self, state_data_iters, pred_state_data_iter):
    #     iter_mse_pressure_data = []
    #     iter_mse_train_measurements = []
    #     iter_mse_test_measurements = []
    #
    #     iter_mse_model_config_data = {}
    #
    #     print("len additional data ", self.additional_data_len)
    #
    #     for state_data, pred_state_data  in zip(state_data_iters, pred_state_data_iter):
    #         print("len state data ", len(state_data))
    #         print("len pred state data ", len(pred_state_data))
    #         len(self.kalman_config["mes_locations_train"]) + len(self.kalman_config["mes_locations_test"]) + len(
    #             self.model_config["params"]["names"])
    #
    #         if "flux_eps" in self.model_config:
    #             pred_state_data = pred_state_data[:-1]
    #
    #         pressure_data = state_data[:-self.additional_data_len]
    #         pred_pressure_data = pred_state_data[:-self.additional_data_len]
    #         print("len pressure data ", len(pressure_data))
    #
    #         print("pressure data ", pressure_data)
    #         print("pred pressure data ", pred_pressure_data)
    #
    #         print("len pressure data ", len(pressure_data))
    #         print("len pred pressure data ", len(pred_pressure_data))
    #
    #         iter_mse_pressure_data.append(np.linalg.norm(pressure_data - pred_pressure_data))
    #
    #         train_measurements = state_data[-self.additional_data_len: -self.additional_data_len + len(self.kalman_config["mes_locations_train"])]
    #         pred_train_measurements = pred_state_data[-self.additional_data_len: -self.additional_data_len + len(
    #             self.kalman_config["mes_locations_train"])]
    #
    #         iter_mse_train_measurements.append(np.linalg.norm(train_measurements - pred_train_measurements))
    #
    #         if self.additional_data_len == len(self.kalman_config["mes_locations_train"]) + len(self.kalman_config["mes_locations_test"]):
    #             test_measurements = state_data[-self.additional_data_len + len(self.kalman_config["mes_locations_train"]):]
    #             pred_test_measurements = pred_state_data[ -self.additional_data_len + len(self.kalman_config["mes_locations_train"]):]
    #         else:
    #             test_measurements = state_data[
    #                                 -self.additional_data_len + len(self.kalman_config["mes_locations_train"]):
    #                                 -self.additional_data_len + len(self.kalman_config["mes_locations_train"])
    #                                 + len(self.kalman_config["mes_locations_test"])]
    #
    #             pred_test_measurements = pred_state_data[
    #                                      -self.additional_data_len + len(self.kalman_config["mes_locations_train"]):
    #                                      -self.additional_data_len + len(self.kalman_config["mes_locations_train"])
    #                                      + len(self.kalman_config["mes_locations_test"])]
    #
    #         iter_mse_test_measurements.append(np.linalg.norm(test_measurements - pred_test_measurements))
    #
    #         if len(self.model_config["params"]["names"]) > 0:
    #             for idx, param_name in enumerate(self.model_config["params"]["names"]):
    #                 l2_norm = np.linalg.norm(state_data[-len(self.model_config["params"]["names"]) + idx] - pred_state_data[-len(self.model_config["params"]["names"]) + idx])
    #
    #                 iter_mse_model_config_data.setdefault(param_name, []).append(l2_norm)
    #
    #
    #     print("iter_mse_pressure_data ", iter_mse_pressure_data)
    #     print("iter_mse_train_measurements ", iter_mse_train_measurements)
    #     print("iter_mse_test_measurements ", iter_mse_test_measurements)
    #     print("iter_mse_model_config_data ", iter_mse_model_config_data)
    #



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

