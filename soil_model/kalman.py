import pde  # py-pde
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from functools import reduce
import dataclasses
from parflow.tools import settings
#from notebooks.Richards.richards_sim import mu_from_h, h_from_mu, RichardsPDE, Hydraulic
from soil_model.parflow_model import ToyProblem
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints
from soil_model.et import ET0


######
# Unscented Kalman Filter for Parflow model
# see __main__ for details
######


model_params = {"Geom.domain.RelPerm.Alpha":0.58,
                "Geom.domain.Saturation.Alpha":  0.58,
                "Geom.domain.RelPerm.N":3.7,
                "Geom.domain.Saturation.N": 3.7,
                "Geom.domain.Saturation.SRes":0.06,
                "Geom.domain.Saturation.SSat":  0.47,
                "Patch.top.BCPressure.alltime.Value":  -2e-2

                #self._run.Geom.domain.Perm.Value
}

# model_params_std = {"Geom.domain.RelPerm.Alpha":  1e-2,
#                 "Geom.domain.Saturation.Alpha": 1e-2,
#                 "Geom.domain.RelPerm.N":3e-1,
#                 "Geom.domain.Saturation.N": 3e-1,
#                 "Geom.domain.Saturation.SRes":  1e-3,
#                 "Geom.domain.Saturation.SSat": 4e-2,
#                 "Patch.top.BCPressure.alltime.Value":  1e-3
#
#                 # self._run.Geom.domain.Perm.Value
#                 }
model_params_std = {"Geom.domain.RelPerm.Alpha":  1e-3,
                "Geom.domain.Saturation.Alpha": 1e-4,
                "Geom.domain.RelPerm.N":3e-2,
                "Geom.domain.Saturation.N": 3e-3,
                "Geom.domain.Saturation.SRes":  1e-4,
                "Geom.domain.Saturation.SSat": 4e-4,
                "Patch.top.BCPressure.alltime.Value":  1e-4
                # self._run.Geom.domain.Perm.Value
                }

#model_params = {}


def model_iteration(flux_bc_pressure, data_name, data_pressure=None):
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
    for params, value in model_params.items():
        if params == "Patch.top.BCPressure.alltime.Value":
            continue
        set_nested_attr(model._run, params, value)
        iter_values.append(value)
    model.run()

    settings.set_working_directory(model._workdir)

    data = model._run.data_accessor
    data.time = 1 / model._run.TimeStep.Value
    data_pressure = data.pressure

    measurement = get_measurements(model._run.data_accessor, space_step=model._run.ComputationalGrid.DZ,
                                   mes_locations=mes_locations_to_train, data_name=data_name)

    measurement_to_test = get_measurements(model._run.data_accessor, space_step=model._run.ComputationalGrid.DZ,
                                           mes_locations=mes_locations_to_test, data_name=data_name)

    iter_state = list(np.squeeze(data.pressure))
    iter_state.extend(list(np.squeeze(measurement[0])))
    iter_state.extend(list(np.squeeze(measurement_to_test[0])))
    iter_state.extend(iter_values)

    return model, data, measurement[0], measurement_to_test[0], iter_values

def generate_measurements(data_name):
    #flux_bcpressure_per_time = [-2e-2] * 24 + [0] * 16 + [-2e-2] * 32
    #flux_bcpressure_per_time = [-1.3889 * 10 ** -6] * 48 + [0] * 24 + [-1.3889 * 10 ** -6] * 36

    #flux_bcpressure_per_time = [-2e-3] * 24 + [0] * 4 + [-2e-3] * 6
    flux_bcpressure_per_time = [-2e-2] * 72

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
    state_data = []

    #############################
    ##   Model initialization  ##
    #############################

    # model, data, measurement_train, measurement_test, iter_values = model_iteration( flux_bcpressure_per_time[0], data_name)
    # measurements.append(measurement_train)
    # measurements_to_test.append(measurement_test)
    #
    # data_pressure = data.pressure
    #
    # iter_state = list(np.squeeze(data_pressure))
    # iter_state.extend(list(np.squeeze(measurement_train)))
    # iter_state.extend(list(np.squeeze(measurement_test)))
    # iter_state.extend(iter_values)
    #
    # state_data.append(iter_state)
    #
    # Loop through time
    for i in range(0, len(flux_bcpressure_per_time)):
        if i == 0:
            data_pressure = None
        model, data, measurement_train, measurement_test, iter_values = model_iteration(flux_bcpressure_per_time[i], data_name, data_pressure=data_pressure)
        measurements.append(measurement_train)
        measurements_to_test.append(measurement_test)

        data_pressure = data.pressure

        iter_state = list(np.squeeze(data_pressure))
        iter_state.extend(list(np.squeeze(measurement_train)))
        iter_state.extend(list(np.squeeze(measurement_test)))
        iter_state.extend(iter_values)

        state_data.append(iter_state)


    noisy_measurements = add_noise_to_measurements(np.array(measurements), level=0.1)
    noisy_measurements_to_test = add_noise_to_measurements(np.array(measurements_to_test), level=0.1)

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
    return model, data, measurements, noisy_measurements, measurements_to_test, noisy_measurements_to_test, state_data


def get_measurements(data, space_step, mes_locations=None, data_name="pressure"):
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


def add_noise(data_array, noise_level=0.1, std=None, type="uniform"):
    if type == "uniform":
        noise = np.random.uniform(-noise_level * data_array, noise_level * data_array)
        data_array = data_array + noise

        print("data array ", data_array)

    elif type == "gaussian":
        pass


    return data_array



def add_noise_to_measurements(measurements, level=0.1):
    noisy_measurements = np.zeros(measurements.shape)
    for i in range(measurements.shape[1]):
        noisy_measurements[:, i] = add_noise(measurements[:, i], noise_level=level, type="uniform")
        # for idx, mes_val in enumerate(measurements[:, i]):
        #     noise = np.random.uniform(-level* mes_val, level*mes_val)
        #     noisy_measurements[idx, i] = measurements[idx, i] + noise
    return noisy_measurements


def add_model_attributes(model, state_data):
    pass


def get_nested_attr(obj, attr):
    """
    Access nested attributes of an object using a dot-separated string.
    """
    return reduce(getattr, attr.split('.'), obj)

def set_nested_attr(obj, attr, value):
    """
    Set the value of a nested attribute of an object using a dot-separated string.
    """
    pre, _, post = attr.rpartition('.')
    return setattr(get_nested_attr(obj, pre) if pre else obj, post, value)


#####################
### Kalman filter ###
#####################
def state_transition_function(state_data, dt):
    #flux_bcpressure_per_time = [-2e-2] * 24 + [0] * 16 + [-2e-2] * 32
    et_per_time = 0
    #print("state function call")
    #print("pressure data ", pressure_data)
    space_indices_train = get_space_indices(type="train")
    space_indices_test = get_space_indices(type="test")

    len_additional_data = get_len_saturation_data() + len(model_params)  # saturation data + model params

    print("state data shape ", state_data)
    pressure_data = state_data[0:-len_additional_data]  # Extract saturation from state vector
    print("pressure data shape ", pressure_data.shape)

    model_params_data = []
    if len(model_params) > 0:
        model_params_data = state_data[-len(model_params):]

    print("model_params_data ", model_params_data)
    
    pressure_data = pressure_data.reshape(pressure_data.shape[0], 1, 1)

    model = ToyProblem(workdir="output-toy")
    model.setup_config()
    model.set_init_pressure(init_p=pressure_data)
    model._run.TimingInfo.StopTime = 1
    #model._run.Patch.top.BCPressure.alltime.Value = flux_bcpressure_per_time[0] + et_per_time

    et_per_time = ET0(n=14, T=20, u2=10, Tmax=27, Tmin=12, RHmax=0.55, RHmin=0.35, month=6) / 1000 / 24


    if len(model_params) > 0:
        for params, value in zip(model_params, model_params_data):
            print("params: {}, value: {}".format(params, value))

            if params == "Patch.top.BCPressure.alltime.Value":
                value += et_per_time
            set_nested_attr(model._run, params, value)

            if params == "Geom.domain.Saturation.Alpha":
                set_nested_attr(model._run, "Geom.domain.RelPerm.Alpha", value)

            if params == "Geom.domain.Saturation.N":
                set_nested_attr(model._run, "Geom.domain.RelPerm.N", value)

    #model._run.Patch.top.BCPressure.alltime.Value = -2e-2


    # value = get_nested_attr(model._run, attributes)
    # print("value ", value)
    # exit()
    #

    ## Aditional attributes
    # model._run.Geom.domain.Perm.Value =
    #
    # model._run.Geom.domain.RelPerm.Alpha
    #
    # model._run.Geom.domain.RelPerm.N
    #
    # model._run.Geom.domain.Saturation.Alpha = 0.58
    # model._run.Geom.domain.Saturation.N = 3.7
    # model._run.Geom.domain.Saturation.SRes = 0.06
    # model._run.Geom.domain.Saturation.SSat = 0.47
    #
    # model._run.Patch.top.BCPressure.alltime.Value =

    model.run()

    #cwd = settings.get_working_directory()
    settings.set_working_directory(model._workdir)

    data = model._run.data_accessor
    #print("data.times ", data.times)

    data.time= 1/model._run.TimeStep.Value
    #@TODO: use fixed stop time instead of time step counting

    #print("data.time ", data.time)
    #print("data pressure shape ", data.pressure)

    saturation_data_loc_train = np.flip(np.squeeze(data.saturation))[space_indices_train]

    #saturation_data_loc = data.saturation[space_indices]

    next_state_init = list(np.squeeze(data.pressure))
    next_state_init.extend(list(np.squeeze(saturation_data_loc_train)))

    if len(space_indices_test) > 0:
        saturation_data_loc_test = np.flip(np.squeeze(data.saturation))[space_indices_test]
        next_state_init.extend(list(np.squeeze(saturation_data_loc_test)))

    # # Add new model values
    # model_params_new_values = []
    # for param in model_params:
    #     model_params_new_values.append(get_nested_attr(model._run, param))


    model_params_new_values = model_params_data

    if len(model_params_new_values) > 0:
        next_state_init.extend(model_params_new_values)

    next_state_init = np.array(next_state_init)

    #next_state_init = np.array([np.squeeze(data.pressure), np.squeeze(saturation_data_loc)]).flatten()
    print("next state init ",  next_state_init)
    print("next state inint shape ", next_state_init.shape)

    return next_state_init


def get_space_indices(type="train"):
    if type == "train":
        return [int(mes_loc / model._run.ComputationalGrid.DZ) for mes_loc in mes_locations_to_train]
    elif type == "test":
        return [int(mes_loc / model._run.ComputationalGrid.DZ) for mes_loc in mes_locations_to_test]


def get_len_saturation_data():
    # @TODO: return additional data length
    return len(get_space_indices(type="train")) + len(get_space_indices(type="test"))


def measurement_function(pressure_data, space_indices_type=None):
    print("measurement_function(pressure_data) ", pressure_data)
    print("measurement_function(pressure_data).shape ", pressure_data.shape)

    len_additional_data = get_len_saturation_data() + len(model_params)
    if len(model_params) > 0:
        additional_data = pressure_data[-len_additional_data:-len(model_params)]
    else:
        additional_data = pressure_data[-len_additional_data:]

    print("measurement function additional data ", additional_data)

    len_space_indices_train = len(get_space_indices(type="train"))
    len_space_indices_test = len(get_space_indices(type="test"))

    if space_indices_type is None:
        space_indices_type = "train"

    if space_indices_type == "train":
        measurements = additional_data[:len_space_indices_train]
        print("train measurements ", measurements)
    elif space_indices_type == "test":
        measurements = additional_data[len_space_indices_train:len_space_indices_train+len_space_indices_test]
        print("test measurements ", measurements)


    #pressure_data = pressure_data[0:-len(space_indices)]
    #pressure_data = np.flip(pressure_data)
    #measurements = pressure_data[space_indices]


    #print("measurement_function measurements ", measurements)
    return measurements


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        print("isPD")
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        print("while not isPD")
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = sc.linalg.cholesky(B)
        return True
    except sc.linalg.LinAlgError:
        return False


def sqrt_func(x):
    print("sqrt func call")
    try:
        result = sc.linalg.cholesky(x)
    except:# np.linalg.LinAlgError:
        #x = (x + x.T)/2
        x = nearestPD(x)
        e_val, e_vec = np.linalg.eigh(x)
        print("e_val ", e_val)
        result = sc.linalg.cholesky(x)
        print("result ", result)
    return result


def set_kalman_filter(data, measurement_noise_covariance):
    num_locations = data.pressure.shape[0] + get_len_saturation_data() + len(model_params) # pressure + saturation + model parameters
    #dt = 60*60  # Time between steps in seconds
    dim_z = len(mes_locations_to_train)  # Number of measurement inputs
    #initial_covariance = np.cov(noisy_measurements, rowvar=False)

    initial_state_covariance = np.eye(num_locations) * 1e1
    #initial_state_covariance[-1] = 1e-5
    #initial_covariance[:-len(model_params), :-len(model_params)] = 0 # 1e-5
    #sigma_points = JulierSigmaPoints(n=n, kappa=1)
    sigma_points = MerweScaledSigmaPoints(n=num_locations, alpha=1e-2, beta=2.0, kappa=1, sqrt_method=sqrt_func)
    # Initialize the UKF filter
    time_step = 1 # one hour time step
    ukf = UnscentedKalmanFilter(dim_x=num_locations, dim_z=dim_z, dt=time_step,
                                fx=state_transition_function, hx=measurement_function, points=sigma_points)
    ukf.Q = np.ones(num_locations) * 5e-8 # 5e-8
    #ukf.Q[:-len(model_params)] = 1e-5
    #ukf.Q[-2:] = 0
    print("ukf.Q.shape ", ukf.Q.shape)
    print("ukf.Q ", ukf.Q)
    #ukf.Q = Q_discrete_white_noise(dim=1, dt=1.0, var=1e-6, block_size=num_locations)  # Process noise covariance
    ukf.R = measurement_noise_covariance #* 1e6
    # print("R.shape ", ukf.R.shape)
    # exit()

    # best setting so far initial_covariance = np.eye(n) * 1e4, ukf.Q = np.ones(num_locations) * 1e-7
    data.time = 0

    space_indices_train = get_space_indices(type="train")
    space_indices_test = get_space_indices(type="test")

    #initial_state_data = np.array([np.squeeze(data.pressure), np.squeeze(data.saturation[space_indices])]).flatten()#np.squeeze(data.pressure)
    #initial_state_data = list(np.squeeze(data.pressure))

    initial_state_data = list(add_noise(np.squeeze(data.pressure), noise_level=0.1, type="uniform"))

    initial_state_data.extend(list(np.squeeze(data.saturation[space_indices_train])))

    if len(space_indices_test) > 0:
        initial_state_data.extend(list(np.squeeze(data.saturation[space_indices_test])))

    for i, (param_name, value) in enumerate(model_params.items()):
        std_val = model_params_std[param_name]
        orig_value_sign = np.sign(value)
        print("value sign ", orig_value_sign)

        #std_val = np.abs(value) * 0.1
        value_noise = np.random.normal(0, std_val)
        print("value: {}, noise: {}, value + noise: {}".format(value, value_noise, value + value_noise))

        print("value noise ", value_noise)

        value = value + value_noise

        if np.sign(value) != orig_value_sign:
            value *= -1

        initial_state_data.append(value)
        initial_state_covariance[-len(model_params) + i] = std_val ** 2
        #@TODO: set ukf.Q in the same matter

    print("initial state data ", initial_state_data)
    print("initial state covariance ", initial_state_covariance)




    # print("model._run.Geom.domain.RelPerm.Alpha ", type(model._run.Geom.domain.RelPerm.Alpha))
    #
    # # Add model attributes
    # model_params_new_values = []
    # for param in model_params:
    #     print("param ", param)
    #     val = get_nested_attr(model._run, param)
    #     print("val ", val)
    #
    #     model_params_new_values.append(get_nested_attr(model._run, param))
    # if len(model_params_new_values) > 0:
    #     initial_state_data.extend(model_params_new_values)
    #
    # print("model params new values ", model_params_new_values)

    initial_state_data = np.array(initial_state_data)

    ukf.x = initial_state_data #initial_state_data #(state.data[int(0.3/0.05)], state.data[int(0.6/0.05)])#state  # Initial state vector
    ukf.P = initial_state_covariance  # Initial state covariance matrix

    return ukf


def set_model_initial_state():
    #@TODO set model initital state including parameters and initial pressure that should be different from model instance to model instance
    pass


def run_kalman_filter(ukf, noisy_measurements, space_indices_train, space_indices_test):
    pred_loc_measurements = []
    test_pred_loc_measurements = []
    pred_model_params = []
    # Assuming you have a measurement at each time step
    print("noisy_meaesurements ", noisy_measurements)
    for measurement in noisy_measurements:
        print("measurement ", measurement)
        ukf.predict()
        ukf.update(measurement)
        print("sum ukf.P ", np.sum(ukf.P))
        print("Estimated State:", ukf.x)

        if len(model_params) > 0:
            model_params_data = ukf.x[-len(model_params):]
            pred_model_params.append(model_params_data)

        est_loc_measurements = measurement_function(ukf.x, space_indices_type="train")
        print("est loc measurements ", est_loc_measurements)

        test_est_loc_measurements = measurement_function(ukf.x, space_indices_type="test")

        pred_loc_measurements.append(est_loc_measurements)
        test_pred_loc_measurements.append(test_est_loc_measurements)

    pred_loc_measurements = np.array(pred_loc_measurements)
    test_pred_loc_measurements = np.array(test_pred_loc_measurements)

    return pred_loc_measurements, test_pred_loc_measurements, pred_model_params

def plot_model_params(pred_model_params, times):

    print("times.shape ", times.shape)
    print("pred_model_params[:, ]shape ", pred_model_params[:, ].shape)


    for idx, (param_name, param_init_value) in enumerate(model_params.items()):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        print("pred_model_params[:, {}]shape ".format(idx), pred_model_params[:, idx].shape)
        axes.hlines(y=param_init_value, xmin=0, xmax=pred_model_params.shape[0], linewidth=2, color='r')
        axes.scatter(times, pred_model_params[:, idx], marker="o", label="predictions")
        #axes.set_xlabel("param_name")
        axes.set_ylabel(param_name)
        fig.legend()
        fig.savefig("model_param_{}.pdf".format(param_name))
        plt.show()



def plot_results(pred_loc_measurements, test_pred_loc_measurements, measurements_to_test, noisy_measurements_to_test,  pred_model_params, measurements_data_name="pressure",):
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
        plot_model_params(np.array(pred_model_params), times)

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

    for i, pos in enumerate(mes_locations_to_test):
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



if __name__ == "__main__":
    import cProfile
    import pstats

    pr = cProfile.Profile()
    pr.enable()

    # Measurements locations
    mes_locations_to_train = [0.5, 1.5]
    mes_locations_to_test = [1, 2]

    measurements_data_name = "saturation"  # "pressure"

    #############################
    ### Generate measurements ###
    #############################
    model, data, measurements, noisy_measurements, measurements_to_test, noisy_measurements_to_test, state_data_iters = generate_measurements(measurements_data_name)
    residuals = noisy_measurements - measurements
    measurement_noise_covariance = np.cov(residuals, rowvar=False)
    print("measurement noise covariance ", measurement_noise_covariance)
    # print("10 percent cov ", np.cov(noisy_measurements*0.1, rowvar=False))
    # exit()
    # pde.plot_kymograph(storage)

    space_indices_train = [int(mes_loc / model._run.ComputationalGrid.DZ) for mes_loc in mes_locations_to_train]
    space_indices_test = [int(mes_loc / model._run.ComputationalGrid.DZ) for mes_loc in mes_locations_to_test]

    #######################################
    ### Unscented Kalman filter setting ###
    ### - Sigma points
    ### - initital state covariance
    ### - UKF metrices
    ########################################
    ukf = set_kalman_filter(data, measurement_noise_covariance)

    #######################################
    ### Kalman filter run ###
    ### For each measurement (time step) ukf.update() and ukf.predict() are called
    ########################################
    pred_loc_measurements, test_pred_loc_measurements, pred_model_params = run_kalman_filter(ukf, noisy_measurements, space_indices_train, space_indices_test)


    ##############################
    ### Results postprocessing ###
    ##############################
    plot_results(pred_loc_measurements, test_pred_loc_measurements, measurements_to_test, noisy_measurements_to_test, pred_model_params, measurements_data_name)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats(50)


