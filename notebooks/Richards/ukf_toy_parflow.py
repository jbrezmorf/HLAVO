import pde  # py-pde
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

import dataclasses
from parflow.tools import settings
#from notebooks.Richards.richards_sim import mu_from_h, h_from_mu, RichardsPDE, Hydraulic
from notebooks.Richards.toy_parflow import ToyProblem
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints

# mes_1_loc = 0.1
# mes_2_loc = 0.3

def generating_meassurements(data_name):
    #############################
    ##   Model initialization  ##
    #############################
    model = ToyProblem(workdir="output-toy")
    model.setup_config()
    model.set_init_pressure()
    model.run()

    cwd = settings.get_working_directory()
    settings.set_working_directory(model._workdir)

    #############################
    ### Generate measurements ###
    #############################
    data = model._run.data_accessor
    ntimes = len(data.times)
    nz = data.pressure.shape[0]
    pressure = np.zeros((ntimes, nz))


    measurements = get_measurements(model._run.data_accessor, space_step=model._run.ComputationalGrid.DZ,
                                    mes_locations=mes_locations_to_train, data_name=data_name)

    measurements_to_test = get_measurements(model._run.data_accessor, space_step=model._run.ComputationalGrid.DZ,
                                            mes_locations=mes_locations_to_test, data_name=data_name)

    noisy_measurements = add_noise(np.array(measurements), level=0.1)
    noisy_measurements_to_test = add_noise(np.array(measurements_to_test), level=0.1)

    measurements = np.array(measurements)
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
    return model, data, measurements, noisy_measurements, measurements_to_test, noisy_measurements_to_test


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


def add_noise(measurements, level=0.1):
    noisy_measurements = np.zeros(measurements.shape)
    for i in range(measurements.shape[1]):
        for idx, mes_val in enumerate(measurements[:, i]):
            noise = np.random.uniform(-level* mes_val, level*mes_val)
            noisy_measurements[idx, i] = measurements[idx, i] + noise
    return noisy_measurements


#####################
### Kalman filter ###
#####################
def state_transition_function(state_pressure_saturation, dt):
    #print("state function call")
    #print("pressure data ", pressure_data)
    space_indices_train = get_space_indices(type="train")
    space_indices_test = get_space_indices(type="test")

    len_additional_data = get_len_additional_data()

    print("len state pressure saturation shape ", state_pressure_saturation.shape)
    pressure_data = state_pressure_saturation[0:-len_additional_data]  # Extract saturation from state vector
    print("pressure data shape ", pressure_data.shape)
    
    pressure_data = pressure_data.reshape(pressure_data.shape[0], 1, 1)

    model = ToyProblem(workdir="output-toy")
    model.setup_config()
    model.set_init_pressure(init_p=pressure_data)
    model._run.TimingInfo.StopTime = 1
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

    next_state_init = np.array(next_state_init)

    #next_state_init = np.array([np.squeeze(data.pressure), np.squeeze(saturation_data_loc)]).flatten()
    print("next state inint shape ", next_state_init.shape)
    return next_state_init


def get_space_indices(type="train"):
    if type == "train":
        return [int(mes_loc / model._run.ComputationalGrid.DZ) for mes_loc in mes_locations_to_train]
    elif type == "test":
        return [int(mes_loc / model._run.ComputationalGrid.DZ) for mes_loc in mes_locations_to_test]


def get_len_additional_data():
    # @TODO: return additional data length
    return len(get_space_indices(type="train")) + len(get_space_indices(type="test"))


def measurement_function(pressure_data, space_indices_type=None):
    print("type(pressure_data) ", pressure_data)

    len_additional_data = get_len_additional_data()
    additional_data = pressure_data[-len_additional_data:]

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

    num_locations = data.pressure.shape[0] + get_len_additional_data() # pressure + saturation
    #dt = 60*60  # Time between steps in seconds
    dim_z = len(mes_locations_to_train)  # Number of measurement inputs
    #initial_covariance = np.cov(noisy_measurements, rowvar=False)

    initial_covariance = np.eye(num_locations) * 1e-1
    #sigma_points = JulierSigmaPoints(n=n, kappa=1)
    sigma_points = MerweScaledSigmaPoints(n=num_locations, alpha=1e-2, beta=2.0, kappa=1, sqrt_method=sqrt_func)
    # Initialize the UKF filter
    time_step = 1 # one hour time step
    ukf = UnscentedKalmanFilter(dim_x=num_locations, dim_z=dim_z, dt=time_step,
                                fx=state_transition_function, hx=measurement_function, points=sigma_points)
    ukf.Q = np.ones(num_locations) * 5e-8
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
    initial_state_data = list(np.squeeze(data.pressure))
    initial_state_data.extend(list(np.squeeze(data.saturation[space_indices_train])))

    if len(space_indices_test) > 0:
        initial_state_data.extend(list(np.squeeze(data.saturation[space_indices_test])))

    initial_state_data = np.array(initial_state_data)

    ukf.x = initial_state_data #initial_state_data #(state.data[int(0.3/0.05)], state.data[int(0.6/0.05)])#state  # Initial state vector
    ukf.P = initial_covariance  # Initial state covariance matrix

    return ukf


def run_kalman_filter(ukf, noisy_measurements, space_indices_train, space_indices_test):
    pred_loc_measurements = []
    test_pred_loc_measurements = []
    # Assuming you have a measurement at each time step
    print("noisy_meaesurements ", noisy_measurements)
    for measurement in noisy_measurements:
        print("measurement ", measurement)

        ukf.predict()
        ukf.update(measurement)
        print("sum ukf.P ", np.sum(ukf.P))
        print("Estimated State:", ukf.x)
        est_loc_measurements = measurement_function(ukf.x, space_indices_type="train")
        print("est loc measurements ", est_loc_measurements)

        test_est_loc_measurements = measurement_function(ukf.x, space_indices_type="test")

        pred_loc_measurements.append(est_loc_measurements)
        test_pred_loc_measurements.append(test_est_loc_measurements)

    pred_loc_measurements = np.array(pred_loc_measurements)
    test_pred_loc_measurements = np.array(test_pred_loc_measurements)

    return pred_loc_measurements, test_pred_loc_measurements


def plot_results(pred_loc_measurements, test_pred_loc_measurements, measurements_to_test, noisy_measurements_to_test, measurements_data_name="pressure"):
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

    mes_locations_to_train = [0.5, 1.5]
    mes_locations_to_test = [1, 2]

    measurements_data_name = "saturation"

    #############################
    ### Generate measurements ###
    #############################
    model, data, measurements, noisy_measurements, measurements_to_test, noisy_measurements_to_test = generating_meassurements(measurements_data_name)
    residuals = noisy_measurements - measurements
    measurement_noise_covariance = np.cov(residuals, rowvar=False)
    print("measurement noise covariance ", measurement_noise_covariance)
    # print("10 percent cov ", np.cov(noisy_measurements*0.1, rowvar=False))
    # exit()
    # pde.plot_kymograph(storage)

    space_indices_train = [int(mes_loc / model._run.ComputationalGrid.DZ) for mes_loc in mes_locations_to_train]
    space_indices_test = [int(mes_loc / model._run.ComputationalGrid.DZ) for mes_loc in mes_locations_to_test]


    ukf = set_kalman_filter(data, measurement_noise_covariance)
    pred_loc_measurements, test_pred_loc_measurements = run_kalman_filter(ukf, noisy_measurements, space_indices_train, space_indices_test)
    plot_results(pred_loc_measurements, test_pred_loc_measurements, measurements_to_test, noisy_measurements_to_test, measurements_data_name)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats(50)


