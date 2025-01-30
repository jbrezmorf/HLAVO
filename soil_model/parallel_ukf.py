import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, unscented_transform
from copy import deepcopy
import numpy as np


# Standalone helper function for parallel processing
def parallel_fn(args):
    """ Helper function to apply the process model in parallel. """
    fx, args, kwargs = args
    return fx(*args, **kwargs)

class ParallelUKF(UnscentedKalmanFilter):
    def __init__(self, *args, pool=None, **kwargs):
        """
        A Parallel Unscented Kalman Filter that can use multiprocessing.

        Parameters:
        -----------
        pool : multiprocessing.Pool or None
            If provided, it will be used for parallel execution of sigma point propagation.
        """
        super().__init__(*args, **kwargs)

        self.pool = pool  # Store the multiprocessing pool

        # Use parallel execution if pool is provided, else fallback to built-in map
        self.map = self.pool.map if self.pool else map

    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        """
        Computes the transformed sigma points using the process model.

        Uses multiprocessing if a Pool is provided.
        """
        if fx is None:
            fx = self.fx

        # Generate sigma points from UKF
        sigmas = self.points_fn.sigma_points(self.x, self.P)

        # Evaluate model in sigma points
        args = [(fx, (sigma, dt), fx_args) for sigma in sigmas]
        sigmas_f = self.map(parallel_fn, args)
        self.sigmas_f = np.array(sigmas_f)

    def update(self, z, R=None, UT=None, hx=None, **hx_args):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        Parameters
        ----------

        z : numpy.array of shape (dim_z)
            measurement vector

        R : numpy.array((dim_z, dim_z)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        **hx_args : keyword argument
            arguments to be passed into h(x) after x -> h(x, **hx_args)
        """

        if z is None:
            self.z = np.array([[None]*self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self._dim_z) * R

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time

        # Evaluate measurements in sigma_f points.
        args = [(hx, (sigma,), hx_args) for sigma in self.sigmas_f]
        sigmas_h = self.map(parallel_fn, args)
        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)
        self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)


        self.K = np.dot(Pxz, self.SI)        # Kalman gain
        self.y = self.residual_z(z, zp)   # residual

        # update Gaussian state estimate (x, P)
        self.x = self.x + np.dot(self.K, self.y)
        self.P = self.P - np.dot(self.K, np.dot(self.S, self.K.T))

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None