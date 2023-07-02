from robot_simulation import RobotSimulation
from kalman_filter import KalmanFilter
# The first part of this notebook is inspired by R. Labbe's work, the second by A. Bewley's.

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

# This code generates the (walk and) measurements at each of N timesteps spaced by a time interval of delta_t = 1.0

process_std = .5
measurement_std = 2.
N = 25

robot = RobotSimulation(0, 1.0, measurement_std, process_std)
zs = [np.array([robot.move_and_locate()]) for _ in range(N)] # Array of measurements at each time step

# Initialize the Kalman filter
kf = KalmanFilter(dim_x=2, dim_z=2)

kf.x = np.array([0.,0.]).T # Initialize the location and velocity at 0.
kf.P = np.diag([20., 2.]) # Initialize the standard deviation of the location at 20. and the std of the velocity at 2.

kf.F = np.array([[1.,1.], [0., 1.]]) # Set the transition matrix to that corresponding to a constant velocity model
kf.Q = np.diag([process_std, 0.1]) # Set the process noise std to process_std for the location and 0.1 for the velocity
kf.H = np.array([1., 0.]) # Set the measurement matrix to only observe the location
kf.R = np.array([measurement_std]) # Set the measurement std to measurement_std

# Run the Kalman filter at each timestep and store the predictions in the arrays below

priors = np.zeros((N, 2)) # store the prediction of the location before each measurement along with its variance
xs = np.zeros((N+1, 2)) # store the (initial location and the) prediction of the location after each measurement + its variance
vs = np.zeros((N+1, 2)) # store the (initial velocity and the) velocity after each measurement along with the predicted variance

xs[0,:] = np.array([zs[0][0], 0.])
vs[0,:] = np.array([1., 0.])

for i, z in enumerate(zs):
    # Run the predict/update steps of the Kalman filter and fill priors, xs and vs
    kf.predict()
    priors[i, :] = np.array([kf.x[0], kf.P[0][0]**2])
    print("BAH MON REUF")
    kf.update(zs[i])
    xs[i,:] = np.array([kf.x[0], kf.P[0][0]**2])
    vs[i,:] = np.array([kf.x[1], kf.P[1][1]**2])