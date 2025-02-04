import struct
import math
import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dataloader import load_data
from quaternions import Quaternion
from covariances import covariance_matrix_creator

datadir = "pcd_output/velodynevlp16/data_pcl"
num_frames = len([name for name in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, name))])

points, quaternions = load_data(datadir, num_frames)

# select first frame
source_dir = datadir + '/' + '0' + '.pcd'
source = o3d.io.read_point_cloud(source_dir)

# select point to track across frames
pt = list(points[0][0])
quaternion = quaternions[0]
quaternion = quaternion.decode().split(' ')
quaternion = [int(x) for x in quaternion]
quaternion = Quaternion.from_list(quaternion)
euler_angles_pt = quaternion.to_euler_angles()
new_pt = np.append(pt, euler_angles_pt)

# Estimate covariances
search_param = o3d.geometry.KDTreeSearchParamKNN(knn=30)  # Define search parameters
source.estimate_covariances(search_param)

# Access the estimated covariances
cov = np.array(source.covariances)

# Estimate normals
source.estimate_normals()

# Compute FPFH features
fpfh = o3d.pipelines.registration.compute_fpfh_feature(source, search_param)

# Extract keypoints based on FPFH feature values
keypoints = []
kp_indices = []

for i in range(len(source.points)):
    keypoints.append([sum(fpfh.data[:, i]), i, source.points[i]])

keypoints = sorted(keypoints, key=lambda x: x[0], reverse=True)
keypoints = keypoints[0:2]
keypoints2 = [kp[1:] for kp in keypoints]

keypoints3 = [kp[2] for kp in keypoints]
new_pt = np.append(new_pt, keypoints3)

center_points = []
for i in range(num_frames):
    ith_frame_dir = datadir + '/' + str(i) + '.pcd'
    ith_frame = o3d.io.read_point_cloud(ith_frame_dir)
    ith_frame_points = np.array(ith_frame.points)
    center_points.append(np.mean(ith_frame_points, axis=0))

center_point = np.mean(center_points, axis=0)

covariances = covariance_matrix_creator(pt, cov, keypoints2, center_point)
mean_covariance = np.mean(covariances)
covariances /= mean_covariance

# covariances = np.zeros(12)
#  Simulation parameter
Qsim = np.diag([0.2, np.deg2rad(1.0)])**2  # Sensor Noise
Rsim = np.diag([1.0, np.deg2rad(10.0),     # Process Noise
                1.0, np.deg2rad(10.0),
                1.0, np.deg2rad(10.0)])**2

# EKF state covariance
# Define standard deviations for the robot's pose
std_dev_x = 0.5  # Standard deviation for x-position
std_dev_y = 0.5  # Standard deviation for y-position
std_dev_z = 0.5  # Standard deviation for z-position
std_dev_roll = np.deg2rad(10.0)  # Standard deviation for roll angle
std_dev_pitch = np.deg2rad(10.0)  # Standard deviation for pitch angle
std_dev_yaw = np.deg2rad(30.0)  # Standard deviation for yaw angle

# Number of keypoints
N = 2  # Example: 2 keypoints

# Construct the process noise covariance matrix for the robot's pose
Cx = np.diag([std_dev_x, std_dev_y, std_dev_z, std_dev_roll, std_dev_pitch, std_dev_yaw]) ** 2
#  Simulation parameter
Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2
R_sim = np.diag([1.0, np.deg2rad(10.0)]) ** 2

STATE_SIZE = 6 # State size [x, y, z, yaw, pitch, roll]
DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 175.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
LM_SIZE = 3  # LM state size [x, y, z]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Create a 3D axis

def ekf_slam(xEst, PEst, u, z):
    """
    Performs an iteration of EKF SLAM from the available information.

    :param xEst: the belief in last position
    :param PEst: the uncertainty in last position
    :param u:    the control function applied to the last position
    :param z:    measurements at this step
    :returns:    the next estimated position and associated covariance
    """

    # Predict
    xEst, PEst = predict(xEst, PEst, u)
    initP = np.eye(3)

    # Update
    xEst, PEst = update(xEst, PEst, z, initP)

    return xEst, PEst

def predict(xEst, PEst, u):
    """
    Performs the prediction step of EKF SLAM

    :param xEst: nx1 state vector
    :param PEst: nxn covariance matrix
    :param u:    6x1 control vector
    :returns:    predicted state vector, predicted covariance, jacobian of control vector, transition fx
    """
    xEst = xEst.reshape(-1,1)
    G, Fx = jacob_motion(xEst, u)
    a = motion_model(xEst[0:STATE_SIZE], u)
    xEst[0:STATE_SIZE] = a
    # Fx is an an identity matrix of size (STATE_SIZE)
    PEst = G.T @ PEst @ G + Fx.T @ Cx @ Fx
    return xEst, PEst

def motion_model(x, u):
    """
    Computes the motion model based on current state and input function.

    :param x: x1 pose estimation
    :param u: 2x1 control input [v; w]
    :returns: the resulting state after the control function is applied
    """
    F = np.identity(x.shape[0])
    # pose est 2D [x, y, yaw]
    # pose est 3D [x, y, z, yaw, pitch, roll]
    x = x.reshape(-1, 1)
    B = np.array([
        [DT * math.cos(x[5, 0]) * math.cos(x[4, 0]), 0, 0, 0, 0, 0],
        [DT * math.sin(x[5, 0]) * math.cos(x[4, 0]), 0, 0, 0, 0, 0],
        [DT * math.sin(x[4, 0]), 0, 0, 0, 0, 0],
        [0, DT, 0, 0, 0, 0],
        [0, 0, DT, 0, 0, 0],
        [0, 0, 0, DT, 0, 0]
    ])
    x = (F @ x) + (B @ u)
    return x

def update(xEst, PEst, z, initP):
    """
    Performs the update step of EKF SLAM

    :param xEst:  nx1 the predicted pose of the system and the pose of the landmarks
    :param PEst:  nxn the predicted covariance
    :param z:     the measurements read at new position
    :param initP: 2x2 an identity matrix acting as the initial covariance
    :returns:     the updated state and covariance for the system
    """
    for iz in range(len(z[:, 0])):  # for each observation
        minid = search_correspond_LM_ID(xEst, PEst, z[iz, 0:3]) # associate to a known landmark
        nLM = calc_n_LM(xEst) # number of landmarks we currently know about
        if minid == nLM: # Landmark is a NEW landmark
            print("New LM")
            # Extend state and covariance matrix
            xAug = np.vstack((xEst, calc_LM_Pos(xEst, z[iz, :])))
            PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
                              np.hstack((np.zeros((LM_SIZE, len(xEst))), initP))))
            xEst = xAug
            PEst = PAug
        lm = get_LM_Pos_from_state(xEst, minid)
        y, S, H = calc_innovation(lm, xEst, PEst, z[iz, 0:3], minid)
        S = S.astype(float)
        K = (PEst @ H.T) @ np.linalg.inv(S) # Calculate Kalman Gain
        xEst = np.squeeze(xEst)
        xEst = xEst + (K @ y)
        xEst = xEst.reshape(-1, 1)
        PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst

    xEst[2] = pi_2_pi(xEst[2])
    return xEst, PEst

def flatten(nested_list):
    return eval(str(nested_list).replace('[', '').replace(']', ''))

def calc_innovation(lm, xEst, PEst, z, LMid):
    """
    Calculates the innovation based on expected position and landmark position

    :param lm:   landmark position
    :param xEst: estimated position/state
    :param PEst: estimated covariance
    :param z:    read measurements
    :param LMid: landmark id
    :returns:    returns the innovation y, and the jacobian H, and S, used to calculate the Kalman Gain
    """
    delta = lm - xEst[0:3]
    q = (delta.T @ delta)[0, 0]
    xEst = xEst.reshape((-1,1))
    zangle1 = math.atan2(delta[1, 0], delta[0, 0]) - xEst[3, 0]
    zangle2 = math.atan2(delta[2, 0], math.sqrt(delta[0, 0]**2 + delta[1, 0]**2)) - xEst[4, 0]
    angle_plus_pi1 = pi_2_pi(zangle1)
    angle_plus_pi2 = pi_2_pi(zangle2)
    angle_plus_pi1 = flatten(angle_plus_pi1)
    angle_plus_pi2 = flatten(angle_plus_pi2)
    zp = np.array([math.sqrt(q), angle_plus_pi1, angle_plus_pi2])
    # zp is the expected measurement based on xEst and the expected landmark position

    y = (z - zp).T # y = innovation
    y[1] = pi_2_pi(y[1])

    H = jacobH(q, delta, xEst, LMid + 1)
    S = H @ PEst @ H.T + Cx[0:3, 0:3]

    return y, S, H

def jacobH(q, delta, x, i):
    """
    Calculates the jacobian of the measurement function

    :param q:     the range from the system pose to the landmark
    :param delta: the difference between a landmark position and the estimated system position
    :param x:     the state, including the estimated system position
    :param i:     landmark id + 1
    :returns:     the jacobian H
    """
    sq = math.sqrt(q)
    delta = delta.astype(float)
    q = flatten(q)
    delta_x = delta[0, 0]
    delta_y = delta[1, 0]
    delta_z = delta[2, 0]
    range_sq_xy = delta_x**2 + delta_y**2
    G = np.array([
                [-delta_x / sq, -delta_y / sq, -delta_z / sq, 0, 0, 0, delta_x / sq, delta_y / sq, delta_z / sq],
                [delta_y / range_sq_xy, -delta_x / range_sq_xy, 0, 0, 0, 0, -delta_y / range_sq_xy, delta_x / range_sq_xy, 0],
                [-delta_x * delta_z / (q * np.sqrt(range_sq_xy)), -delta_y * delta_z / (q * np.sqrt(range_sq_xy)), 
                np.sqrt(range_sq_xy) / q, 0, 0, 0, delta_x * delta_z / (q * np.sqrt(range_sq_xy)), 
                delta_y * delta_z / (q * np.sqrt(range_sq_xy)), -np.sqrt(range_sq_xy) / q]
                ])
    G = G / q
    nLM = calc_n_LM(x)
    F1 = np.hstack((np.eye(6), np.zeros((6, 3 * nLM))))
    F2 = np.hstack((np.zeros((3, 3)), 
                    np.zeros((3, 3 * (i))),
                    np.eye(3), 
                    np.zeros((3, 3 * nLM - 3 * i))))
    F = np.vstack((F1, F2))
    H = G @ F
    return H

def observation(xTrue, xd, u, RFID):
    """
    :param xTrue: the true pose of the system
    :param xd:    the current noisy estimate of the system
    :param u:     the current control input
    :param RFID:  the true position of the landmarks

    :returns:     Computes the true position, observations, dead reckoning (noisy) position,
                  and noisy control function
    """
    xTrue = motion_model(xTrue, u)
    # add noise to gps x-y
    z = np.zeros((0, 3))

    for i in range(len(RFID[:, 0])): # Test all beacons, only add the ones we can see (within MAX_RANGE)
        dx = RFID[i, 0] - xTrue[0, 0]
        dy = RFID[i, 1] - xTrue[1, 0]
        dz = RFID[i, 2] - xTrue[2, 0]
        d = math.sqrt(dx**2 + dy**2 + dz**2)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            print('new beacon')
            dn = d + np.random.randn() * Qsim[0, 0]  # add noise
            anglen = angle + np.random.randn() * Qsim[1, 1]  # add noise
            zi = np.array([dn, anglen, i])
            z = np.vstack((z, zi))

    # add noise to input
    ud = np.array([[
        u[0, 0] + np.random.randn() * Rsim[0, 0],
        u[1, 0] + np.random.randn() * Rsim[1, 1],
        u[2, 0] + np.random.randn() * Rsim[2, 2],
        u[3, 0] + np.random.randn() * Rsim[3, 3],
        u[4, 0] + np.random.randn() * Rsim[4, 4],
        u[5, 0] + np.random.randn() * Rsim[5, 5]
        ]]).T

    xd = motion_model(xd, ud)
    return xTrue, z, xd, ud

def calc_n_LM(x):
    """
    Calculates the number of landmarks currently tracked in the state
    :param x: the state
    :returns: the number of landmarks n
    """
    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n


def jacob_motion(x, u):
    """
    Calculates the jacobian of motion model.

    :param x: The state, including the estimated position of the system
    :param u: The control function
    :returns: G:  Jacobian
              Fx: STATE_SIZE x (STATE_SIZE + 2 * num_landmarks) matrix where the left side is an identity matrix
    """

    x = x.reshape((-1, 1))
    NEW_STATE_SIZE = x.shape[0]
    Fx = np.hstack((np.eye(STATE_SIZE), np.zeros(
        (STATE_SIZE, LM_SIZE * calc_n_LM(x)))))
    jF = np.array([
        [0.0, 0.0, 0.0, -DT * u[0, 0] * np.sin(x[3, 0]), 0.0, 0.0],
        [0.0, 0.0, 0.0, DT * u[0, 0] * np.cos(x[3, 0]), 0.0, 0.0],
        [0.0, 0.0, 0.0, DT * u[0, 0], 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
    G = np.eye(NEW_STATE_SIZE) + Fx.T @ jF.T @ Fx
    return G, Fx,

def calc_LM_Pos(x, z):
    """
    Calculates the pose in the world coordinate frame of a landmark at the given measurement.

    :param x: [x; y; z; theta; phi; psi]
    :param z: [range; bearing]
    :returns: [x; y; z] for given measurement
    """
    zp = np.zeros((3, 1))
    zp[0, 0] = x[0, 0] + z[0] * math.cos(z[2]) * math.cos(x[3, 0] + z[1])
    zp[1, 0] = x[1, 0] + z[0] * math.cos(z[2]) * math.sin(x[3, 0] + z[1])
    zp[2, 0] = x[2, 0] + z[0] * math.sin(z[2])

    return zp


def get_LM_Pos_from_state(x, ind):
    """
    Returns the position of a given landmark

    :param x:   The state containing all landmark positions
    :param ind: landmark id
    :returns:   The position of the landmark
    """
    STATE_SIZE = 6  # State size [x,y,z,yaw,pitch,roll]
    LM_SIZE = 3  # LM state size [x,y,z]
    x = x.reshape(-1,1)
    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]
    return lm

def search_correspond_LM_ID(xAug, PAug, zi):
    """
    Landmark association with Mahalanobis distance.

    If this landmark is at least M_DIST_TH units away from all known landmarks,
    it is a NEW landmark.

    :param xAug: The estimated state
    :param PAug: The estimated covariance
    :param zi:   the read measurements of specific landmark
    :returns:    landmark id
    """

    nLM = calc_n_LM(xAug)

    mdist = []
    for i in range(nLM):
        xAug = xAug.reshape(-1,1)
        lm = get_LM_Pos_from_state(xAug, i)
        y, S, H = calc_innovation(lm, xAug, PAug, zi, i)
        S = np.squeeze(S).astype(float)
        mdist.append(y.T @ np.linalg.inv(S) @ y)

    mdist.append(M_DIST_TH)  # new landmark

    minid = mdist.index(min(mdist))

    return minid

def calc_input():
    # Linear velocities in x, y, z directions [m/s]
    v_x = 1.0
    v_y = 0.5
    v_z = 0.2
    
    # Angular velocities around roll, pitch, yaw axes [rad/s]
    roll_rate = 0.1
    pitch_rate = 0.05
    yaw_rate = 0.1

    # Control input vector u
    u = np.array([[v_x, v_y, v_z, roll_rate, pitch_rate, yaw_rate]]).T
    return u

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def main():
    print(" start!!")

    time = 0.0

    # RFID positions [x, y, z]
    RFID = np.vstack(keypoints3)

    # State Vector [x, y, z, yaw, pitch, roll]'
    xTrue = np.array(new_pt)
    xEst = np.zeros((xTrue.shape[0], 1))
    xTrue2 = xTrue[0: STATE_SIZE]
    PEst = covariances

    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue2
    hxDR = xTrue2
    hxEst = hxEst.reshape(-1, 1)
    hxDR = hxDR.reshape(-1, 1)
    hxTrue = hxTrue.reshape(-1, 1)

    count = 0

    while SIM_TIME >= time:
        print('COUNT', count)
        time += DT
        u = calc_input()
        xTrue2, z, xDR, ud = observation(xTrue2, xDR, u, RFID)
        xEst, PEst = ekf_slam(xEst, PEst, ud, z)
        xEst2 = xEst[0: len(hxEst)]

        # store data history
        hxEst = np.hstack((hxEst, xEst2))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue2))

        # Clear current plot
        ax.cla()

        ax.plot(RFID[:, 0], RFID[:, 1], RFID[:, 2], "*k")
        ax.plot(xEst[0], xEst[1], xEst[2], ".r")

        # plot landmark
        for i in range(calc_n_LM(xTrue)):
            ax.plot(xEst[STATE_SIZE + i * 2],
                    xEst[STATE_SIZE + i * 2 + 1],
                    xEst[STATE_SIZE + i * 2 + 2],
                    "xg")

        ax.plot(hxTrue[0, :],
                hxTrue[1, :],
                hxTrue[2, :],
                "-b", label = 'True')
        ax.plot(hxDR[0, :],
                hxDR[1, :],
                hxDR[2, :],
                "-k", label = 'Dead reckoning')
        ax.plot(hxEst[0, :],
                hxEst[1, :],
                hxEst[2, :],
                "-r", label = 'Estimate')
        ax.axis("equal")
        ax.grid(True)
        plt.pause(0.001)
        count += 1

main()

# Create animation
plt.show()