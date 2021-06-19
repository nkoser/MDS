import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy

from math import cos, sin
from scipy import signal
from utils import savitzky_golay


def calc_update(t, mu):
    """

    :param t:
    :param mu:
    :return:
    """
    if t == 0:
        return g_B_s
    else:
        return mu * calc_update(t - 1, mu) + (1 - mu) * df['acc_norm_sg'].values


def calc_R(g_B_s):
    """
    Calculate the Rotation/Basis Matrix R for one epoch
    :param g_B_s: get the actual g_b' gravity vector
    :return: the Rotation/Basis Matrix R
    """
    u_z = g_B_s
    u_x = np.cross(np.array([0, 1, 0]).transpose(), u_z)
    u_y = np.cross(u_z, u_x)
    R = np.array([(u_x / np.linalg.norm(u_x)), (u_y / np.linalg.norm(u_y)), (u_z / np.linalg.norm(u_z))])
    print(R)
    return R


# Load phone data into a Dataframe
df = pd.read_csv('/content/MDS/Scenario_2/static/data_circle.csv')
df = df.iloc[:, :-1]

# Plot the measured accelerometer
plt.figure()
df.iloc[:, 1:4].plot()
plt.savefig("accelerometer.png")

# Calculate the norm over the accelerometer values and plot them
df['acc_norm'] = np.linalg.norm(df[['gFx', 'gFy', 'gFz']].values, axis=1)
plt.figure()
df.iloc[:, -1].plot()
plt.savefig("norm_acc.png")

# Smoothing of the normalized accelerator data
df['acc_norm_sg'] = scipy.signal.medfilt(df['acc_norm'].values,
                                         5)  # savitzky_golay(df['acc_norm'].values, 51, 3)#savitzky_golay(df['acc_norm'].values, 51, 3) #scipy.signal.medfilt(df['acc_norm'].values, 5) #savitzky_golay(df['acc_norm'].values, 51, 3)
plt.figure()
df.iloc[:, -1].plot()
plt.savefig("norm_acc_sg.png")

# Peaks are found where you expect a step by using a threshold
idx, peaks = signal.find_peaks(df['acc_norm_sg'].values, height=1.03, )

# **********************************************************************************************************************
# ******************************************* Gyros ********************************************************************
# **********************************************************************************************************************

# Plot the gyroscope data
plt.figure()
df.iloc[:, 4:7].plot()
plt.savefig("gyros.png")

# Take the measurements where the peaks were found
df = df.iloc[idx].reset_index()

# Calculate the gravity vector g_b' for the first timestep
g_B_s_t0 = np.array([-0.05, 0.703, 0.69])  # np.array([df['wx'].mean(), df['wy'].mean(), df['wz'].mean()])
g_B_s = [g_B_s_t0]
R = [calc_R(g_B_s[0])]
mu = 0.9

# Calculate the velocity angles for the first timestep
angles = [
    np.matmul(np.array(R).transpose().squeeze(-1), np.expand_dims(df[['wx', 'wy', 'wz']].loc[0].values, -1)).squeeze(
        -1)]

# Calculate g_B' for each timestep
for t in range(1, len(df)):
    g_B_s.append(mu * g_B_s[t - 1] + (1 - mu) * df['acc_norm_sg'].values[t])
    R.append(calc_R(g_B_s[-1]))
    angles.append(np.matmul(np.array(R[t]).transpose(), np.expand_dims(df[['wx', 'wy', 'wz']].loc[t].values, -1)))

# Calculate the estimated turning angle for each epoch
tmp = [angles[0][-1] * df['time'].loc[0]]
for t in range(1, len(angles)):
    tmp.append(tmp[t - 1] + (angles[t][-1] * (df['time'].loc[t] - df['time'].loc[t - 1])))

angles = tmp

lambda_t = 70

x_t = []
y_t = []

# angles = [math.degrees(x) for x in angles]

# Calculate the x and y Position for each timestep
for i in range(len(angles)):
    if i == 0:
        x_t.append(lambda_t * cos(angles[i]))
        y_t.append(lambda_t * sin(angles[i]))
    else:
        x_t.append(x_t[i - 1] + (lambda_t * cos(angles[i])))
        y_t.append(y_t[i - 1] + (lambda_t * sin(angles[i])))

# Plot the estimated Track
plt.figure()
x_t = np.array(x_t)
y_t = np.array(y_t)
plt.plot(x_t, y_t)
plt.savefig("Track.png")
