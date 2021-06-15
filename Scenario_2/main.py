from math import cos, sin

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import math
from scipy import signal
from utils import savitzky_golay


def calc_update(t, mu):
    if t == 0:
        return g_B_s
    else:
        return mu * calc_update(t - 1, mu) + (1 - mu) * df['acc_norm_sg'].values


def calc_R(g_B_s):
    u_z = g_B_s
    u_x = np.cross(np.array([0, 1, 0]).transpose(), u_z)
    u_y = np.cross(u_z, u_x)
    R = np.array([(u_x / np.linalg.norm(u_x)), (u_y / np.linalg.norm(u_y)), (u_z / np.linalg.norm(u_z))])
    return R


# Load phone data into a Dataframe
df = pd.read_csv('static/data_circle.csv')
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

df['acc_norm_sg'] = savitzky_golay(df['acc_norm'].values, 51, 3)
plt.figure()
df.iloc[:, -1].plot()
plt.savefig("norm_acc_sg.png")

peaks, _ = signal.find_peaks(df['acc_norm_sg'].values, height=1.05, )

# **********************************************************************************************************************
# ******************************************* Gyros ********************************************************************
# **********************************************************************************************************************

plt.figure()
df.iloc[:, 4:7].plot()
plt.savefig("gyros.png")

g_B_s_t0 = np.array([df['wx'].mean(), df['wy'].mean(), df['wz'].mean()])

g_B_s = [g_B_s_t0]
R = [calc_R(g_B_s[0])]
mu = 0.9

angles = [
    np.matmul(np.array(R).transpose().squeeze(-1), np.expand_dims(df[['wx', 'wy', 'wz']].loc[0].values, -1)).squeeze(
        -1)]

for t in range(1, len(df)):
    g_B_s.append(mu * g_B_s[t - 1] + (1 - mu) * df['acc_norm_sg'].values[t])
    R.append(calc_R(g_B_s[-1]))
    angles.append(np.matmul(np.array(R[t]).transpose(), np.expand_dims(df[['wx', 'wy', 'wz']].loc[t].values, -1)))

# angles = [np.sum(np.array(angles[1:t]) * (df['time'].loc[t] - df['time'].loc[t - 1]))
#          for t, angle in enumerate(angles, start=0)]
print((df['time'].loc[1] - df['time'].loc[0]))
print(angles[0])
tmp = [angles[0][-1] * df['time'].loc[0]]
for t in range(1, len(angles)):
    # t1 -> Oz_1 * (t1-t0)
    # t2 -> Oz_1 * (t1-t0) + Oz_2 * (t2-t1)
    # t3 -> t1 + t2 + Oz_2 * (t3-t2)
    # .
    # t_540
    tmp.append(tmp[t - 1] + (angles[t][-1] * (df['time'].loc[t] - df['time'].loc[t - 1])))

angles = tmp

# Plotting :D

lambda_t = 75

x_t = []
y_t = []
# angles = [math.degrees(x) for x in angles]

for i in range(len(angles)):
    if i == 0:
        x_t.append(lambda_t * cos(angles[i]))
        y_t.append(lambda_t * sin(angles[i]))
    else:
        x_t.append(x_t[i - 1] + (lambda_t * cos(angles[i])))
        y_t.append(y_t[i - 1] + (lambda_t * sin(angles[i])))
plt.figure()
plt.scatter(x_t,y_t)
for (x, y) in zip(x_t, y_t):
    print("X: {}, Y: {}".format(x, y))
plt.savefig("Hallo.png")
