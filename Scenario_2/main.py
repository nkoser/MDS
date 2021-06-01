import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
df = pd.read_csv('static/phone_data.csv')
df = df.iloc[:, :-1]
print(df.head())

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
print(peaks)

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
    np.matmul(np.array(R).transpose().squeeze(-1), np.expand_dims(df[['wx', 'wy', 'wz']].loc[0].values, -1)).squeeze(-1)]

for t in range(1, len(df)):
    g_B_s.append(mu * g_B_s[t - 1] + (1 - mu) * df['acc_norm_sg'].values[t])
    R.append(calc_R(g_B_s[-1]))
    angles.append(np.matmul(np.array(R[t]).transpose(), np.expand_dims(df[['wx', 'wy', 'wz']].loc[t].values, -1)))

angles = [np.sum(angles[1:t + 1]) * (df['time'].loc[t-1]-df['time'].loc[t]) for t, angle in enumerate(angles)]

print(angles[0].shape)


