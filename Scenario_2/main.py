import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from utils import savitzky_golay

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

g_B_s = np.array([df['wx'].mean(), df['wy'].mean(), df['wz'].mean()])
u_z = g_B_s
u_x = np.cross(np.array([0, 1, 0]).transpose(), u_z)
u_y = np.cross(u_z, u_x)
R = np.array([(u_x/np.linalg.norm(u_x)), (u_y/np.linalg.norm(u_y)), (u_z/np.linalg.norm(u_z))])

print(R)