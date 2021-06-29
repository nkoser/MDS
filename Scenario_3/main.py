import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Conv1D, MaxPool1D, Dense, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical


def slice_per(source, step):
    slices = []
    for i in range(0, len(source), step):
        slices.append(source[i:i + step])
    return slices


root_dir = "/Users/niklaskoser/Documents/MDS/Scenario3"
patients = [x for x in os.listdir(root_dir)]
data = {}
for patient in patients:
    temp = {}
    for file in os.listdir(os.path.join(root_dir, patient)):
        temp[file] = pd.read_csv(os.path.join(root_dir, patient, file))
    data[patient] = temp

decode_dict = {'N1': 0, 'N2': 1, 'N3': 2, 'REM': 3, "WK": 4}
labels = data['A']['SleepStaging.csv']['Schlafstadium'].values[:-1]
look_up, t_labels = np.unique(labels, return_inverse=True)
print(np.unique(data['A']['SleepStaging.csv']['Schlafstadium'].values, return_counts=True))

scaler = MinMaxScaler()
training_data = slice_per(data['A']['O1M2_10HZ.csv'].iloc[:, 0].values, 300)
training_data.pop(-1)
training_data = scaler.fit_transform(np.asarray(training_data))
print(training_data.shape)

model = Sequential()
model.add(Conv1D(3, 3))
model.add(Conv1D(16, 3))
model.add(MaxPool1D(2))
model.add(Conv1D(32, 3))
model.add(MaxPool1D(2))
model.add(Conv1D(64, 3))
model.add(MaxPool1D(2))
model.add(Conv1D(128, 3))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.AUC(name="AUROC", curve='ROC')])
training_data = np.expand_dims(training_data, -1)
print(training_data.shape)
model.fit(x=training_data, y=to_categorical(t_labels), batch_size=16, epochs=10, verbose=1)

plt.figure()
data['A']['O1M2_10HZ.csv'].iloc[:, :2].plot()
plt.savefig("Overview.png")
