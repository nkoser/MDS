import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf


from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from tensorflow.python.keras.layers import Conv1D, MaxPool1D, Dense, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical


def plot_data(name="Overview", patient="A", sensor="O1M2_10HZ.csv"):
    plt.figure()
    data[patient][sensor].iloc[:, :2].plot()
    plt.savefig(name + ".png")


def slice_per(source, step):
    slices = []
    for i in range(0, len(source), step):
        slices.append(source[i:i + step])
    return slices


def get_model(num_sensors=1):
    """
    Build a sequential model
    :param num_sensors:
    :return:
    """
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
    return model


def load_data(root_dir="/Users/niklaskoser/Documents/MDS/Scenario_3"):
    """
    Build a dictionary of dataframes. So we get:
    {patient_name : {sensor_name: }}
    :param root_dir:
    :return:
    """
    patients = [x for x in os.listdir(root_dir)]
    ffp = []
    data = {}
    for patient in patients:
        temp = {}
        for file in os.listdir(os.path.join(root_dir, patient)):
            temp[file] = pd.read_csv(os.path.join(root_dir, patient, file))
            ffp.append(file)
        data[patient] = temp
    return data, patients, ffp


def get_training_data(patients: list, sensor='O1M2_10HZ.csv'):
    train = []
    labels = []
    for k, patient in enumerate(patients):
        scaler = MinMaxScaler()
        temp = slice_per(data[patient][sensor].iloc[:, 0].values, 300)
        temp.pop(-1)
        train.append(temp)
        lbl = data[patient]['SleepStaging.csv']['Schlafstadium'].values[:-1]
        look_up, p_labels = np.unique(lbl, return_inverse=True)
        labels.append(p_labels)
    res = [scaler.fit_transform(np.asarray(d)) for d in train]
    res = np.concatenate(res, axis=0)
    labels = np.concatenate(labels)
    return res, labels


# loading data into a dataframe
data, patients, file_names = load_data()

# loading labels and convert them to numbers
decode_dict = {'N1': 0, 'N2': 1, 'N3': 2, 'REM': 3, "WK": 4}
training_data, t_labels = get_training_data(patients)

# Compute Class weights
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(t_labels),
                                                  t_labels)
class_weights = dict(enumerate(np.array(class_weights)))

# Get a sequential model for using one channel
model = get_model(num_sensors=1)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.AUC(name="AUROC", curve='ROC')])
training_data = np.expand_dims(training_data, -1)
model.fit(x=training_data,
          y=to_categorical(t_labels),
          batch_size=32,
          epochs=30,
          validation_split=0.2,
          verbose=1,
          class_weight=class_weights)
