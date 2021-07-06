import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from sklearn.utils import class_weight
from tensorflow.python.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, Dropout, Conv2D,MaxPool2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical


def data_augmentation(train, label):
    temp_noise_list = []
    labels = []
    for std in [0.05, 0.2, 0.5, 0.6]:
        noise = np.random.normal(0, std, train.shape)
        temp_noise_list.append(train + noise)
        labels.append(label)
    return np.vstack(temp_noise_list), np.hstack(labels)


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
    """model = Sequential()
    model.add(Conv1D(3, 3))
    model.add(Conv1D(16, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(2))
    model.add(Conv1D(32, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPool1D(2))
    model.add(Conv1D(64, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPool1D(2))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    return model"""

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu',padding='SAME'))
    #model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(MaxPool2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    return model



def load_data(root_dir="static/patientdata"):
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


def get_training_data(patients: list, sensors=["O1M2_10HZ.csv"]):
    sensors=sensors[sensors !='SleepStaging.csv']
    train=[]
    for sensor in sensors:
        train_temp = []
        labels = []
        for k, patient in enumerate(patients):
            temp = slice_per(data[patient][sensor].iloc[:, 0].values, 300)
            temp.pop(-1)
            train_temp.append(temp)
            lbl = data[patient]['SleepStaging.csv']['Schlafstadium'].values[:-1]
            look_up, p_labels = np.unique(lbl, return_inverse=True)
            labels.append(p_labels)
        scaler = MinMaxScaler()
        res = [scaler.fit_transform(np.asarray(d)) for d in train_temp]
        res = np.concatenate(res, axis=0)
        labels = np.concatenate(labels)
        train.append(res)

        print(res.shape)
       

    return np.expand_dims(np.stack(train,1),-1), labels


# loading data into a dataframe
data, patients, file_names = load_data()

# loading labels and convert them to numbers
decode_dict = {'N1': 0, 'N2': 1, 'N3': 2, 'REM': 3, "WK": 4}
print(file_names)
training_data, t_labels = get_training_data(patients,sensors= np.unique(np.array(file_names)))
print(training_data.shape)
print(t_labels.shape)

plt.imsave("test.png",training_data[0,:,:,0],)

training_data, t_labels = data_augmentation(training_data, t_labels)
print(training_data.shape)
print(t_labels.shape)

# Compute Class weights
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(t_labels),
                                                  t_labels)
class_weights = dict(enumerate(np.array(class_weights)))

# Get a sequential model for using one channel
model = get_model(num_sensors=1)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.AUC(name="AUROC", curve='ROC')])
training_data = np.expand_dims(training_data, -1)
model.fit(x=training_data,
          y=to_categorical(t_labels),
          batch_size=64,
          epochs=30,
          validation_split=0.2,
          verbose=1,
          class_weight=class_weights)
