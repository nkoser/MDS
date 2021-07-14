import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import cv2
import keract
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K

from sklearn.utils import class_weight
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, Dropout, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Activation
from tensorflow.python.keras.utils.np_utils import to_categorical

# from vis.visualization import visualize_saliency
from tensorflow.python.layers.pooling import MaxPooling2D

from Scenario_1.meanAveragePrecision import computeMeanAveragePrecision
from Scenario_3.plotting import plotting_history_1, customize_axis_plotting


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


def get_model(num_sensors=1, input_shape=None):
    """
    Build a sequential model
    :param input_shape:
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

    model = tf.keras.Sequential()

    model.add(Conv2D(32, (1, 3)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(strides=1, pool_size=(1, 2)))

    model.add(Conv2D(64, (1, 3), ))
    model.add(Activation("relu"))
    model.add(MaxPool2D(strides=1, pool_size=(1, 2), ))

    model.add(Conv2D(64, (1, 3), ))
    model.add(Activation("relu"))
    model.add(MaxPool2D(strides=1, pool_size=(1, 2), ))

    model.add(Conv2D(128, (1, 3), ))
    model.add(Activation("relu"))
    model.add(MaxPool2D(strides=1, pool_size=(1, 2), name="last_conv"))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(5, activation='softmax'))
    # model = Sequential()
    # model.add(Conv2D(64, kernel_size=(1, 3), activation='relu'))
    # model.add(Conv2D(32, kernel_size=(1, 3), activation='relu', name="last_conv"))
    # model.add(GlobalAveragePooling2D())
    # model.add(Dense(5, activation='softmax'))
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
    sensors = sensors[sensors != 'SleepStaging.csv']
    train = []
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

    return np.expand_dims(np.stack(train, 1), -1), labels


# loading data into a dataframe
data, patients, file_names = load_data()

# loading labels and convert them to numbers
decode_dict = {'N1': 0, 'N2': 1, 'N3': 2, 'REM': 3, "WK": 4}
training_data, t_labels = get_training_data(patients, sensors=np.unique(np.array(file_names)))
X_train, X_test, y_train, y_test = train_test_split(training_data, t_labels, test_size=0.2, random_state=1)
plt.imsave("test.png", training_data[0, :, :, 0], )

training_data, t_labels = data_augmentation(training_data, t_labels)

# Compute Class weights
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_train),
                                                  y_train)
class_weights = dict(enumerate(np.array(class_weights)))

# Get a sequential model for using one channel
model = get_model(num_sensors=1, input_shape=(11, 300, 1))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.AUC(name="AUROC", curve='ROC')])
# training_data = np.expand_dims(training_data, -1)
history = model.fit(x=X_train,
                    y=to_categorical(y_train),
                    batch_size=32,
                    epochs=80,
                    validation_split=0.2,
                    verbose=1,
                    class_weight=class_weights)

plotting_history_1(history.history,
                   "final_classifier.png",
                   f=customize_axis_plotting("loss"))


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def visualize_class_activation_map(output_name, target_class):
    original_img = np.expand_dims(training_data[1, :, :, 0], -1)
    width, height, _ = original_img.shape

    # Reshape to the network input shape (3, w, h).
    img = np.array([np.transpose(np.float32(original_img), (0, 1, 2))])

    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, "last_conv")
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, _] = get_output([img])
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
    for i, w in enumerate(class_weights[:, target_class]):
        cam += w * conv_outputs[:, :, i]
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap * 0.5 + original_img
    cv2.imwrite(output_name, img)


# Evaluation

for target_class in range(5):
    visualize_class_activation_map("heatmap_class_{}.png".format(target_class), target_class)

print(decode_dict)
print(np.unique(np.array(file_names)))


predictions = model.predict_classes(X_test, verbose=1)

report = classification_report(y_test, predictions, output_dict=True)
df = pd.DataFrame(report).transpose()
p = computeMeanAveragePrecision(y_test, model.predict(X_test))
df["Mean_average_percision"] = np.concatenate([p[1], np.array([-1, -1, p[0]])])
df.to_csv("results.csv")
# activations = keract.get_activations(model, np.expand_dims(np.expand_dims(image, -1), 0))
# print("Hi")
# out = keract.display_heatmaps(activations, np.expand_dims(image, -1), save=True, directory='activation')
