import os
import glob
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def get_data(base_dir):
    """
    Load the data (index 5 are the labels)
    :param base_dir: the directory where the data is stored
    :return: training and test data in a numpy list
    """
    train_path = os.path.join(base_dir, 'training')
    test_path = os.path.join(base_dir, 'testing')
    train_data = [np.load(os.path.join(train_path, f)) for f in os.listdir(train_path)]
    test_data = [np.load(os.path.join(test_path, f)) for f in os.listdir(test_path)]
    return train_data, test_data


def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


if __name__ == '__main__':
    root = r'E:\Dokumente\MDS\bbh\bbh'
    train, test = get_data(root)
    print(normalize_data(train))
