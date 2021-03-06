import sys
import os
cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_path)

import numpy as np
import os
import pymde

MNIST_FOLDER = r"F:\Git\Bachelor2.1\Data\MNIST"
PYMDE_MNIST_FOLDER = r"F:\Git\Bachelor2.0\Data\MNIST\PYMDE"

def load_mnist_train():
    train_X = np.load(os.path.join(MNIST_FOLDER, "train_data.npy"))
    train_y = np.load(os.path.join(MNIST_FOLDER, "train_labels.npy"))
    return train_X, train_y

def load_mnist_train_data():
    return np.load(os.path.join(MNIST_FOLDER, "train_data.npy"))

def load_mnist_train_labels():
    return np.load(os.path.join(MNIST_FOLDER, "train_labels.npy"))
    
def load_mnist_test():
    test_X = np.load(os.path.join(MNIST_FOLDER, "test_data.npy"))
    test_y = np.load(os.path.join(MNIST_FOLDER, "test_labels.npy"))
    return test_X, test_y

def load_pymde_mnist(size=60000):
    mnist = pymde.datasets.MNIST(root=PYMDE_MNIST_FOLDER)
    return mnist.data.numpy()[:size], mnist.attributes["digits"].numpy()[:size]

def load_pymde_testset():
    mnist = pymde.datasets.MNIST(root=PYMDE_MNIST_FOLDER)
    return mnist.data.numpy()[60000:], mnist.attributes["digits"].numpy()[60000:]

if __name__ == "__main__":
    # Reformats mnist csv files to .npy
    # MNIST csv files found here on:
    # https://www.kaggle.com/oddrationale/mnist-in-csv
    # Remember to change folder unless it magically already matches

    TRAIN_PATH = os.path.join(MNIST_FOLDER, "mnist_train.csv")
    TEST_PATH = os.path.join(MNIST_FOLDER, "mnist_test.csv")
    
    mnist_train = np.genfromtxt(TRAIN_PATH, delimiter=",", skip_header=1)
    train_lbls = mnist_train[:, 0].astype("int8")
    train_data = mnist_train[:, 1:].astype("int16")
    np.save(os.path.join(MNIST_FOLDER, "train_data"), train_data)
    np.save(os.path.join(MNIST_FOLDER, "train_labels"), train_lbls)

    mnist_test = np.genfromtxt(TEST_PATH, delimiter=",", skip_header=1)
    test_lbls = mnist_test[:, 0].astype("int8")
    test_data = mnist_test[:, 1:].astype("int16")
    np.save(os.path.join(MNIST_FOLDER, "test_data"), test_data)
    np.save(os.path.join(MNIST_FOLDER, "test_labels"), test_lbls)