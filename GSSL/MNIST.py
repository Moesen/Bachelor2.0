import numpy as np
import os

MNIST_FOLDER = r"F:\Git\Bachelor2.1\Data\MNIST"

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

if __name__ == "__main__":
    MNIST_PATH = r"F:\Git\Bachelor2.1\Data\MNIST"
    TRAIN_PATH = os.path.join(MNIST_PATH, "mnist_train.csv")
    TEST_PATH = os.path.join(MNIST_PATH, "mnist_test.csv")
    
    mnist_train = np.genfromtxt(TRAIN_PATH, delimiter=",", skip_header=1)
    train_lbls = mnist_train[:, 0].astype("int8")
    train_data = mnist_train[:, 1:].astype("int16")
    np.save(os.path.join(MNIST_PATH, "train_data"), train_data)
    np.save(os.path.join(MNIST_PATH, "train_labels"), train_lbls)

    mnist_test = np.genfromtxt(TEST_PATH, delimiter=",", skip_header=1)
    test_lbls = mnist_test[:, 0].astype("int8")
    test_data = mnist_test[:, 1:].astype("int16")
    np.save(os.path.join(MNIST_PATH, "test_data"), test_data)
    np.save(os.path.join(MNIST_PATH, "test_labels"), test_lbls)