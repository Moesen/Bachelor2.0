import numpy as np
from os import path

test = np.genfromtxt(r"F:\Git\Bachelor2.1\Data\MNIST\mnist_test.csv", delimiter=",", skip_header=1)
train = np.genfromtxt(r"F:\Git\Bachelor2.1\Data\MNIST\mnist_train.csv", delimiter=",", skip_header=1)

test_label = test[:, 0]
test_data = test[:, 1:]

train_label = train[:, 0]
train_data = train[:, 1:]

folder = r"F:\Git\Bachelor2.1\Data\MNIST"

np.save(path.join(folder, "test_label"), test_label)
np.save(path.join(folder, "test_data"), test_data)
np.save(path.join(folder, "train_label"), train_label)
np.save(path.join(folder, "train_data"), train_data)
