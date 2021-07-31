# Gustav Lang Moesmand (s174169@student.dtu.dk)

# Framework for constructing graphs from a given dataset.
import sys
import os
cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_path)

import numpy as np
from scipy.spatial.distance import euclidean
import concurrent.futures
import os
import concurrent.futures
from tqdm import tqdm
import MNIST

DIST_PATH = r"F:\Git\Bachelor2.1\Data\DistMatrices"

def calc_dist(i, numbers):
    cur = numbers[i]
    l = numbers.shape[0]
    dists = np.zeros(l, dtype=np.float16)
    for index, number in enumerate(numbers):
        dists[index] = euclidean(cur, number)
    return i, dists

def construct_dist_matrix(mnist_data: np.array, workers=8) -> np.array:
    entries = mnist_data.shape[0]
    dm = np.zeros((entries, entries), dtype=np.float16)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        fs = [executor.submit(calc_dist, i, mnist_data) for i in range(entries)]
        for f in tqdm(concurrent.futures.as_completed(fs), total=entries, position=0, leave=True):
            i, d = f.result()
            dm[i] = d
    return dm

def save_dist_matrix(dm: np.array):
    filename = "dist_matrice_" + str(len(dm))
    np.save(os.path.join(DIST_PATH, filename), dm)

def load_dist_matrix(size: str) -> np.array:
    filename = "dist_matrice_" + str(size) + ".npy"
    if filename not in os.listdir(DIST_PATH):
        raise Exception("Dist matrix size not in folder")
    return np.load(os.path.join(DIST_PATH, filename))

def is_symmetric(arr: np.array) -> bool:
    return np.allclose(arr.transpose(1, 0), arr)

if __name__ == "__main__":
    data, labels = MNIST.load_mnist_train()
    dm = construct_dist_matrix(data, workers=10)
    save_dist_matrix(dm)


