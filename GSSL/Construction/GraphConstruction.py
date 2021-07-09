# Gustav Lang Moesmand (s174169@student.dtu.dk)

# Framework for constructing graphs from a given dataset.
# Right now only works with mnist numbers
# Pygel3d is used to visualise a lower dimensional representation of graph

import matplotlib.pyplot as plt
import numpy as np
# from scipy.sparse import coo_matrix
from scipy.spatial.distance import euclidean
import concurrent.futures
import os
import concurrent.futures
from tqdm import tqdm

DIST_PATH = r"F:\Git\Bachelor2.1\Data\DistMatrices"

def load_mnist_train():
    folder = r"F:\Git\Bachelor2.1\Data\MNIST"
    train_X = np.load(os.path.join(folder, "train_data.npy"))
    train_y = np.load(os.path.join(folder, "train_label.npy"))
    return train_X, train_y

def calc_dist(i, numbers):
    cur = numbers[i]
    l = numbers.shape[0]
    dists = np.empty(l, dtype=np.float16)
    for index, number in enumerate(numbers):
        dists[index] = euclidean(cur, number)
    return i, dists

def construct_dist_matrix(mnist_data: np.array) -> np.array:
    entries = mnist_data.shape[0]
    dm = np.empty((entries, entries), dtype=np.float16)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        fs = [executor.submit(calc_dist, i, mnist_data) for i in range(entries)]
        for f in tqdm(concurrent.futures.as_completed(fs), total=entries, position=0, leave=True):
            i, d = f.result()
            dm[i] = d
    return dm

def save_dist_matrix(dm: np.array):
    filename = "dist_matrice_" + str(len(dm))
    np.save(os.path.join(DIST_PATH, filename), dm)

def load_dist_matrix(size: str) -> np.array:
    filename = "dist_matrice_" + size + ".npy"
    if not(os.listdir(os.path.join(DIST_PATH, filename))):
        raise Exception("Dist matrix size not in folder")
    return np.load(os.path.join(DIST_PATH, filename))

# Visualizations of numbers
def show_number(image, label) -> None:
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.title(f"{label=}")
    plt.show()

def show_numbers(images: list[list[float]], labels: list[str]) -> None:
    rows = (len(images) // 5) + 1
    cols = min(len(images), 5)

    fig = plt.figure(figsize=(1.5 * 10, 1.9*3))
    for index, (image, label) in enumerate(zip(images, labels)):
        ax = fig.add_subplot(rows, cols, index+1)
        ax.imshow(image.reshape(28, 28), cmap="gray")
        ax.axis("off")
        ax.set_title(f"Label: {label}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data, labels = load_mnist_train ()
    data = data[:10000]
    dm = construct_dist_matrix(data)
    save_dist_matrix(dm)