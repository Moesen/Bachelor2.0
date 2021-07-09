from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing
import concurrent.futures 

from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


from os import path

def load_mnist_train():
    folder = r"F:\Git\Bachelor2.1\Data\MNIST"
    train_X, train_y = np.load(path.join(folder, "train_data.npy")), np.load(path.join(folder, "train_label.npy"))
    return train_X, train_y

# Visualizations of numbers
def show_number(image, label) -> None:
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.title(f"{label=}")
    plt.show()

def calc_dist(i, numbers):
    cur = numbers[i]
    l = numbers.shape[0]
    dists = np.empty(l, dtype=np.float16)
    for index, number in enumerate(numbers):
        dists[index] = euclidean(cur, number)
    return i, dists



if __name__ == "__main__":
    lim = 3000
    train_X, train_y = load_mnist_train()
    smol_dat = train_X[:lim]

    dm = np.empty((lim, lim), dtype=np.float16)
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        fs = [executor.submit(calc_dist, i, smol_dat) for i in range(lim)]
        for f in tqdm(concurrent.futures.as_completed(fs), total=lim):
            i, d = f.result()
            dm[i] = d
    
    print(dm.shape)