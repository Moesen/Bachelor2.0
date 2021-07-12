from tqdm import tqdm
import time
import numpy as np

import Graph as G

def tqdm_experiment():
    for i in tqdm(range(10), position=0, leave=False):
        for j in tqdm(range(5), position=1, leave=False):
            time.sleep(.1)

def eyesperiment():
    print(np.eye(5)[[3, 1, 0, 1]])


def test_itter_to_acc(start_itter: int = 1, end_itter: int = 10, num: int = 1000, forget_percentage=.9) -> list:
                      
    g, sparse_oht, lbls = G.load_knn_g_and_oht_labels(num=num, k=2, forget_percentage=forget_percentage)
    accs = []

    for i in range(start_itter, end_itter, 5):
        print("Testing: " + str(i))
        new_g = g.copy()
        new_oht = sparse_oht.copy()
        p = G.propagate_labels(new_g, new_oht, max_itter=i)
        acc = G.test_accuracy(lbls, p)

        if len(accs) > 0 and np.isclose(accs[-1][1], acc):
            break

        accs.append((i, acc))
    return accs

if __name__ == "__main__":
    eyesperiment()