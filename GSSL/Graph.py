import sys
import os
cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_path)

import bottleneck
import numpy as np
import scipy.sparse
from scipy.sparse.csr import csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm

import DistMatrix as DMatrix
import MNIST
import Visualization as V
import experiments as ex


GRAPH_FOLDER = r"F:\Git\Bachelor2.1\Data\Graphs"
TESTRES_FOLDER = r"F:\Git\Bachelor2.1\Data\TestResults"

# Constructs graph with k nearest neighbors
# Input:
#   dm: Distance matrix - See DistMatrix.py
#   k: number of neighbors
def construct_knn_graph(dm: np.array, k: int = 5) -> csr_matrix:
    np_g = np.zeros(dm.shape, dtype=np.int8)
    for index, row in tqdm(enumerate(dm), total=len(dm)):
        k_max = bottleneck.argpartition(row, k) # Find k+1 smallest value in no particular order place in front
        k_max = k_max[k_max != index] # Remove the current index from results (always in due to 0 being lowest)
        k_max = k_max[:k] # Take the k front elements as they are the lowest

        np_g[index, k_max] = 1
        np_g[k_max, index] = 1
    return scipy.sparse.csr_matrix(np_g, dtype=np.int8)

def one_hot_encode_labels(lbls, nb_classes=10):
    res = np.eye(nb_classes)[np.array(lbls).reshape(-1)]
    return res

def forget_oht_labels(oht_lbls: np.array, forget_percentage: float):
    num = oht_lbls.shape[0]
    forget_num = int(num * forget_percentage)
    rng = np.random.RandomState(0)
    indices = rng.choice(num, forget_num, replace=False)
    oht_lbls[indices, :] = 0
    return oht_lbls, indices

def propagate_labels(g: csr_matrix, csr_oht_lbls: csr_matrix, max_itter=20) -> np.array:
    org_row, org_col = csr_oht_lbls.nonzero()
    for _ in tqdm(range(max_itter)):
        # Propogate Labels
        p = g.dot(csr_oht_lbls)
        # Normalise rows
        n = normalize(p, norm="l1", axis=1)

        # Reinsert already known values
        clamped = n.tolil()
        clamped[org_row] = 0
        clamped[org_row, org_col] = 1
        csr_oht_lbls = clamped.tocsr()

        

    label_matrix = csr_oht_lbls.argmax(axis=1)
    return np.squeeze(np.asarray(label_matrix))

def test_accuracy(true_lbls, pred_labels):
    return np.count_nonzero(pred_labels==true_lbls)/len(true_lbls) * 100

def load_knn_g(num, k = 2, do_print=True):
    if do_print: print("Loading dist matrix and labels")
    dm = DMatrix.load_dist_matrix(num)
    if do_print: print("Constructing knn graph")
    g = construct_knn_graph(dm, k=k)
    

    return g

def oht_labels(num, forget_percentage=.5, do_print=True) -> tuple[scipy.sparse.csr_matrix, np.array, np.array]:
    lbls = MNIST.load_mnist_train_labels()
    lbls = lbls[:num]

    if do_print: print("Encoding labels")
    oht_lbls = one_hot_encode_labels(lbls)
    oht_lbls, forg_indices = forget_oht_labels(oht_lbls, forget_percentage=forget_percentage)
    sparse_oht = scipy.sparse.csr_matrix(oht_lbls)  

    return (sparse_oht, lbls, forg_indices)

def save_graph_and_labels(g: np.array, lbls: np.array, neighbors: int):
    l = len(g)
    np.save(os.path.join(GRAPH_FOLDER, f"graph_{l}_n_{neighbors}"), g)
    np.save(os.path.join(GRAPH_FOLDER, f"labels_{l}_n_{neighbors}"), lbls)

def load_graph_and_labels(size: int, neighbors: int):
    filename = f"graph_{size}_n_{neighbors}.npy"
    if filename not in os.listdir(GRAPH_FOLDER):
        raise Exception(f"no graph with size {size}")
    g = np.load(os.path.join(GRAPH_FOLDER, f"graph_{size}_n_{neighbors}.npy"))
    l = np.load(os.path.join(GRAPH_FOLDER, f"labels_{size}_n_{neighbors}.npy"))
    return (g, l)

def save_test_result(true, pred):
    l = len(true)
    np.save(os.path.join(TESTRES_FOLDER, f"true_{l}"), true)
    np.save(os.path.join(TESTRES_FOLDER, f"pred_{l}"), pred)

def load_test_results(size: int):
    filename = f"pred_{size}.npy"
    if filename not in os.listdir(TESTRES_FOLDER):
        raise Exception(f"no graph with size {size}")
    true = np.load(os.path.join(TESTRES_FOLDER, f"true_{size}.npy"))
    pred = np.load(os.path.join(TESTRES_FOLDER, f"pred_{size}.npy"))
    return (true, pred)

if __name__ == "__main__":
    size = 10000
    neighbors = 10
    p = .5

    g = load_knn_g(num=size, k=neighbors)
    sparse_oht_lbls, lbls, forget_indices = oht_labels(size, forget_percentage=p)

    new_lbls = propagate_labels(g, sparse_oht_lbls, max_itter=20)
    
    true_lbls = lbls[forget_indices]
    pred_lbls = new_lbls[forget_indices]


    save_test_result(true_lbls, pred_lbls)
    print(test_accuracy(true_lbls, pred_lbls))
    # V.pred_map(pred_lbls, true_lbls)

    save_graph_and_labels(g.tolil().toarray(), new_lbls, neighbors)
    
