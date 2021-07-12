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


# Constructs graph with k nearest neighbors
# Input:
#   dm: Distance matrix - See DistMatrix.py
#   k: number of neighbors
def construct_knn_graph(dm: np.array, k: int = 5) -> csr_matrix:
    np_g = np.zeros(dm.shape, dtype=np.float16)
    for index, row in tqdm(enumerate(dm), total=len(dm)):
        temp = np.delete(row, index)
        k_max = bottleneck.argpartition(temp, k)[:k]
        k_max[k_max > index] += 1

        g_row = np.zeros(dm.shape[0])
        g_row[k_max] = 1

        np_g[index] = g_row
    return scipy.sparse.csr_matrix(np_g, dtype=np.float16)
    
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

def load_knn_g_and_oht_labels(num = 1000, k = 2, forget_percentage=.5, do_print=True):
    if do_print: print("Loading dist matrix and labels")
    dm = DMatrix.load_dist_matrix(num)
    lbls = MNIST.load_mnist_train_labels()
    lbls = lbls[:num]

    if do_print: print("Constructing knn graph")
    g = construct_knn_graph(dm, k=k)
    if do_print: print("Encoding labels")
    oht_lbls = one_hot_encode_labels(lbls)
    oht_lbls, forg_indices = forget_oht_labels(oht_lbls, forget_percentage=forget_percentage)
    sparse_oht = scipy.sparse.csr_matrix(oht_lbls)  

    return g, sparse_oht, lbls, forg_indices

def test_graph(g, pred_lbls):


if __name__ == "__main__":
    g, oht_lbls, lbls, indices = load_knn_g_and_oht_labels(num=50000, k=2, forget_percentage=.9)
    pred_lbls = propagate_labels(g, oht_lbls, max_itter=100)
    
    true_lbls = lbls[indices]
    pred_lbls = pred_lbls[indices]

    print(test_accuracy(true_lbls, pred_lbls))
    V.pred_map(pred_lbls, true_lbls)