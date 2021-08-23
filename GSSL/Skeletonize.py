from pygel3d import graph, gl_display as gd
import networkx as nx
from scipy.sparse import csr_matrix
import numpy as np
from tqdm import tqdm
import os

def from_matrix_to_pygel(g: csr_matrix) -> graph.Graph:
    print("Converting matrix to pygel")
    rows, cols = g.nonzero()
    node_pairs = list(zip(rows, cols))
    pos = np.array([[x, x] for x in range(g.shape[0])], dtype=np.float64)
    
    print("Creating nodes")
    pg = graph.Graph()
    for i, p in tqdm(enumerate(pos)):
        temp = pg.add_node(p)
        if i != temp:
            print("Wrong indice to node: ", i, temp)
    
    print("Connecting nodes")
    for px, py in tqdm(node_pairs):
        pg.connect_nodes(px, py)

    return pg

def from_matrix_embedding_to_pygel(g: csr_matrix, embedding: np.array) -> graph.Graph:
    print("Converting matrix to pygel using precomputed embedding")
    rows, cols = g.nonzero()
    node_pairs = list(zip(rows, cols))
    pg = graph.Graph()
    print("Creating nodes")
    for i, p in tqdm(enumerate(embedding)):
        temp = pg.add_node(np.float64(p))
        if i != temp:
            print("Wrong indice to node: ", i, temp)
    
    print("Connecting Nodes")
    for px, py in tqdm(node_pairs):
        pg.connect_nodes(px, py)

    return pg

def pretty_from_matrix_to_pygel(g: csr_matrix, view=False) -> graph.Graph:
    print("Converting matrix to pygel")
    rows, cols = g.nonzero()
    node_pairs = list(zip(rows, cols))
    indices = [x for x in range(g.shape[0])]    

    print("Adding nodes and edges")
    nxg = nx.Graph()
    nxg.add_nodes_from(indices)
    nxg.add_edges_from(node_pairs)

    print("Creating positions")
    pos = nx.spring_layout(nxg)
    
    print("Adding nodes to pygel")
    ppg = graph.Graph()
    for i, p in tqdm(pos.items()):
        temp = ppg.add_node(p)
        if i != temp:
            print("Something went wrong adding nodes to pygel: ", i, temp)

    print("Adding edges to pygel")
    for px, py in tqdm(node_pairs):
        ppg.connect_nodes(px, py)

    if view:
        nx.draw(nxg, pos=pos)
        viewer = gd.Viewer()
        viewer.display(ppg)

    return ppg

def from_pygel_to_matrix(spg: graph.Graph, labels: np.array, smap: np.array) -> tuple[csr_matrix, csr_matrix]:
    # If something wrong in numbering of nodes, this becomes
    # a hazzle when using np.array
    for i, nid in enumerate(spg.nodes()):
        if i != nid:
            print("Error on node")
            raise Exception()

    size = len(spg.nodes())
    smg = np.zeros((size, size), dtype=np.int8)
    
    print("Creating matrix graph")
    for i, node in tqdm(enumerate(spg.nodes()), total=size):
        neighbors = spg.neighbors(node)
        smg[i, neighbors] = 1
        smg[neighbors, i] = 1
    
    print("Creating oht label graph")
    slbls = np.zeros((size, 10), dtype=np.int16)
    for i, sid in tqdm(enumerate(smap), total=len(smap)):
            temp = labels[i]
            if temp >= 0:
                    slbls[sid, temp] += 1

    print("Choosing the most present label")
    for row in slbls:
        if np.sum(row) > 0:
            max_occurence = np.argmax(row)
            row[:] = 0
            row[max_occurence] = 1            

    return csr_matrix(smg), csr_matrix(slbls)

def remap_labels(skel_labels: csr_matrix, smap: np.array) -> np.array:
    print("Remaping labels")
    relbls = np.zeros(len(smap), dtype=np.int16)
    for i, nid in tqdm(enumerate(smap), total=len(smap)):
        relbls[i] = skel_labels[nid]
    return relbls


def front_skeletonize_pg(pg: graph.Graph, dm: np.array, num_of_colorings: int):
    np.random.seed(0)
    col_indices = np.random.choice(dm.shape[0], size=num_of_colorings, replace=False)
    colorings = dm[col_indices]
    
    skel, smap = graph.front_skeleton_and_map(pg, colorings)
    return skel, smap

def local_skeletonize_pg(pg: graph.Graph) -> graph.Graph:
    skel, smap = graph.LS_skeleton_and_map(pg)
    print(f"Skeletonization has yielded {len(pg.nodes())} -> {len(skel.nodes())} nodes")
    return skel, np.array(smap), len(skel.nodes())

def load_skeleton_and_smap(k) -> tuple[graph.Graph, np.array]:
    SKEL_PATH = r"F:\Git\Bachelor2.0\Data\Skeletons"
    s = graph.load(os.path.join(SKEL_PATH, f"skel_{k}.g"))
    smap = np.load(os.path.join(SKEL_PATH, f"smap_{k}.npy"))

    return s, smap