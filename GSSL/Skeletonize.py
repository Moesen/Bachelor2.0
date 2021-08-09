from pygel3d import graph, gl_display as gd
import Graph as G
import networkx as nx
from scipy.sparse import csr_matrix
import numpy as np
from tqdm import tqdm

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

def from_pygel_to_matrix(g: graph.Graph) -> csr_matrix:
    pass

def skeletonize_pg(pg: graph.Graph, dm: np.array, num_of_colorings: int):
    np.random.seed(0)
    col_indices = np.random.choice(dm.shape[0], size=num_of_colorings, replace=False)
    colorings = dm[col_indices]
    
    skel, smap = graph.front_skeleton_and_map(pg, colorings)
    return skel, smap

def map_labels(skeleton_map: np.array, lbls: np.array):
    pass