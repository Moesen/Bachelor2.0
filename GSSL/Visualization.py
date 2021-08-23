import sys
import os
cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_path)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import networkx as nx
from scipy.sparse import csr_matrix



DTU_COLS = {
    "DTU_RED": (153, 0, 0),
    "BLACK": (0, 0, 0),
    "BLUE": (47, 62, 234),
    "BRGREEN": (31, 208, 130),
    "NABLUE": (3, 15, 79),
    "YELLOW": (246, 208, 77),
    "ORANGE": (252, 118, 52),
    "PINK": (247, 187, 177),
    "GREY": (218, 218, 218),
    "RED": (232, 63, 72),
    "GREEN": (0, 136, 53),
    "PURPLE": (121, 35, 142),
    "WHITE": (255, 255, 255),
}

# Visualizations of numbers
def show_number(image, label) -> None:
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.title(f"{label=}")
    plt.show()

def show_numbers(images: list[list[float]], labels: list[str]) -> None:
    rows = (len(images) // 20) + 1
    cols = min(len(images), 20)

    fig = plt.figure(figsize=(1.5 * 100, 1.9*30))
    for index, (image, label) in enumerate(zip(images, labels)):
        ax = fig.add_subplot(rows, cols, index+1)
        ax.imshow(image.reshape(28, 28), cmap="gray")
        ax.axis("off")
        ax.set_title(
            f"Etiket: {label}",
            fontsize=200,)
    plt.tight_layout()
    plt.savefig("test")
    plt.show()

def g_neighbormap(g: csr_matrix, lbls: np.array, show=True) -> np.array:
    np_g = g.tolil().toarray()
    heat = np.zeros((10, 10), dtype=np.int32)
    for i, row in enumerate(np_g):
        lbl = lbls[i]
        neighbors = np.where(row == 1)
        neighbor_lbls = lbls[neighbors]
        for n in neighbor_lbls:
            heat[lbl, n] += 1
    
    sns.heatmap(heat, cmap="PuBu", annot=True, fmt="g")
    if show: plt.show()

    return heat

def dm_similarity(dm: np.array, lbls: np.array, show=True) -> np.array:
    sim = np.zeros((10, 10), dtype=float)
    for i in range(10):
        lbl_idxs = np.where(lbls==i)
        lbl_dists = dm[lbl_idxs]
        
        for j in range(10):
            compared_dist_idxs = np.where(lbls==j)[0]
            compared_dists_data = lbl_dists[:, compared_dist_idxs]
            sim[i, j] = np.average(compared_dists_data)//1

    sns.heatmap(sim, cmap="PuBu", annot=True, fmt="g")
    if show: plt.show()
    
    return sim

def pred_map(pred_lbls: np.array, true_lbls: np.array, show=True) -> np.array:
    pred = np.zeros((10, 10), dtype=np.int16)
    for p, t in zip(pred_lbls, true_lbls):
        pred[t, p] += 1
    
    fig = sns.heatmap(pred, cmap="PuBu", annot=True, fmt="d")
    if show: plt.show()

    return pred

def line_diag(vals):
    labels = [x[0] for x in vals]
    values = [x[1] for x in vals]
    
    ax = plt.subplot()
    ax.set_xticks(labels)
    ax.set_xscale("log")
    
    for l, v in vals:
        ax.text(l, v, f"{v:.1f}%")

    ax.plot(labels, values)
    
    plt.show()

if __name__ == "__main__":
    seed = 13648  # Seed random number generators for reproducibility
    G = nx.random_k_out_graph(10, 3, 0.5, seed=seed)
    pos = nx.spring_layout(G, seed=seed)

    node_sizes = [3 + 10 * i for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    cmap = plt.cm.plasma

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="indigo")
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=10,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=2,
    )
    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)
    plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()
    plt.show()