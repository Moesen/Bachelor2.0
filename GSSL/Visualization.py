import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import networkx as nx

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
        ax.set_title(f"{label}")
    plt.show()

def g_neighbormap(g: np.array, lbls: np.array, show=True) -> np.array:
    heat = np.zeros((10, 10), dtype=np.int16)
    for i, row in enumerate(g):
        lbl = lbls[i]
        neighbors = np.where(row == 1)
        neighbor_lbls = lbls[neighbors]
        for n in neighbor_lbls:
            heat[lbl, n] += 1
    
    fig = sns.heatmap(heat, cmap="PuBu", annot=True, fmt="g")
    if show: plt.show()

    return fig

def dm_similarity(dm: np.array, lbls: np.array, show=True):
    sim = np.zeros((10, 10), dtype=float)
    for i in range(10):
        lbl_idxs = np.where(lbls==i)[0]
        lbl_dists = dm[lbl_idxs]
        
        for j in range(10):
            compared_dist_idxs = np.where(lbls==j)[0]
            compared_dists_data = lbl_dists[:, compared_dist_idxs]
            sim[i, j] = np.average(compared_dists_data)//1

    fig = sns.heatmap(sim, cmap="PuBu", annot=True, fmt="g")
    if show: plt.show()
    
    return fig

def pred_map(pred_lbls: np.array, true_lbls: np.array, show=True) -> np.array:
    pred = np.zeros((10, 10), dtype=np.int16)
    for p, t in zip(pred_lbls, true_lbls):
        pred[t, p] += 1
    
    fig = sns.heatmap(pred, cmap="PuBu", annot=True, fmt="d")
    if show: plt.show()

    return fig


def visualise_graph(g: np.array, lbls):
    r, c = np.nonzero(g > 0)
    edges = np.c_[r, c]
    
    G = nx.Graph()
    G.add_edges_from(edges)
    cols = [(x[0]/255, x[1]/255, x[2]/255) for x in DTU_COLS.values()]
    
    labels = {key:val for key, val in enumerate(lbls)}
    colors = [cols[x] for x in lbls]
    cmap = plt.cm.plasma

    node_sizes = [3 + 10 * i for i in range(len(g))]
    edge_alphas = [(5 + i) / (G.number_of_edges() + 4) for i in range(G.number_of_edges())]
    
    

    seed = 13648
    pos = nx.spring_layout(G, seed=seed)
    nx_nodes = nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50)
    nx_edges = nx.draw_networkx_edges(
        G,
        pos,
        edge_color="indigo",
        edge_cmap=cmap,
        width=2
    )
    
    ax = plt.gca()
    ax.set_axis_off()
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