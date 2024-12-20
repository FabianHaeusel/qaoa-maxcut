import rustworkx as rx
from rustworkx.rustworkx import PyGraph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def generate_random_regular_graph(degree, nodes):
    """
    Generates a random regular graph
    :param degree: degree of the regular graph (number of neighbors that each vertex has)
    :param nodes: total number of vertices of the wanted graph
    """
    if (degree * nodes) % 2 != 0:
        raise ValueError('degree * nodes must be even')

    g = nx.random_regular_graph(degree, nodes)

    graph = rx.PyGraph()
    graph.add_nodes_from(np.arange(0, nodes, 1))

    edge_list = [(a, b, 1.0) for a, b in g.edges]
    graph.add_edges_from(edge_list)

    return graph


def plot_graph_bitstring_colored(graph, bitstring, filename):
    """
    Plots the graph with the given bitstring as vertex partition into 2 disjoint sets.
    :param graph: the graph to print
    :param bitstring: the bitstring which is the partition
    :param filename: the filename to save the image file to
    """
    node_color = ["red" if x == "1" else "cyan" for x in bitstring]
    print("printing graph for bitstring", bitstring)
    print("node color:", node_color)
    show_and_save_graph(graph, filename, node_color)


def show_graph(graph: PyGraph, node_color=None):
    """
    Shows a visualization of the given graph using matplotlib.
    :param graph: the graph to show
    :param node_color: the colors of then nodes of the graph
    """
    if node_color is None:
        node_color = ["cyan" for _ in range(len(graph.nodes()))]

    g = nx.Graph()
    g.add_edges_from(graph.edge_list())
    pos = nx.spring_layout(g, seed=1)
    nx.draw(g, pos, node_size=800, with_labels=True, node_color=node_color)
    plt.show()


def save_graph(graph: PyGraph, filename: str):
    """
    Saves an image of a visualization of the given graph to a file with the given filename to the /output folder.
    :param graph: The graph to visualize
    :param filename: the filename to save to
    """
    g = nx.Graph()
    g.add_edges_from(graph.edge_list())
    pos = nx.spring_layout(g, seed=1)
    nx.draw(g, pos, node_size=800, with_labels=True)
    plt.savefig("output/" + filename, dpi=500)


def show_and_save_graph(graph: PyGraph, filename: str, node_color=None):
    """
    Shows the graph and saves an image of it to the /output folder.
    :param graph: the graph
    :param filename: the filename to save the image to
    :param node_color: the colors of the nodes
    """
    if node_color is None:
        node_color = ["cyan" for _ in range(len(graph.nodes()))]

    g = nx.Graph()
    g.add_nodes_from(np.arange(0, len(graph.nodes()), 1))
    g.add_edges_from(graph.edge_list())
    pos = nx.spring_layout(g, seed=1)
    nx.draw(g, pos, node_size=800, with_labels=True, node_color=node_color)
    plt.savefig("output/" + filename, dpi=500)
    plt.show()


def example_graphs():
    """
    Returns 3 interesting example graphs to performa the maxCut calculation on.
    Of course the algorithm can be applied to any other graphs :)
    :return: list of 3 graphs
    """
    n = 5
    graph = rx.PyGraph()
    graph.add_nodes_from(np.arange(0, n, 1))
    edge_list = [(0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
    graph.add_edges_from(edge_list)

    # simple small 4 nodes 4 edges graph
    n = 4
    graph2 = rx.PyGraph()
    graph2.add_nodes_from(np.arange(0, n, 1))
    edge_list = [(0, 1, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    graph2.add_edges_from(edge_list)

    # 3-regular graph with 20 nodes (randomly generated)
    g_3reg_n20 = rx.PyGraph()
    g_3reg_n20.add_nodes_from(np.arange(0, 20, 1))
    g_3reg_n20.add_edges_from(
        [(0, 10, 1.0), (0, 13, 1.0), (0, 18, 1.0), (1, 6, 1.0), (1, 12, 1.0), (1, 7, 1.0), (2, 8, 1.0), (2, 14, 1.0),
         (2, 17, 1.0), (3, 13, 1.0), (3, 9, 1.0), (3, 15, 1.0), (4, 5, 1.0), (4, 10, 1.0), (4, 16, 1.0), (5, 7, 1.0),
         (5, 6, 1.0), (6, 14, 1.0), (7, 13, 1.0), (8, 15, 1.0), (8, 11, 1.0), (9, 11, 1.0), (9, 19, 1.0),
         (10, 12, 1.0), (11, 14, 1.0), (12, 19, 1.0), (15, 17, 1.0), (16, 18, 1.0), (16, 17, 1.0), (18, 19, 1.0)])

    return graph, graph2, g_3reg_n20
