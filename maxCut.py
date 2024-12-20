import numpy as np
from qiskit import *
from qiskit_aer import AerSimulator
from rustworkx.rustworkx import PyGraph
import qaoa


def maxcut_bruteforce(graph: PyGraph, log=True):
    """
    Solves the maxCut problem for the given graph by bruteforce calculation
    :param graph: the graph for which to calculate maxCut problem optimal solution
    :return: optimal_config containing the optimal bitstring and its value
    """
    max_value = -1
    optimal_config = None

    n = len(graph.nodes())

    print("Performing bruteforce search for optimal solution... (", 2**n, " possibilities)")

    for i in range(2 ** n):
        bitstring = np.binary_repr(i, n)

        value = get_bitstring_maxcut_value(graph, bitstring)

        if value > max_value:
            max_value = value
            bitstring_flipped = "".join("1" if x == "0" else "0" for x in bitstring)
            optimal_config = {
                "bitstring": bitstring,
                "bitstring_flipped": bitstring_flipped,
                "value": value
            }

    if log:
        print("[MaxCut Bruteforce] optimal bitstring: ", optimal_config["bitstring"], "and ",
              optimal_config["bitstring_flipped"], " with value: ", optimal_config["value"])

    return optimal_config


def get_bitstring_maxcut_value(graph: PyGraph, bitstring, log=False):
    """
    Returns the cut value of a given graph when partitioning the vertices by the given bitstring.
    :param graph: the graph
    :param bitstring: bitstring has 0/1 at index i when node i of the graph is part of set A/B
    :param log: True when result should be logged
    :return: cut value
    """
    if len(bitstring) != len(graph.nodes()):
        raise ValueError("Invalid bitstring of length ", len(bitstring))

    value = 0
    set = [0 if x == "0" else 1 for x in bitstring]
    for e in graph.edge_list():
        if set[e[0]] != set[e[1]]: value += 1

    if log:
        print("Graph-Cut Value for bitstring ", bitstring, " is: ", value)

    return value


def get_bitstring_maxcut_relative_value(graph: PyGraph, bitstring, optimal_config, log=False):
    """
    Returns the cut value of a graph (using the given bitstring as partition) relative to the maximum cut value.
    :param graph:
    :param bitstring: the bitstring for partitioning the vertices
    :param optimal_config: the optimal solution of the problem
    :param log: set to True if result should be logged
    :return: the relative cut value
    """
    return (get_bitstring_maxcut_value(graph, bitstring, log=log)
            / get_bitstring_maxcut_value(graph, optimal_config["bitstring"], log=log))


def simulate_qaoa_maxcut(graph, p_layers, optimal_config, shots=10000):
    """
    Simulates the qaoa maxcCut circuit using qiskit AerSimulator
    :param graph: graph for which the maxCut problem is computed
    :param p_layers: number of layers (p)
    :param optimal_config: optimal solution of the problem
    :param shots: number of samples that are taken by the simulator
    :return: the counts, which contain number of samples for each sampled bitstring
    """
    qc = qaoa.qaoa_circuit_maxcut_linear_schedule(graph, p_layers)

    # measure all qubits
    qc.measure_all()

    print("Simulating quantum circuit...")

    # simulate the quantum circuit locally using AerSimulator
    aer_sim = AerSimulator()
    circ = transpile(qc, aer_sim)
    result = aer_sim.run(circ, shots=shots).result()
    counts_wrong_order = result.get_counts(circ)

    # reverse bitstrings in counts because qiskit bit order is reversed
    counts = {}
    for bitstring in counts_wrong_order:
        counts[bitstring[::-1]] = counts_wrong_order[bitstring]

    # add all bitstrings of samples that have optimal value to optimal_config
    optimal_bitstrings = get_all_optimal_bitstrings(graph, counts, optimal_config)
    optimal_config["optimal_bitstrings"] = optimal_bitstrings
    print("All optimal bitstrings: ", optimal_bitstrings)

    return counts


def get_all_optimal_bitstrings(graph, counts, optimal_config):
    """
    Returns ALL optimal bitstrings for the given graph and maxCut problem.
    :param graph: graph
    :param counts: simulation results
    :param optimal_config: optimal solution
    :return: list of all optimal bitstrings
    """
    optimal_bitstrings = []
    for bitstring in counts:
        val = get_bitstring_maxcut_relative_value(graph, bitstring, optimal_config)
        if val == 1.0:
            optimal_bitstrings.append(bitstring)
    return optimal_bitstrings
