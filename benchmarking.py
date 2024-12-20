from enum import Enum
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from rustworkx.rustworkx import PyGraph
import maxCut
import util
import graph_util


class BenchmarkStrategy(Enum):
    MOST_SAMPLED = 0,
    MAX_VALUE = 1,
    OPTIMAL_SAMPLES = 2,

def qaoa_benchmark_most_sampled_bitstring(graph: PyGraph, p_layers, optimal_config, counts, plot_config):
    """
    Benchmarks the simulation results by returning the value of the most sampled bitstring relative to the value
    of the optimal bitstring.
    :param graph: the graph where maxCut is applied to
    :param p_layers: number of layers of qaoa
    :param optimal_config: optimal solution of problem
    :param counts: qaoa simulation results
    :param plot_config: configures whether to plot
    :return: benchmark score in [0,1]
    """
    most_sampled_bitstring = None
    max_sample = -1
    for bitstring in counts:
        if counts[bitstring] > max_sample:
            most_sampled_bitstring = bitstring
            max_sample = counts[bitstring]

    val = maxCut.get_bitstring_maxcut_relative_value(graph, most_sampled_bitstring, optimal_config, True)
    print()
    if val == 1:
        print("p =", p_layers, ": most sampled bitstring: ", most_sampled_bitstring,
              ", relative value of most sampled bitstring: ", util.bcolors.OKGREEN, val, util.bcolors.ENDC)
        print(util.bcolors.OKGREEN + "Most sampled bitstring is optimal solution" + util.bcolors.ENDC)
    else:
        print("p =", p_layers, ": most sampled bitstring: ", most_sampled_bitstring,
              ", relative value of most sampled bitstring: ", val)

    if plot_config["plot_result_graph"]:
        graph_util.plot_graph_bitstring_colored(graph, most_sampled_bitstring, plot_config["result_graph_filename"])

    return val


def qaoa_benchmark_max_value_bitstring(graph: PyGraph, p_layers, optimal_config, counts):
    """
    Benchmarks the simulation results by returning the maximum value out of all sampled bitstring relative to the value
    of the optimal bitstring.
    :param graph: the graph where maxCut is applied to
    :param p_layers: number of layers of qaoa
    :param optimal_config: optimal solution of problem
    :param counts: qaoa simulation results
    :return: benchmark score in [0,1]
    """
    max_value = -1
    best_bitstring = None
    for bitstring in counts:
        val = maxCut.get_bitstring_maxcut_relative_value(graph, bitstring, optimal_config, False)
        if val > max_value:
            max_value = val
            best_bitstring = bitstring

    print("p=", p_layers, ": bitstring sampled with highest value: ", best_bitstring,
          " val: ", max_value)
    return max_value


def qaoa_benchmark_optimal_samples(optimal_config, counts, shots):
    """
    Benchmarks the simulation by returning the number of samples of optimal bitstring(s) divided by all samples
    :param optimal_config: optimal solution to the problem
    :param counts: sampling results from qaoa
    :param shots: number of samples done
    :return: percentage of optimal bitstrings
    """
    # get number of samples of optimal solution of problem
    samples = 0

    if "optimal_bitstrings" not in optimal_config:
        raise ValueError("Optimal config does not contain optimal_bitstrings")

    for bs in optimal_config["optimal_bitstrings"]:
        if bs in counts:
            samples += counts[bs]

    print("[OPTIMAL SAMPLES] optimal bitstrings were sampled", samples, "times of", shots, "shots (",
          samples / shots * 100, "%)")

    return samples / shots


def qaoa_benchmark(graph: PyGraph, p_layers, shots, optimal_config, strategy: BenchmarkStrategy, counts, plot_config):
    """
    Benchmarks the given qaoa maxCut configuration with the given strategy and returns a percentage value which
    can be used to compare.
    :param graph: the graph to which the maxCut problem is applied
    :param p_layers: number of layers of qaoa
    :param shots: number of samples
    :param optimal_config: optimal solution of the problem
    :param strategy: the benchmark strategy to use
    :param counts: sampling results from simulation
    :param plot_config: configuration for plotting graphs
    :return: percentage (= benchmark score)
    """

    val = 0
    if strategy == BenchmarkStrategy.MOST_SAMPLED:
        # the bitstring that is sampled the most -> value of that bitstring relative to max value
        val = qaoa_benchmark_most_sampled_bitstring(graph, p_layers, optimal_config, counts, plot_config)
    elif strategy == BenchmarkStrategy.MAX_VALUE:
        # the bitstring from the samples that gives the highest problem value -> value of that bitstring relative
        val = qaoa_benchmark_max_value_bitstring(graph, p_layers, optimal_config, counts)
    elif strategy == BenchmarkStrategy.OPTIMAL_SAMPLES:
        # the percentage of samples of the optimal solution (from bruteforce)
        val = qaoa_benchmark_optimal_samples(optimal_config, counts, shots)

    print("number of samples: ", shots)

    return val


def qaoa_run_and_benchmark(graph: PyGraph, p_layers, shots, optimal_config, strategy: BenchmarkStrategy, plot_config):
    """
    Runs the qaoa maxCut simulation and performs benchmarking of the results.
    :param graph: the graph to which the maxCut problem is applied
    :param p_layers: number of layers of qaoa
    :param shots: number of samples
    :param optimal_config: optimal solution of the problem
    :param strategy: the benchmark strategy to use
    :param plot_config: configuration for plotting graphs
    :return: percentage (= benchmark score)
    """
    print("[QAOA Benchmark] Using graph with", len(graph.nodes()), "nodes: ")
    print("edges: ", graph.edge_list())

    counts = maxCut.simulate_qaoa_maxcut(graph, p_layers, optimal_config, shots)

    if plot_config["plot_counts_diagram"]:
        # plot sample counts diagram

        cut = 24
        if len(counts) > cut:
            # limit to top 24 counts (for better readability of the diagram)
            counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:cut])

        fig, ax = plt.subplots()
        plot_histogram(counts, title="Bitstring samples for p=" + str(p_layers) + " and " + str(shots) + " shots (optimal solutions in red)", ax=ax)
        bars = ax.patches
        sorted_keys = sorted(counts.keys())

        for bar, key in zip(bars, sorted_keys):
            if key in optimal_config["optimal_bitstrings"]:
                # Change optimal solution sample bars to red
                bar.set_color('red')

        # save the plot if needed
        if "counts_diagram_filename" in plot_config and plot_config["counts_diagram_filename"] != "":
            plt.savefig("output/" + plot_config["counts_diagram_filename"], dpi=500)
        plt.show()

    return qaoa_benchmark(graph, p_layers, shots, optimal_config, strategy, counts, plot_config)
