import matplotlib.pyplot as plt
from rustworkx.rustworkx import PyGraph
from matplotlib import pyplot as plot
import maxCut
import benchmarking
import graph_util


def plot_influence_of_p(graph: PyGraph, plot_config):
    """
    Plots diagrams showing the influence on the quantum circuit simulation outcome bitstring values when increasing p,
    the number of layers of the quantum circuit for a fixed given graph.
    :param graph: the graph for which the calculations are done
    :param plot_config: configures what should be logged and plotted while running
    """
    shots = 100
    # optimal solution of the maxCut problem via simple bruteForce solution
    optimal_config = maxCut.maxcut_bruteforce(graph)

    values_most_sampled = []
    values_max_value = []
    values_optimal_samples = []

    p_values = range(2, 30)

    for p_layers in p_values:
        print("========== p = ", p_layers, " ==========")
        counts = maxCut.simulate_qaoa_maxcut(graph, p_layers, optimal_config, shots)

        # benchmark value of bitstring that was most sampled
        val = benchmarking.qaoa_benchmark(graph, p_layers, shots, optimal_config,
                                          benchmarking.BenchmarkStrategy.MOST_SAMPLED,
                                          counts, plot_config)
        values_most_sampled.append(val)

        # benchmark value of bitstring out of all samples that gives the maximum value
        val = benchmarking.qaoa_run_and_benchmark(graph, p_layers, 50, optimal_config,
                                                  benchmarking.BenchmarkStrategy.MAX_VALUE, plot_config)
        values_max_value.append(val)

        # benchmark: percentage samples of optimal bitstring from all samples
        val = benchmarking.qaoa_benchmark(graph, p_layers, shots, optimal_config,
                                          benchmarking.BenchmarkStrategy.OPTIMAL_SAMPLES,
                                          counts, plot_config)
        values_optimal_samples.append(val)

    print("BM Values most sampled: ", values_most_sampled)
    print("BM Values max value: ", values_max_value)
    print("BM Values optimal samples: ", values_optimal_samples)

    # plot
    plt.plot(p_values, values_most_sampled, label="Value of most sampled bitstring (relative to optimum)")
    plt.plot(p_values, values_max_value, label="Value of best sampled bitstring (50 shots)")
    plt.plot(p_values, values_optimal_samples, label="Percentage of samples of optimal bitstring")
    plt.xlabel("p: Number of layers")
    plt.ylabel("Benchmark result (relative)")
    plt.title("Influence of p for n=" + str(len(graph.nodes())))
    plt.legend()
    if "influence_p_filename" in plot_config:
        plot.savefig("output/" + plot_config["influence_p_filename"], dpi=500)
    plt.show()


def plot_influence_of_problem_size(plot_config, pEqualN=False):
    """
    Plots diagram showing on the samples values when increasing the problem size, i.e. the size of the graphs.
    :param plot_config: configures what should be logged and plotted while running
    :param pEqualN: set this to True if the number of layers should scale linearly with the number of nodes of the graph
    """
    p_layers = 10
    shots = 10000

    values_most_sampled = []
    values_max_value = []
    values_optimal_samples = []

    item_counts = [i for i in range(4, 21) if (3 * i) % 2 == 0]
    print(item_counts)

    for item_count in item_counts:
        if pEqualN:
            p_layers = item_count
        graph = graph_util.generate_random_regular_graph(3, item_count)
        optimal_config = maxCut.maxcut_bruteforce(graph)

        counts = maxCut.simulate_qaoa_maxcut(graph, p_layers, optimal_config, shots)

        val = benchmarking.qaoa_benchmark(graph, p_layers, shots, optimal_config,
                                          benchmarking.BenchmarkStrategy.MOST_SAMPLED,
                                          counts, plot_config)
        values_most_sampled.append(val)
        val = benchmarking.qaoa_run_and_benchmark(graph, p_layers, 100, optimal_config,
                                                  benchmarking.BenchmarkStrategy.MAX_VALUE, plot_config)
        values_max_value.append(val)
        val = benchmarking.qaoa_benchmark(graph, p_layers, shots, optimal_config,
                                          benchmarking.BenchmarkStrategy.OPTIMAL_SAMPLES,
                                          counts, plot_config)
        values_optimal_samples.append(val)

    print("BM Values most sampled: ", values_most_sampled)
    print("BM Values max value: ", values_max_value)
    print("BM Values optimal samples: ", values_optimal_samples)

    # plot
    plt.plot(item_counts, values_most_sampled, label="Value of most sampled bitstring (relative to optimum)")
    plt.plot(item_counts, values_max_value, label="Value of best sampled bitstring (50 shots)")
    plt.plot(item_counts, values_optimal_samples, label="Percentage of samples of optimal bitstring")
    if pEqualN:
        plt.title("Benchmark for different problem-sizes with p=n and shots=" + str(shots))
    else:
        plt.title("Benchmark for different problem-sizes with p=" + str(p_layers) + " and shots=" + str(shots))
    plt.xlabel("n: problem size (number of vertices)")
    plt.ylabel("Benchmark result (relative)")
    plt.legend()

    if "influence_problem_size_filename" in plot_config:
        plot.savefig("output/" + plot_config["influence_problem_size_filename"], dpi=500)
    plt.show()


def create_plots_for_small_example():
    """
    creates the plots (sample counts for different p values and influence of p) for a small graph with 4 nodes and
    4 edges
    """
    plot_config = {
        "plot_counts_diagram": True,
        "counts_diagram_filename": "small-graph-counts-p3.png",
        "plot_result_graph": True,
        "result_graph_filename": "small-graph-result-p3.png"
    }
    g = graph_util.example_graphs()[1]
    graph_util.show_and_save_graph(g, "small-graph.png")
    # run for p=3
    benchmarking.qaoa_run_and_benchmark(g, 3, 10000, maxCut.maxcut_bruteforce(g),
                                        benchmarking.BenchmarkStrategy.MOST_SAMPLED, plot_config)

    # run for p=10
    plot_config["counts_diagram_filename"] = "small-graph-counts-p10.png"
    plot_config["result_graph_filename"] = "small-graph-result-p10.png"
    benchmarking.qaoa_run_and_benchmark(g, 10, 10000, maxCut.maxcut_bruteforce(g),
                                        benchmarking.BenchmarkStrategy.MOST_SAMPLED, plot_config)

    plot_config = {
        "plot_counts_diagram": False,
        "plot_result_graph": False,
        "influence_p_filename": "small-graph-influence-p.png"
    }
    plot_influence_of_p(g, plot_config)


def create_plots_for_big_example():
    """
    creates the plots (sample counts for different p values and influence of p) for a big 3-regular graph with 20 nodes
    """

    g = graph_util.example_graphs()[2]
    weighted_edge_list = [(a, b, 1.0) for a, b in g.edge_list()]
    print(g.edge_list())
    print(weighted_edge_list)

    graph_util.show_and_save_graph(g, "3reg_n20_graph.png")
    plot_config = {
        "plot_counts_diagram": True,
        "counts_diagram_filename": "3reg_n20_graph-counts-p5.png",
        "plot_result_graph": True,
        "result_graph_filename": "3reg_n20_graph-result-p5.png"
    }
    benchmarking.qaoa_run_and_benchmark(g, 5, 10000, maxCut.maxcut_bruteforce(g),
                                        benchmarking.BenchmarkStrategy.MOST_SAMPLED, plot_config)

    plot_config["counts_diagram_filename"] = "3reg_n20_graph-counts-p20.png"
    plot_config["result_graph_filename"] = "3reg_n20_graph-result-p20.png"
    benchmarking.qaoa_run_and_benchmark(g, 20, 10000, maxCut.maxcut_bruteforce(g),
                                        benchmarking.BenchmarkStrategy.MOST_SAMPLED, plot_config)

    plot_config = {
        "plot_counts_diagram": False,
        "plot_result_graph": False,
        "influence_p_filename": "3reg_n20_graph-influence-p.png"
    }
    plot_influence_of_p(g, plot_config)


def create_plot_for_very_large_example():
    """
    performs the maxCut qaoa simulation for a big 3-regular graph with 30 nodes without a bruteforce solution
    (that would take too long)
    """
    g = graph_util.generate_random_regular_graph(3, 30)
    optimal_config = {
        "bitstring": "",
        "bitstring_flipped": "",
        "value": 0
    }
    plot_config = {
        "plot_counts_diagram": True,
        "counts_diagram_filename": "verylarge.png",
        "plot_result_graph": True,
        "result_graph_filename": "verylargegraph.png"
    }
    benchmarking.qaoa_run_and_benchmark(g, 10, 10000, optimal_config, benchmarking.BenchmarkStrategy.MOST_SAMPLED,
                                        plot_config)


def create_plots_for_influence_problem_size(pEqualN=False):
    """
    Creates plots for performance benchmarks of the qaoa when increasing the problem size (i.e. the number of nodes
    of the input graph)
    :param pEqualN: set this to True if the number of layers should scale linearly with the number of nodes of the graph
    """
    plot_config = {
        "plot_counts_diagram": False,
        "plot_result_graph": False,
        "influence_problem_size_filename": "influence-problem-size.png"
    }
    if pEqualN:
        plot_config["influence_problem_size_filename"] = "influence-problem-size-p=n.png"
    plot_influence_of_problem_size(plot_config, pEqualN)


def create_plots():
    """
    Creates all plots (used for the presentation)
    """
    create_plots_for_small_example()
    create_plots_for_big_example()
    create_plots_for_influence_problem_size()
    create_plots_for_influence_problem_size(True)


create_plots()
