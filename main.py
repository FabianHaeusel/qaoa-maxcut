import matplotlib.pyplot as plt
import numpy as np
from qaoa import *
from knapsack import optimize_knapsack_bruteforce, generate_random_problem, use_tutorial_problem
from matplotlib import pyplot as plot, scale


# calculate the Ising Hamiltonian of the given Q matrix (of the QUBO formulation)
# from: https://pennylane.ai/qml/demos/tutorial_QUBO/
def calculate_hamiltonian_tut(Q, offset):
    n = len(Q)
    h = {}
    J = np.zeros((n, n))

    for i in range(n): h[i] = 0

    for i in range(n):
        h[i] -= Q[i, i] / 2
        offset += Q[i, i] / 2

        for j in range(i + 1, n):
            J[i, j] += Q[i, j] / 4
            h[i] -= Q[i, j] / 4
            h[j] -= Q[i, j] / 4
            offset += Q[i, j] / 4

    h_rounded = [round(h[i]) for i in h]
    J_rounded = np.rint(J)
    print("=== Hamiltonian from tut ====")
    print("vector h ", h_rounded)
    print("matrix J: \n", J_rounded)
    print("offset: ", offset)
    print("")


# try with specific parameters
# best_config = optimize_knapsack_bruteforce()
# qaoa_benchmark(10000, 2, 10000, best_config, "most_sampled")
# benchmark different parameters
# optimize_qaoa("optimal_samples", 10000) # most_sampled / max_value / optimal_samples

# plots



def plot_influence_of_p():
    penalty_lambda = 1  # TODO: choose good value of lambda
    shots = 10000
    optimal_config = optimize_knapsack_bruteforce()

    values_most_sampled = []
    values_max_value = []
    values_optimal_samples = []
    valid_percentages = []

    p_values = range(2, 30)

    for p_layers in p_values:
        counts = simulate_qaoa_knapsack(penalty_lambda, p_layers, shots)

        val, vp = qaoa_benchmark(penalty_lambda, p_layers, shots, optimal_config, BenchmarkStrategy.MOST_SAMPLED,
                                 counts)
        values_most_sampled.append(val)
        val, vp = qaoa_run_and_benchmark(penalty_lambda, p_layers, 100, optimal_config, BenchmarkStrategy.MAX_VALUE)
        values_max_value.append(val)
        val, vp = qaoa_benchmark(penalty_lambda, p_layers, shots, optimal_config, BenchmarkStrategy.OPTIMAL_SAMPLES,
                                 counts)
        values_optimal_samples.append(val / 40)

        valid_percentages.append(vp)

    print("BM Values most sampled: ", values_most_sampled)
    print("BM Values max value: ", values_max_value)
    print("BM Values optimal samples (normalized by 40): ", values_optimal_samples)
    print("Valid percentages: ", valid_percentages)

    # plot
    plt.plot(p_values, values_most_sampled, label="Most sampled bitstring")
    plt.plot(p_values, values_max_value, label="Bitstring with max value")
    plt.plot(p_values, values_optimal_samples, label="Number of samples of optimal bitstring")
    plt.plot(p_values, valid_percentages, label="Percentage of valid sample results")
    plt.xlabel("p: Number of layers")
    plt.ylabel("Benchmark result (relative)")
    plt.legend()
    plt.savefig("influence-p")
    plt.show()


def plot_influence_of_lambda():
    p_layers = 10  # p_layers is number of qubits
    shots = 10000
    optimal_config = optimize_knapsack_bruteforce()

    values_most_sampled = []
    values_max_value = []
    values_optimal_samples = []
    valid_percentages = []

    lambda_values = [-10, 0, 1, 2, 3, 5, 10, 15, 20, 50, 100, 200, 500, 800, 1000, 5000, 10000, 20000, 50000, 100000]

    for penalty_lambda in lambda_values:
        counts = simulate_qaoa_knapsack(penalty_lambda, p_layers, shots)

        val, vp = qaoa_benchmark(penalty_lambda, p_layers, shots, optimal_config, BenchmarkStrategy.MOST_SAMPLED,
                                 counts)
        values_most_sampled.append(val)
        val, vp = qaoa_run_and_benchmark(penalty_lambda, p_layers, 100, optimal_config, BenchmarkStrategy.MAX_VALUE)
        values_max_value.append(val)
        val, vp = qaoa_benchmark(penalty_lambda, p_layers, shots, optimal_config, BenchmarkStrategy.OPTIMAL_SAMPLES,
                                 counts)
        values_optimal_samples.append(val / 40)

        valid_percentages.append(vp)

    print("BM Values most sampled: ", values_most_sampled)
    print("BM Values max value: ", values_max_value)
    print("BM Values optimal samples (normalized by 40): ", values_optimal_samples)
    print("Valid percentages: ", valid_percentages)

    # plot
    plt.plot(lambda_values, values_most_sampled, label="Most sampled bitstring")
    plt.plot(lambda_values, values_max_value, label="Bitstring with max value")
    plt.plot(lambda_values, values_optimal_samples, label="Number of samples of optimal bitstring")
    plt.plot(lambda_values, valid_percentages, label="Percentage of valid sample results")
    plt.xlabel("penalty lambda (logarithmic scale)")
    plt.ylabel("Benchmark result (relative)")
    plt.xscale("log")
    plt.legend()
    plot.savefig("influence-lambda")
    plt.show()

def plot_influence_of_problem_size():
    p_layers = 10  # p_layers is number of qubits
    penalty_lambda = 2
    shots = 10000

    values_most_sampled = []
    values_max_value = []
    values_optimal_samples = []
    valid_percentages = []

    item_counts = [5, 8, 10, 12, 13]

    for item_count in item_counts:
        items_values, items_weights, max_cap = generate_random_problem(item_count)
        print("items_values:", items_values)
        print("items_weights:", items_weights)
        print("max_cap:", max_cap)
        optimal_config = optimize_knapsack_bruteforce()

        counts = simulate_qaoa_knapsack(penalty_lambda, p_layers, shots)

        val, vp = qaoa_benchmark(penalty_lambda, p_layers, shots, optimal_config, BenchmarkStrategy.MOST_SAMPLED,
                                 counts)
        values_most_sampled.append(val)
        val, vp = qaoa_run_and_benchmark(penalty_lambda, p_layers, 100, optimal_config, BenchmarkStrategy.MAX_VALUE)
        values_max_value.append(val)
        val, vp = qaoa_benchmark(penalty_lambda, p_layers, shots, optimal_config, BenchmarkStrategy.OPTIMAL_SAMPLES,
                                 counts)
        values_optimal_samples.append(val / 40)

        valid_percentages.append(vp)

    print("BM Values most sampled: ", values_most_sampled)
    print("BM Values max value: ", values_max_value)
    print("BM Values optimal samples (normalized by 40): ", values_optimal_samples)
    print("Valid percentages: ", valid_percentages)

    # plot
    plt.plot(item_counts, values_most_sampled, label="Most sampled bitstring")
    plt.plot(item_counts, values_max_value, label="Bitstring with max value")
    plt.plot(item_counts, values_optimal_samples, label="Number of samples of optimal bitstring")
    plt.plot(item_counts, valid_percentages, label="Percentage of valid sample results")
    plt.xlabel("problem size (number of items)")
    plt.ylabel("Benchmark result (relative)")
    plt.legend()
    plot.savefig("influence-problem-size")
    plt.show()

def plots():
    plot_influence_of_p()
    plot_influence_of_lambda()
    plot_influence_of_problem_size()
# plot_influence_of_lambda()
# items_values, items_weights, max_cap = generate_random_problem(13)
# print("items_values:", items_values)
# print("items_weights:", items_weights)
# print("max_cap:", max_cap)

# optimize_knapsack_bruteforce()
# plot_influence_of_p()

plots()
#
# use_tutorial_problem()
# # plots()
# print("Number of items: ", len(Knapsack.items_values))
# h = get_cost_hamiltonian(2)
# print(h)
#
# qaoa_knapsack_expectation_value(2, 10)
# # expectation value: 136.50289123153007+2.842170943040401e-14j
# # eigenvalue of cost hamiltonian: -4140+0j
