from knapsack import get_bitstring_knapsack_score, optimize_knapsack_bruteforce, is_bitstring_valid, Knapsack
import numpy as np
import math
from enum import Enum
from qiskit import *
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp, Operator
import matplotlib.pyplot as plt
from qiskit.quantum_info.operators.symplectic import Pauli


# calculate cost function (in QUBO representation) matrix Q
def calculate_Q_matrix(penalty_lambda):
    n = len(Knapsack.items_values)

    # add slack variables
    slack_variable_count = round(np.ceil(np.log2(Knapsack.max_cap)))
    values = list(Knapsack.items_values.values()) + [0 for _ in range(slack_variable_count)]
    weights = list(Knapsack.items_weights.values()) + [2**k for k in range(slack_variable_count)]
    # number of qubits (number of items + number of slack variables)
    N = n + slack_variable_count

    Q = np.zeros(shape=(N,N))

    for i in range(N):
        Q[i,i] = -values[i] + penalty_lambda * weights[i] * (weights[i] - 2 * Knapsack.max_cap)
        for j in range(N):
            if i == j: continue # TODO: change back
            Q[i, j] = 2 * penalty_lambda * weights[i] * weights[j]

    offset = penalty_lambda * Knapsack.max_cap**2

    # dummy hamiltonian
    # Q = np.identity(n=N)

    return Q, offset

def calculate_hamiltonian_vector_b(Q):
    n = Q.shape[0]
    b = np.zeros(shape=(n))
    for i in range(n):
        b[i] = -sum([Q[i,j] + Q[j,i] for j in range(n)])

    # dummy hamiltonian
    # b = np.ones(shape=(Q.shape[0]))

    return b


def create_pauli_z_string(N, j, k):
    """
    creates a pauli string that has only 'I' characters, but at position j and k it has 'Z'
    """
    output = ["I"] * N

    if j >= 0 and j < N: output[j] = "Z"
    if k >= 0 and k < N: output[k] = "Z"

    return "".join(output)
def get_cost_hamiltonian(penalty_lambda):
    """
    returns the cost hamiltonian as a SparsePauliOp
    """
    Q, offset = calculate_Q_matrix(penalty_lambda)
    b = calculate_hamiltonian_vector_b(Q)

    N = Q.shape[0] # number of qubits

    pauli_list = []

    for i in range(N):

        # single Z paulis
        coefficient = b[i]
        pauli_string = create_pauli_z_string(N, N - i - 1, -1)
        pauli_list.append(("-" + pauli_string, coefficient))
        # TODO: is this "-" needed?

        # double Z paulis
        for j in range(N):
            if i == j: continue
            coefficient = Q[i, j]
            # right most character of pauli word is qubit 0 (qiskit's little-endian convention)
            pauli_string = create_pauli_z_string(N, N - i - 1, N - j - 1)
            pauli_list.append((pauli_string, coefficient))


    hamiltonian = SparsePauliOp.from_list(pauli_list)
    # print(hamiltonian)
    # print(hamiltonian.to_matrix())
    return hamiltonian

# creates a quantum circuit for qaoa applied to the knapsack problem with given knapsack problem input
# p_layers must be >= 2
def qaoa_circuit_knapsack_linear_schedule(penalty_lambda, p_layers, shots=10000):

    if len(Knapsack.items_values) != len(Knapsack.items_weights):
        raise ValueError('invalid input values or weights')

    Q, offset = calculate_Q_matrix(penalty_lambda)
    b = calculate_hamiltonian_vector_b(Q)

    num_slack_variables = math.ceil(math.log2(Knapsack.max_cap))
    # number of qubits equals number of items + slack variables
    num_items = len(Knapsack.items_values)
    num_qubits = num_items + num_slack_variables
    # print("Numer of qubits: " + str(num_qubits), "(", num_items, " items, ", num_slack_variables, " slack variables)")

    # quantum circuit
    qc = QuantumCircuit(num_qubits, num_items)

    # apply hadamard gate to all qubits
    for i in range(num_qubits):
        qc.h(i)

    # number of layers (repetitions of cost and mixer layer)
    # p = 10 # default: 10 (=> number of qubits)

    # qaoa as trotterization of quantum adiabatic algorithm (linear schedule)
    # => move beta from 1 to 0 and gamma from 0 to 1
    # => skip optimization part in QAOA
    gammas = [round(i / (p_layers - 1), 2) for i in range(p_layers)]

    betas = gammas.copy()
    betas.reverse()

    # print("gammas: ", gammas)
    # print("betas: ", betas)

    # all qubit pairs
    pairs = [(a,b) for i, a in enumerate(range(num_qubits)) for b in range(num_qubits)[i + 1:]]

    # p repetitions of cost and mixer layer
    for layer in range(p_layers):
        # apply cost layer
        # single z rotations
        for i in range(num_qubits):
            qc.rz(2 * gammas[layer] *  b[i], i)
        # double z rotations
        for i,j in pairs:
            qc.rzz(2 * gammas[layer] * Q[i,j], i, j)

        # apply mixer layer
        for i in range(num_qubits):
            qc.rx(2 * betas[layer], i)


    return qc, num_items

def qaoa_knapsack_expectation_value(penalty_lambda, p_layers):
    """
    Returns the expectation value of the qaoa knapsack quantum circuit
    :param penalty_lambda:
    :param p_layers:
    :return:
    """
    qc, num_items = qaoa_circuit_knapsack_linear_schedule(penalty_lambda, p_layers)

    qc.save_statevector("sv")

    operator = Operator(get_cost_hamiltonian(penalty_lambda))

    aer_sim = AerSimulator()
    circ = transpile(qc, aer_sim)
    result = aer_sim.run(circ, shots=10000).result()
    statevector = result.data()["sv"]
    exv = statevector.expectation_value(oper=operator) # operator is cost hamiltonian

    print("expectation value: ", exv)

    return exv


def simulate_qaoa_knapsack(penalty_lambda, p_layers, shots=10000):
    qc, num_items = qaoa_circuit_knapsack_linear_schedule(penalty_lambda, p_layers, shots)

    # measure qubits (should I?)
    for i in range(num_items):
        qc.measure(i, i)

    # draw circuit
    # qc.draw(output="mpl")
    # plt.savefig("circuit.png")

    # simulate the quantum circuit locally using aer (cannot be installed)
    aer_sim = AerSimulator()
    circ = transpile(qc, aer_sim)
    result = aer_sim.run(circ, shots=shots).result()
    counts = result.get_counts(circ)

    # optionally plot histogram
    # plot_histogram(counts, title="Bell-State counts", filename="histogram")

    return counts


# creates a quantum circuit for qaoa applied to the knapsack problem with given knapsack problem input
def qaoa_circuit_knapsack_parameterized(penalty_lambda, p_layers):

    if len(Knapsack.items_values) != len(Knapsack.items_weights):
        raise ValueError('invalid input values or weights')

    Q, offset = calculate_Q_matrix(penalty_lambda)
    b = calculate_hamiltonian_vector_b(Q)

    num_slack_variables = math.ceil(math.log2(Knapsack.max_cap))
    # number of qubits equals number of items + slack variables
    num_items = len(Knapsack.items_values)
    num_qubits = num_items + num_slack_variables
    # print("Numer of qubits: " + str(num_qubits), "(", num_items, " items, ", num_slack_variables, " slack variables)")

    # parameters
    gamma = Parameter("gamma")
    beta = Parameter("beta")

    # quantum circuit
    qc = QuantumCircuit(num_qubits, num_items)

    # apply hadamard gate to all qubits
    for i in range(num_qubits):
        qc.h(i)

    # number of layers (repetitions of cost and mixer layer)
    # p = 10 # default: 10 (=> number of qubits)

    # print("gammas: ", gammas)
    # print("betas: ", betas)

    # all qubit pairs
    pairs = [(a,b) for i, a in enumerate(range(num_qubits)) for b in range(num_qubits)[i + 1:]]
    # print(pairs)

    # p repetitions of cost and mixer layer
    for layer in range(p_layers):
        # apply cost layer
        # single z rotations
        for i in range(num_qubits):
            qc.rz(2 * gamma *  b[i], i)
        # double z rotations
        for i,j in pairs:
            qc.rzz(2 * gamma * Q[i,j], i, j)

        # apply mixer layer
        for i in range(num_qubits):
            qc.rx(2 * beta, i)


    # measure qubits (should I?)
    for i in range(num_items):
        qc.measure(i, i)

    # draw circuit
    # qc.draw(output="mpl")
    # plt.savefig("circuit.png")

    # simulate the quantum circuit locally using aer (cannot be installed)
    shots = 10000
    aer_sim = AerSimulator()
    # pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    # isa_qc = pm.run(qc)
    circ = transpile(qc, aer_sim)
    # circ.save_statevector()
    result = aer_sim.run(circ, shots=shots).result()
    counts = result.get_counts(circ)


# get benchmark score for given qaoa configuration
def qaoa_benchmark_most_sampled_bitstring(penalty_lambda, p_layers, optimal_config, counts):

    # get value of configuration from most sampled bitstring
    most_sampled_bitstring = None
    max_sample = -1
    for bitstring in counts:
        if counts[bitstring] > max_sample:
            most_sampled_bitstring = bitstring
            max_sample = counts[bitstring]

    val = get_bitstring_knapsack_score(most_sampled_bitstring, optimal_config)
    print("lambda=", penalty_lambda, ", p=", p_layers, ": val of best solution: ", val)
    return val

def qaoa_benchmark_max_value_bitstring(penalty_lambda, p_layers, optimal_config, counts):

    # get bitstring from samples that has the highest value
    max_value = -1
    best_bitstring = None
    for bitstring in counts:
        val = get_bitstring_knapsack_score(bitstring, optimal_config, False)
        if val > max_value:
            max_value = val
            best_bitstring = bitstring

    print("lambda=", penalty_lambda, ", p=", p_layers, ": bitstring sampled with highest value: ", best_bitstring, " val: ", max_value)
    return max_value

def qaoa_benchmark_optimal_samples(optimal_config, counts):

    # get number of samples of optimal solution of problem
    if optimal_config["bitstring"] in counts:
        return counts[optimal_config["bitstring"]]
    return 0

class BenchmarkStrategy(Enum):
    MOST_SAMPLED = 0,
    MAX_VALUE = 1,
    OPTIMAL_SAMPLES = 2,

# returns a value that benchmarks the current qaoa configuration with different strategies
def qaoa_benchmark(penalty_lambda, p_layers, shots, optimal_config, strategy : BenchmarkStrategy, counts):

    # print("counts sampled bitstrings: ", len(counts))
    # get number of invalid bitstrings
    valid_count = 0
    for bitstring in counts:
        if is_bitstring_valid(bitstring):
            valid_count += counts[bitstring]

    if valid_count == shots:
        print("???")
        print(counts)

    val = 0
    if strategy == BenchmarkStrategy.MOST_SAMPLED:
        # the bitstring that is sampled the most -> value of that bitstring
        val = qaoa_benchmark_most_sampled_bitstring(penalty_lambda, p_layers, optimal_config, counts)
    elif strategy == BenchmarkStrategy.MAX_VALUE:
        # the bitstring from the samples that gives the highest problem value -> value of that bitstring
        val = qaoa_benchmark_max_value_bitstring(penalty_lambda, p_layers, optimal_config, counts)
    elif strategy == BenchmarkStrategy.OPTIMAL_SAMPLES:
        # the number of samples of the optimal solution (from bruteforce)
        val = qaoa_benchmark_optimal_samples(optimal_config, counts)

    valid_percentage = valid_count / shots
    print("number of samples: ", shots, ", valid bitstrings samples: " , valid_count, "(", valid_percentage * 100 , "%)" )

    return val, valid_percentage

def qaoa_run_and_benchmark(penalty_lambda, p_layers, shots, optimal_config, strategy : BenchmarkStrategy):
    counts = simulate_qaoa_knapsack(penalty_lambda, p_layers, shots)
    return qaoa_benchmark(penalty_lambda, p_layers, shots, optimal_config, strategy, counts)

# tries different values for the parameters p and lambda and benchmarks the QAOA result
# by getting the value of the most sampled solution from the QAOA
def optimize_qaoa(strategy, shots=10000):
    optimal_config = optimize_knapsack_bruteforce()

    max = -1
    max_config = None
    for penalty_lambda in [0.5, 1, 2, 5, 10, 50, 100, 200, 1000, 10000, 100000]:
        for p_layers in [2, 3, 5, 10, 12, 15, 20, 30]:
            val = qaoa_run_and_benchmark(penalty_lambda, p_layers, shots, optimal_config, strategy)
            print("lambda=", penalty_lambda, "p=", p_layers,", val of best solution: ", val)
            if val > max:
                max_config = {
                    "penalty_lambda": penalty_lambda,
                    "p_layers": p_layers,
                    "val": val
                }
                max = val

    print("\n=== QAOA Benchmark complete ===")
    print("Best config: ")
    print(max_config)
    if "val" in max_config:
        print("With val: ", max_config["val"])

    # first experiment gives: lambda=1, p_layers=15 (?)
    # first experiment gives: lambda=0, p_layers=5 (?) (1278 samples of optimum (10K samples total))

def calculate_hamiltonian_ground_state(h):
    calc = GroundStateEigensolver(mapper, numpy_solver)
    res = calc.solve(es_problem)
    print(res)

# summary
# - TODO: estimation or sampling using simulator to get expectation values/early results => verify circuit
# - either let circuit be "adiabatic" (parameters fixed before) or do actual optimizing
