import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
import math
from qiskit.quantum_info import Pauli
from qiskit.primitives import StatevectorEstimator
from qiskit.result import counts
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
# from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Session
from qiskit.visualization import plot_histogram, plot_state_city
from knapsack import get_bitstring_knapsack_score, optimize_knapsack_bruteforce, items_values, items_weights, max_cap, is_bitstring_valid

# calculate cost function (in QUBO representation) matrix Q
# by calculation we get: Q_ij = 2 * lambda * w_i * w_j
# and: Q_ii = -v_i + lambda * w_i * (w_i - 2 * max_cap)
def calculate_Q_matrix(penalty_lambda):
    n = len(items_values)

    # add slack variables
    slack_variable_count = round(np.ceil(np.log2(max_cap)))
    values = list(items_values.values()) + [0 for _ in range(slack_variable_count)]
    weights = list(items_weights.values()) + [2**k for k in range(slack_variable_count)]
    # number of qubits (number of items + number of slack variables)
    N = n + slack_variable_count

    Q = np.empty(shape=(N,N))

    for i in range(N):
        Q[i,i] = -values[i] + penalty_lambda * weights[i] * (weights[i] - 2 * max_cap)
        for j in range(i + 1, N):
            Q[i, j]  = 2 * penalty_lambda * weights[i] * weights[j]

    offset = penalty_lambda * max_cap**2

    return Q, offset

def calculate_hamiltonian_vector_b(Q):
    b = {}
    n = len(Q)
    for i in range(n):
        b[i] = -sum([Q[i,j] + Q[j,i] for j in range(n)])

    return b


# creates a quantum circuit for qaoa applied to the knapsack problem with given knapsack problem input
# p_layers must be >= 2
def qaoa_circuit_knapsack_linear_schedule(penalty_lambda, p_layers, shots=10000):

    if len(items_values) != len(items_weights):
        raise ValueError('invalid input values or weights')

    Q, offset = calculate_Q_matrix(penalty_lambda)
    b = calculate_hamiltonian_vector_b(Q)

    num_slack_variables = math.ceil(math.log2(max_cap))
    # number of qubits equals number of items + slack variables
    num_items = len(items_values)
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
    # print(pairs)

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

    if len(items_values) != len(items_weights):
        raise ValueError('invalid input values or weights')

    Q, offset = calculate_Q_matrix(penalty_lambda)
    b = calculate_hamiltonian_vector_b(Q)

    num_slack_variables = math.ceil(math.log2(max_cap))
    # number of qubits equals number of items + slack variables
    num_items = len(items_values)
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
def qaoa_benchmark(penalty_lambda, p_layers, shots, optimal_config):

    counts = qaoa_circuit_knapsack_linear_schedule(penalty_lambda, p_layers, shots)

    # get value of configuration from most sampled bitstring
    most_sampled_bitstring = None
    max_sample = -1
    for bitstring in counts:
        if counts[bitstring] > max_sample:
            most_sampled_bitstring = bitstring
            max_sample = counts[bitstring]

    # get number of invalid bitstrings
    valid_count = 0
    for bitstring in counts:
        if is_bitstring_valid(bitstring):
            valid_count += counts[bitstring]

    if valid_count == shots:
        print("???")
        print(counts)

    val = get_bitstring_knapsack_score(most_sampled_bitstring, optimal_config)
    print("lambda=", penalty_lambda, ", p=", p_layers, ": val of best solution: ", val)
    print("number of samples: ", shots, ", valid bitstrings samples: " , valid_count, "(",  valid_count / shots * 100, "%)" )
    return val

# tries different values for the parameters p and lambda and benchmarks the QAOA result
# by getting the value of the most sampled solution from the QAOA
def optimize_qaoa(shots=10000):
    optimal_config = optimize_knapsack_bruteforce()

    max = -1
    max_config = None
    for penalty_lambda in [0.5, 1, 2, 5, 10, 50, 100, 200]:
        for p_layers in [2, 3, 5, 10, 12, 15, 20]:
            val = qaoa_benchmark(penalty_lambda, p_layers, shots, optimal_config)
            print("lambda=", penalty_lambda, "p=", p_layers,", val of best solution: ", val)
            if val > max:
                max_config = {
                    "penalty_lambda": penalty_lambda,
                    "p_layers": p_layers,
                    "val": val
                }
                max = val

    print("Best config: ")
    print(max_config)
    if "val" in max_config:
        print("With val: ", max_config["val"])

    # first experiment gives: lambda=1, p_layers=15 (?)
    # first experiment gives: lambda=0, p_layers=5 (?) (1278 samples of optimum (10K samples total))

# summary
# - TODO: estimation or sampling using simulator to get expectation values/early results => verify circuit
# - either let circuit be "adiabatic" (parameters fixed before) or do actual optimizing