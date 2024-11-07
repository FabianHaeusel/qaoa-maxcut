import numpy as np
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import math
from qiskit.quantum_info import Pauli
from qiskit.primitives import StatevectorEstimator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
# from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Session

# calculate cost function (in QUBO representation) matrix Q
# by calculation we get: Q_ij = 2 * lambda * w_i * w_j
# and: Q_ii = -v_i + lambda * w_i * (w_i - 2 * max_cap)
lambd = 2
def calculate_Q_matrix(items_values, items_weights, max_cap):
    n = len(items_values)

    # add slack variables
    slack_variable_count = round(np.ceil(np.log2(max_cap)))
    values = list(items_values.values()) + [0 for _ in range(slack_variable_count)]
    weights = list(items_weights.values()) + [2**k for k in range(slack_variable_count)]
    # number of qubits (number of items + number of slack variables)
    N = n + slack_variable_count

    Q = np.empty(shape=(N,N))

    for i in range(N):
        Q[i,i] = -values[i] + lambd * weights[i] * (weights[i] - 2 * max_cap)
        for j in range(i + 1, N):
            Q[i, j]  = 2 * lambd * weights[i] * weights[j]

    offset = lambd * max_cap**2

    Q_rounded = np.rint(Q)
    print("\n==== QUBO Q matrix from tut ====")
    print(Q_rounded)
    print("offset: ", offset)
    print("")
    return Q_rounded, offset

def calculate_hamiltonian_vector_b(Q):
    b = {}
    n = len(Q)
    for i in range(n):
        b[i] = -sum([Q[i,j] + Q[j,i] for j in range(n)])

    b_rounded = [round(b[i]) for i in range(n)]

    print("\n== Hamiltonian from qiskit formula ==")
    print("vector b", b_rounded)
    print("")
    return b_rounded


# creates a quantum circuit for qaoa applied to the knapsack problem with given knapsack problem input
def qaoa_circuit_knapsack(items_values : dict, items_weights : dict, max_cap):

    if len(items_values) != len(items_weights):
        raise ValueError('invalid input values or weights')

    print("Running qaoa knapsack circuit")

    Q, offset = calculate_Q_matrix(items_values, items_weights, max_cap)
    b = calculate_hamiltonian_vector_b(Q)

    num_slack_variables = math.ceil(math.log2(max_cap))
    # number of qubits equals number of items + slack variables
    num_items = len(items_values)
    num_qubits = num_items + num_slack_variables
    print("Numer of qubits: " + str(num_qubits))

    # quantum circuit
    qc = QuantumCircuit(num_qubits)

    # apply hadamard gate to all qubits
    for i in range(num_qubits):
        qc.h(i)

    # number of layers (repetitions of cost and mixer layer)
    p = 5 # default: 10

    # qaoa as trotterization of quantum adiabatic algorithm
    # => move beta from 1 to 0 and gamma from 0 to 1
    # => skip optimization part in QAOA
    gammas = [round(i / (p-1), 2) for i in range(p)]

    betas = gammas.copy()
    betas.reverse()

    print("gammas: ", gammas)
    print("betas: ", betas)

    # all qubit pairs
    pairs = [(a,b) for i, a in enumerate(range(num_qubits)) for b in range(num_qubits)[i + 1:]]
    print(pairs)

    # p repetitions of cost and mixer layer
    for layer in range(p):
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
    qc.measure_all()

    # draw circuit
    qc.draw(output="mpl")
    plt.savefig("circuit.png")


    # StatevectorEstimator
    # observables = Pauli('Z' * num_qubits)
    # print("observables: ", observables)
    #
    # estimator = StatevectorEstimator()
    #
    # pub = (qc, observables)
    # job = estimator.run([pub])
    # result = job.result()[0]
    #
    # print(result)
    #
    # for idx, pauli in enumerate(observables):
    #     plt.plot(result.data.evs[idx], label=pauli)
    # plt.legend()


    # estimate results (number of qubits greater than device, has only 5 qubits)
    # fake_manila = FakeManilaV2()
    # pm = generate_preset_pass_manager(backend=fake_manila, optimization_level=1)
    # isa_qc = pm.run(qc)
    #
    # options = {"simulator": {"seed_simulator": 42}}
    # sampler = Sampler(mode=fake_manila, options=options)
    #
    # result = sampler.run([isa_qc]).result()
    # print(result)

    # simulate the quantum circuit locally using aer (cannot be installed)
    # aer_sim = AerSimulator()
    # pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    # isa_qc = pm.run(qc)
    # with Session(backend=aer_sim) as session:
    #     sampler = Sampler(mode=session)
    #     result = sampler.run([isa_qc]).result()

# summary
# - qiskit-aer cannot be installed via pip (problem with C compiler, or python 32 bit (shouldnt be))
# - TODO: estimation or sampling using simulator to get expectation values/early results => verify circuit
# - either let circuit be "adiabatic" (parameters fixed before) or do actual optimizing
# - statevectorEstimator: how should observable be set? (Pauli Z_1, ..., Z_n word?)
# - statevectorSampler: only with parameterized circuit?
# - should circuit measured in the end?