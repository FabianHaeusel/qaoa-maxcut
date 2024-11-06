import numpy as np

# 0-1 knapsack problem, example data
items_values = {
    0: 1,
    1: 6,
    2: 4,
    3: 8,
    4: 3,
}
items_weights = {
    0: 8,
    1: 4,
    2: 6,
    3: 8,
    4: 2
}
max_cap = 15

items_values_tut = {
    "football": 8,
    "laptop": 47,
    "camera": 10,
    "books": 5,
    "guitar": 16
}
items_weights_tut = {
    "football": 3,
    "laptop": 11,
    "camera": 14,
    "books": 19,
    "guitar": 5
}
max_cap_tut = 26

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


# calculate the Ising Hamiltonian of the given Q matrix (of the QUBO formulation)
# from: https://pennylane.ai/qml/demos/tutorial_QUBO/
def calculate_hamiltonian_tut(Q, offset):
    n = len(Q)
    h = {}
    J = np.empty((n,n))

    for i in range(n): h[i] = 0

    for i in range(n):
        h[i] -= Q[i,i] / 2
        offset += Q[i,i] / 2

        for j in range(i + 1, n):
            J[i,j] += Q[i,j] / 4
            h[i] -= Q[i,j] / 4
            h[j] -= Q[i,j] / 4
            offset += Q[i,j] / 4

    h_rounded = [round(h[i]) for i in h]
    J_rounded = np.rint(J)
    print("=== Hamiltonian from tut ====")
    print("vector h ", h_rounded)
    print("matrix J: \n", J_rounded)
    print("offset: ", offset)
    print("")

def calculate_hamiltonian_vector_b(Q, offset):
    b = {}
    n = len(Q)
    for i in range(n):
        b[i] = -sum([Q[i,j] + Q[j,i] for j in range(n)])

    b_rounded = [round(b[i]) for i in range(n)]

    print("\n== Hamiltonian from qiskit formula ==")
    print("vector b", b_rounded)
    print("")


# brute force solution
def optimize_knapsack_bruteforce(items_values, items_weights, max_cap):
    n = len(items_values)
    print("\n==== knapsack bruteforce ====")
    print("items: ", n)
    print("iterations: ", 2**n)

    max_value = -1
    optimal_config = None

    for i in range(2**n):
        binstring = np.binary_repr(i, n)

        items = list(items_values.keys())
        chosen_indices = [j for j, i in enumerate(binstring) if i == "1"]
        chosen_items = [items[i] for i in chosen_indices]
        total_value = sum([items_values[i] for i in chosen_items])
        total_weight = sum([items_weights[i] for i in chosen_items])

        # print(chosen_items)
        # print(total_value)
        # print(total_weight)
        # print("")

        if(total_weight > max_cap):
            continue

        if(total_value > max_value):
            optimal_config = {
                "indices": chosen_indices,
                "items": chosen_items,
                "value": total_value,
                "weight": total_weight
            }

    print("The optimal configuration is: ", optimal_config)
    print("\n")
    return optimal_config


optimize_knapsack_bruteforce(items_values, items_weights, 15)
Q, offset = calculate_Q_matrix(items_values_tut, items_weights_tut, 26)
calculate_hamiltonian_tut(Q, offset)
calculate_hamiltonian_vector_b(Q, offset)
