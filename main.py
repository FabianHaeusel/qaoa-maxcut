import numpy as np
from qaoa import qaoa_circuit_knapsack

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


# optimize_knapsack_bruteforce(items_values, items_weights, 15)
# Q, offset = calculate_Q_matrix(items_values_tut, items_weights_tut, 26)
# calculate_hamiltonian_tut(Q, offset)
# calculate_hamiltonian_vector_b(Q, offset)

qaoa_circuit_knapsack(items_values, items_weights, max_cap)