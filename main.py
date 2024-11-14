import numpy as np
from qaoa import qaoa_benchmark, optimize_qaoa
from knapsack import optimize_knapsack_bruteforce

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

# try with specific parameters
best_config = optimize_knapsack_bruteforce()
qaoa_benchmark(1, 2, 10000, best_config)

# benchmark different parameters
# optimize_qaoa()
