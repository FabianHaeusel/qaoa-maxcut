import numpy as np
from qiskit.primitives import Estimator
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.quantum_info import Operator, SparsePauliOp
from qaoa import get_cost_hamiltonian
from knapsack import use_tutorial_problem
import sys
from qiskit_algorithms.minimum_eigensolvers import VQE

from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import QAOAAnsatz

# np.set_printoptions(threshold=sys.maxsize)

use_tutorial_problem()
hamiltonian = get_cost_hamiltonian(2)
# print("hamiltonian: ", hamiltonian)


# sparse_pauli = SparsePauliOp.from_operator(operator)

numpy_solver = NumPyMinimumEigensolver()
result = numpy_solver.compute_minimum_eigenvalue(hamiltonian)
print("NumPyEigensolver:")
print("eigenvalue: ", result.eigenvalue)
print("eigenstate: ", result.eigenstate)

# result most of the time: -4140+0j but often different

# verify by using numpy eigenvalue solver
# hamiltonian_matrix = hamiltonian.to_matrix()
# print("Hamiltonian as matrix: ", hamiltonian_matrix)
# eigenvalues = np.linalg.eigh(hamiltonian_matrix)
#
# print("eigenvalues from hamiltonian matrix: ", eigenvalues)


# verify using qiskit's predefined QAOA Ansatz
circuit = QAOAAnsatz(cost_operator=hamiltonian, reps=2)
circuit.measure_all()
print(circuit.parameters)