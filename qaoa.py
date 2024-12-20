from qiskit import QuantumCircuit
from rustworkx.rustworkx import PyGraph
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt


def qaoa_circuit_maxcut_linear_schedule(graph: PyGraph, p_layers):
    """
    Creates a quantum circuit for qaoa applied to the maxCut problem with given graph as problem input
    :param graph: the graph where maxcut is applied
    :param p_layers: number of qaoa layers
    :return: quantum circuit
    """

    # number of qubits equals number of vertices of graph
    num_qubits = len(graph.nodes())

    # quantum circuit
    qc = QuantumCircuit(num_qubits)

    # apply hadamard gate to all qubits for superposition
    for i in range(num_qubits):
        qc.h(i)

    # linear schedule of circuit parameters gamma, beta
    if p_layers == 1:
        gammas = [1]
    else:
        gammas = [round(i / (p_layers - 1), 2) for i in range(p_layers)]

    betas = gammas.copy()
    betas.reverse()

    # p repetitions of cost and mixer layer
    for layer in range(p_layers):
        # apply cost layer
        for e in graph.edge_list():

            # (equivalent to qc.rzz(gammas[layer], e[0], e[1]))
            qc.cx(e[0], e[1])
            qc.rz(-gammas[layer], e[1])
            qc.cx(e[0], e[1])

        # apply mixer layer
        for i in range(num_qubits):
            qc.rx(2 * betas[layer], i)

    return qc


def draw_qaoa_maxcut_circuit(graph : PyGraph, p_layers, style):
    """
    Draws the qaoa maxCut circuit and saves the image
    :param graph: the problem graph
    :param p_layers: number of qaoa layers
    :param style: style of the circuit (can specify colors of gates etc.)
    """
    n = len(graph.nodes())
    qc = qaoa_circuit_maxcut_linear_schedule(graph, p_layers)
    # measure qubits
    qc.measure_all()
    # draw circuit
    circuit_drawer(qc, output='mpl', style=style)
    plt.savefig("maxcut-qaoa-circuit-n" + str(n) + "-p" + str(p_layers) + ".png", dpi=500)
    plt.show()
