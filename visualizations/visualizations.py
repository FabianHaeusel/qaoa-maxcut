from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
from qiskit.circuit import Parameter
import qaoa
import graph_util


def visualize_dummy_qaoa_circuit():
    """
    Saves an image of a dummy qaoa circuit where cost and mixer layer are generalized (used for presentation)
    """
    qc = QuantumCircuit(4, name="Quantum Circuit")

    # hadamard gates
    for i in range(4):
        qc.h(i)

    # cost layer
    qc.barrier([0, 1, 2, 3])
    qc.append(QuantumCircuit(4, name="Cost Layer").to_instruction(), [0, 1, 2, 3])

    # mixer layer
    qc.barrier([0, 1, 2, 3])
    qc.append(QuantumCircuit(4, name="Mixer Layer").to_instruction(), [0, 1, 2, 3])

    qc.measure_all()

    # Draw the circuit
    circuit_drawer(qc, output='mpl', style={'displaycolor': {'Cost Layer': '#28a8ff',
                                                             'Mixer Layer': '#8d51ff',
                                                             "h": "#002da3"}})

    plt.savefig("qaoa-dummy-circuit.png", dpi=500)
    plt.show()


def visualize_maxcut_qaoa_circuit():
    """
    Saves images of different maxCut qaoa circuits (for different input problem graphs)
    """
    style = {"displaycolor": {
        "h": "#002da3",
        "cx": '#28a8ff',
        "rz": '#28a8ff',
        "rx": '#8d51ff'
    }}
    qaoa.draw_qaoa_maxcut_circuit(graph_util.example_graphs()[1], 1, style)
    qaoa.draw_qaoa_maxcut_circuit(graph_util.example_graphs()[1], 2, style)
    qaoa.draw_qaoa_maxcut_circuit(graph_util.example_graphs()[2], 1, style)

def visualize_only_cost_layer():
    """
    Saves image of simplified (isolated) qaoa cost layer (quantum gates)
    """
    qc = QuantumCircuit(2, name="Quantum Circuit")

    gamma = Parameter("\u03B3")

    qc.cx(0, 1)
    qc.rz(- gamma, 1)
    qc.cx(0, 1)

    circuit_drawer(qc, output='mpl', style={'displaycolor': {
        'rx': '#8d51ff',
    }})

    plt.savefig("qaoa-cost-layer.png", dpi=500)
    plt.show()

def visualize_only_mixer_layer():
    """
    Saves image of qaoa mixer layer gates
    """
    qc = QuantumCircuit(2, name="Quantum Circuit")

    p = Parameter("\u03b2")

    for i in range(qc.num_qubits):
        qc.rx(2 * p, i)

    circuit_drawer(qc, output='mpl', style={'displaycolor': {
        'rx': '#8d51ff',
    }})

    plt.savefig("qaoa-mixer-layer.png", dpi=500)
    plt.show()


def create_all_visualizations():
    """
    Creates all visualizations (images in the visualizations folder)
    """
    visualize_dummy_qaoa_circuit()
    visualize_maxcut_qaoa_circuit()
    visualize_only_cost_layer()
    visualize_only_mixer_layer()

create_all_visualizations()
