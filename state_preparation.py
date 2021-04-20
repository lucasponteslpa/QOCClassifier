import qiskit
import numpy as np
from utils import ctrl_bin

def initializer(vetor, label="qV", ctrl_str=None):

    circuit = qiskit.QuantumCircuit(int(np.log2(len(vetor))))

    norms = lambda v: np.sqrt(np.absolute(v[0::2])**2 + np.absolute(v[1::2])**2)
    select_alpha = lambda v,p,i: 2*np.arcsin(v[2*i + 1]/p) if v[2*i]>0 else 2*np.pi - 2*np.arcsin(v[2*i + 1]/p)

    alphas = []
    parents = norms(vetor)
    alphas = np.append(alphas, np.array([ select_alpha(vetor,parents,i) for i in range(vetor.shape[0]//2)]))[::-1]

    for _ in range(int(np.log2(len(vetor)))-1):
        new_parents = norms(parents)
        alphas = np.append(alphas, np.array([ select_alpha(parents,new_parents,i) for i in range(parents.shape[0]//2)]))[::-1]
        parents = new_parents

    level = 1
    gate_op = qiskit.circuit.library.RYGate(alphas[-1])
    circuit.append(gate_op, [int(np.log2(len(vetor)))-1])
    qlines = range(int(np.log2(len(vetor))))[::-1]
    ctrl_state = 0

    for i in range(len(vetor)-2):
        gate_op = qiskit.circuit.library.RYGate(alphas[len(alphas)-2-i]).control(num_ctrl_qubits=level,ctrl_state=ctrl_bin(ctrl_state,level))
        circuit.append(gate_op, qlines[0:level+1])

        if ctrl_state == (2**level - 1):
            ctrl_state = 0
            level += 1
        else:
            ctrl_state +=1
    qvetor = circuit.to_gate(label=label).control(num_ctrl_qubits=len(ctrl_str),ctrl_state=ctrl_str)
    qvetor.name = label

    return qvetor