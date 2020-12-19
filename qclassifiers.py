import qiskit
import numpy as np
from qiskit.quantum_info.operators import Operator
from utils import ctrl_bin
from state_preparation import initializer

class DBQClassifier():

    def __init__(self):
        q = qiskit.QuantumRegister(4)
        c = qiskit.ClassicalRegister(2)
        self.circuito = qiskit.QuantumCircuit(q,c)
        # A
        self.circuito.h(q[0])
        self.circuito.h(q[1])
        
        # B
        gRy = qiskit.circuit.library.RYGate(4.304).control(num_ctrl_qubits=1)
        self.circuito.append(gRy, [q[0],q[2]])
        self.circuito.x(q[0])

        # C
        gToffoli = qiskit.circuit.library.XGate().control(num_ctrl_qubits=2)
        self.circuito.append(gToffoli, [q[0],q[1],q[2]])
        self.circuito.x(q[1])

        # D
        gRy = qiskit.circuit.library.RYGate(1.325).control(num_ctrl_qubits=2)
        self.circuito.append(gRy, [q[0],q[1],q[2]])

        # E
        self.circuito.swap(q[2],q[3])
        self.circuito.cnot(1,2)

        # Classifying
        self.circuito.h(0)
        self.circuito.measure(q[0],c[1])
        self.circuito.measure(q[2],c[0])

class QOCClassifier():

    def __init__(self, sample_0, sample_1, sample_test):
        self.x0 = sample_0
        self.x1 = sample_1
        self.xt = sample_test

        self.q = qiskit.QuantumRegister(3)
        self.c = qiskit.ClassicalRegister(1)
        self.circuito = qiskit.QuantumCircuit(self.q,self.c)

    def run_classification(self):
        # A
        self.circuito.h(self.q[0])
        self.circuito.h(self.q[1])
        
        # B
        gRy_alpha = initializer(self.xt, 'x~', ctrl_str='1')
        self.circuito.append(gRy_alpha, [self.q[0],self.q[2]])
        self.circuito.x(self.q[0])

        # C
        gRy_beta = initializer(self.x0, 'x0', ctrl_str='11')
        self.circuito.append(gRy_beta, [self.q[0],self.q[1],self.q[2]])
        self.circuito.x(self.q[1])

        # D
        gRy_gamma = initializer(self.x1, 'x1', ctrl_str='11')
        self.circuito.append(gRy_gamma, [self.q[0],self.q[1],self.q[2]])

        # Classifying
        self.circuito.h(0)
        self.circuito.measure(self.q[0],self.c[0])