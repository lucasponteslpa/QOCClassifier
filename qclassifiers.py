import qiskit
import numpy as np
from qiskit.quantum_info.operators import Operator
from utils import ctrl_bin
from state_preparation import initializer

class QClassifier(object):

    def __init__(self, train_vectors, labels, test_vector, name='QOCC'):

        # Load the data and your parameters
        self.name = name
        self.train_vectors_size = train_vectors.shape[0]
        self.num_features = train_vectors.shape[1]
        self.log2M = int(np.log2(self.train_vectors_size))
        self.log2N = int(np.log2(self.num_features))
        self.train_vectors = train_vectors
        self.target = labels
        self.test_vector = test_vector

        # Create the respective registers for each component of
        # the implemented classifier
        self.ancilla = qiskit.QuantumRegister(1)
        self.ket_m = qiskit.QuantumRegister(self.log2M)
        self.ket_phi = qiskit.QuantumRegister(self.log2N)
        self.c_anc = qiskit.ClassicalRegister(1)
        if name == 'DBQC':
            self.ket_y = qiskit.QuantumRegister(1)
            self.c_class = qiskit.ClassicalRegister(1)
            self.circuito = qiskit.QuantumCircuit(self.ancilla,self.ket_m,self.ket_phi,self.ket_y,self.c_anc,self.c_class)
        else:
            self.circuito = qiskit.QuantumCircuit(self.ancilla,self.ket_m,self.ket_phi,self.c_anc)

    def preparation(self):

        # Stage A: put the qubits of ancilla and ket_m register in superposition
        self.circuito.h(self.ancilla)
        for i in range(int(np.log2(self.train_vectors_size))):
            self.circuito.h(self.ket_m[i])

        # Stage B: Preparation of the test state, with controlled in ancilla = 0
        q_test = initializer(self.test_vector, label="x~", ctrl_str='0')
        qb = np.array([self.ancilla])

        if(self.log2N > 1):
            for i in range(self.log2M):
                qb = np.append(qb,self.ket_phi[i])
        else:
            qb = np.append(qb,self.ket_phi)

        self.circuito.append(q_test,list(qb))

        # Stage C: Preparation of the exemples states, with controlled in ancilla = 1
        for i in range(self.train_vectors_size):
            q_m = initializer(self.train_vectors[i,:],label='x_'+str(i), ctrl_str=ctrl_bin(i,self.log2M)+'1')
            qb = np.array([self.ancilla])
            for j in range(self.log2M):
                qb = np.append(qb,self.ket_m[j])

            if(self.log2N > 1):
                for j in range(self.log2M):
                    qb = np.append(qb,[self.ket_phi[j]])
            else:
                qb = np.append(qb,self.ket_phi)
            self.circuito.append(q_m,list(qb))

        # Stage D: Tangle the class states with the index ket_m
        if self.name == 'DBQC':
            for i in range(self.train_vectors_size):
                if self.target[i] == 1:
                    target_gate = qiskit.circuit.library.XGate().control(num_ctrl_qubits=self.log2M, ctrl_state=ctrl_bin(i,self.log2M))
                    qb = np.array([])
                    for j in range(self.log2M):
                        qb = np.append(qb,self.ket_m[j])

                    qb = np.append(qb,self.ket_y)
                    self.circuito.append(target_gate,list(qb))


        self.circuito.h(self.ancilla)
        self.circuito.measure(self.ancilla,self.c_anc)
        if self.name=='DBQC':
            self.circuito.measure(self.ket_y,self.c_class)