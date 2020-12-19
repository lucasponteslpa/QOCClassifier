import qiskit
from dataset import ProcessData
from qclassifiers import DBQClassifier, QOCClassifier
from qiskit.visualization import plot_histogram, circuit_drawer
from pdb import set_trace
import numpy as np

def print_state(cq):
    backend = qiskit.Aer.get_backend('statevector_simulator')
    job = qiskit.execute(cq, backend)
    result = job.result()
    print(result.get_statevector())

def print_res(cq):
    backend = qiskit.Aer.get_backend('qasm_simulator')
    results = qiskit.execute(cq, backend=backend, shots=1024).result()
    answer = results.get_counts()
    print(answer)

def run_classifier(params):
    dataexp = ProcessData()
    if(params["show_data"]):
        dataexp.show_data(55,54,61)
    if(params["circuit"]=="QOCC"):
        #qclass = QOCClassifier(dataexp.norm[33,:], dataexp.norm[2,:], dataexp.norm[51,:])
        qclass = QOCClassifier(dataexp.norm[55,:], dataexp.norm[54,:], dataexp.norm[58,:])
        qclass.run_classification()

        if(params["draw"]):
            circuit_drawer(qclass.circuito,filename='qclass.tex',output='latex_source')

        print_res(qclass.circuito)
    #set_trace()
    else:

        mini = DBQClassifier()

        if(params["draw"]):
            circuit_drawer(mini.circuito,filename='minimum.tex',output='latex_source')
        print_res(mini.circuito)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--circuit', type=str, help='Define what circuit will be used')
    parser.add_argument('--num_exemples', type=int,default=2, help='Number of training exemples')
    parser.add_argument('--test_index', type=str, help='Index of test data')
    parser.add_argument('--show_data', type=bool, default=False, help='Plot the data distribution')
    parser.add_argument('--draw', type=bool, default=False, help='Write a tex file with the circuit scheme')

    params = vars(parser.parse_args())

    print('PARAMS:')
    print(params)
    run_classifier(params)
