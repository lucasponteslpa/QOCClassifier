import qiskit
from dataset import ProcessData
from qclassifiers import DBQClassifier, QOCClassifier
from qiskit.visualization import plot_histogram, circuit_drawer
from pdb import set_trace
import numpy as np
from utils import accuracy

def print_state(cq):
    backend = qiskit.Aer.get_backend('statevector_simulator')
    job = qiskit.execute(cq, backend)
    result = job.result()
    print(result.get_statevector())

def print_res(cq):
    backend = qiskit.Aer.get_backend('qasm_simulator')
    results = qiskit.execute(cq, backend=backend, shots=1024).result()
    answer = results.get_counts()
    #print(answer)

    return answer

def run_classifier(params):
    dataexp = ProcessData()
    if(params["show_data"]):
        dataexp.show_data(55,54,61)
    if(params["circuit"]=="QOCC"):
        #qclass = QOCClassifier(dataexp.norm[33,:], dataexp.norm[2,:], dataexp.norm[51,:])
        #qclass = QOCClassifier(dataexp.norm[55,:], dataexp.norm[54,:], dataexp.norm[i,:])
        x_0_c0 = 55
        x_1_c0 = 54
        inferences = np.zeros(100) - 1
        for i in range(100):
            if(i != x_0_c0 and i != x_1_c0):
                qclass = QOCClassifier(dataexp.norm[x_0_c0,:], dataexp.norm[x_1_c0,:], dataexp.norm[i,:])
                qclass.run_classification()
                dic_measure = print_res(qclass.circuito)
                
                if dic_measure['0'] > dic_measure['1']:
                    inferences[i] = dataexp.Y[x_0_c0]

        print(accuracy(inferences, dataexp.Y[:100],dataexp.Y[x_0_c0]))         
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
    parser.add_argument('--show_data', type=bool, default=False, help='Plot the data distribution')
    parser.add_argument('--draw', type=bool, default=False, help='Write a tex file with the circuit scheme')

    params = vars(parser.parse_args())

    print('PARAMS:')
    print(params)
    run_classifier(params)
