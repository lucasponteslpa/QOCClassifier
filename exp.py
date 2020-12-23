import qiskit
from dataset import ProcessData
from qclassifiers import DBQClassifier, QOCClassifier
from qiskit.visualization import plot_histogram, circuit_drawer
from pdb import set_trace
import numpy as np
from tqdm import tqdm
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
    print(answer)

    return answer

def run_classifier(params):
    dataskin = ProcessData(name=params['dataset'])
    dataexp = ProcessData()
    if(params["show_data"]):
        #dataexp.show_data( 55, 54, 61, all_data=True)
        dataskin.show_data( 55, 54, 61, all_data=True)
    if(params["circuit"]=="QOCC"):
        #qclass = QOCClassifier(dataexp.norm[33,:], dataexp.norm[2,:], dataexp.norm[51,:])
        #qclass = QOCClassifier(dataexp.norm[55,:], dataexp.norm[54,:], dataexp.norm[i,:])
        x_0_c0 = 8
        x_1_c0 = 23
        x_0_c1 = 2
        x_1_c1 = 34
        if params["train"]:
            best_0 = 0
            best_1 = 0
            best_acc = 0
            with tqdm(total=1225) as t:
                for x_0_c0 in range(49):
                    for x_1_c0 in range(x_0_c0+1,50):
                        inferences = np.zeros(100) - 1
                        for i in range(100):
                            if(i != x_0_c0 and i != x_1_c0):
                                qclass = QOCClassifier(dataexp.norm[x_0_c0,:], dataexp.norm[x_1_c0,:], dataexp.norm[i,:])
                                qclass.run_classification()
                                dic_measure = print_res(qclass.circuito)
                                if not '1' in dic_measure:
                                    dic_measure['1'] = 0
                                if not '0' in dic_measure:
                                    dic_measure['0'] = 0
                                if dic_measure['0'] > dic_measure['1']:
                                    inferences[i] = dataexp.Y[x_0_c0]

                        act_acc = accuracy(inferences, dataexp.Y[:100],dataexp.Y[x_0_c0])
                        if best_acc < act_acc:
                            best_acc = act_acc
                            best_0 = x_0_c0
                            best_1 = x_1_c0
                        acc_postfix = {"best":best_acc,"act":act_acc}
                        t.set_postfix(acc_postfix)
                        t.update()
            print(best_acc)
            print(best_0)
            print(best_1)
        else:
            qclass = QOCClassifier(dataexp.norm[x_0_c0,:], dataexp.norm[x_1_c0,:], dataexp.norm[params['test'],:])
            qclass.run_classification()
            dic_measure = print_res(qclass.circuito)
            print(dic_measure)

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
    parser.add_argument('--dataset', type=str, default='iris', help='Choose what dataset will be used')
    parser.add_argument('--show_data', type=bool, default=False, help='Plot the data distribution')
    parser.add_argument('--train', type=bool, default=False, help='Search for the best two samples')
    parser.add_argument('--test', type=int, default=0, help='The test sample')
    parser.add_argument('--draw', type=bool, default=False, help='Write a tex file with the circuit scheme')

    params = vars(parser.parse_args())

    print('PARAMS:')
    print(params)
    run_classifier(params)
