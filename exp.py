import qiskit
from dataset import ProcessData
from qclassifiers import DBQClassifier, QOCClassifier
from qiskit.visualization import plot_histogram, circuit_drawer
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
from pdb import set_trace
import numpy as np
from tqdm import tqdm
from utils import accuracy, get_res, print_res, split_batch, load_peers, batch_shuffle, IBM_computer
from sklearn import svm, tree, neighbors, linear_model

def classic_classifier(clf, data, batch_index, val=10):
    kfold = data.X_b.shape[1]//val
    val_acc = np.zeros(kfold)

    data_shuffled, target_shuffled = batch_shuffle(data.X_b[batch_index,:,:], data.Y_b[batch_index,:])
    for k in range(kfold):
        train_data, train_target, val_data, val_target = split_batch(data_shuffled, target_shuffled, val, k)

        clf.fit(train_data,train_target)
        inferences = clf.predict(val_data)
        val_acc[k] = accuracy(inferences, val_target)

    return val_acc

def print_acc(data, val=10, split=10):
    svm_mean, trees_mean, knn_mean, sgd_mean = [], [], [], []
    for i in range(split):
        svm_acc = classic_classifier(svm.SVC(), data, i, val)
        trees_acc = classic_classifier(tree.DecisionTreeClassifier(), data, i, val)
        knn_acc = classic_classifier(neighbors.KNeighborsClassifier(), data, i, val)
        sgd_acc = classic_classifier(linear_model.SGDClassifier(), data, i, val)
        svm_mean = np.append(svm_mean,svm_acc)
        trees_mean = np.append(trees_mean,trees_acc)
        knn_mean = np.append(knn_mean,knn_acc)
        sgd_mean = np.append(sgd_mean,sgd_acc)

    print("SVM accuracy: "+str(np.mean(svm_mean)))
    print("DT accuracy: "+str(np.mean(trees_mean)))
    print("KNN accuracy: "+str(np.mean(knn_mean)))
    print("SGD accuracy: "+str(np.mean(sgd_mean)))

def train(dataexp, batch_index, c=1, batch=100, val=10):

    # qiskit.IBMQ.load_account()
    # provider = qiskit.IBMQ.get_provider(hub='ibm-q')
    # backend = provider.get_backend('ibmq_santiago')
    provider = qiskit.IBMQ.load_account()
    backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and not x.configuration().simulator and x.status().operational==True))

    kfold = batch//val
    batch_c = (batch-val)//2
    best_0 = 0
    best_1 = 0
    best_acc = np.zeros(kfold)
    val_acc = np.zeros(kfold)
    val_acc_ibm = np.zeros(kfold)

    data_shuffled, target_shuffled = batch_shuffle(dataexp.norm_b[batch_index,:,:], dataexp.Y_b[batch_index,:])

    # ISSUE: THE SAMPLES OF TRAIN PROBABLY CONTAIN MEMBER OF ANOTHER CLASSE
    for k in range(kfold):
        train_data, target_train, val_data, target_val = split_batch(data_shuffled, target_shuffled, val, k)
        peers = load_peers(batch_c,30,(c-1)*batch_c)
        with tqdm(total=peers.shape[0]-1) as t:
            for (x_0, x_1) in zip(peers[:-1], peers[1:]):
                inferences = np.zeros(batch-val) - 1
                inferences = np.zeros(batch-val) - 1
                for i in range(2*batch_c):
                    qclass = QOCClassifier(train_data[x_0,:], train_data[x_1,:], train_data[i,:])
                    qclass.run_classification()
                    dic_measure = get_res(qclass.circuito)
                    if not '1' in dic_measure:
                        dic_measure['1'] = 0
                    if not '0' in dic_measure:
                        dic_measure['0'] = 0
                    if dic_measure['0'] > dic_measure['1']:
                            inferences[i] = target_train[x_0]

                act_acc = accuracy(inferences, target_train, target_train[x_0])
                if best_acc[k] < act_acc:
                    best_acc[k] = act_acc
                    best_0 = x_0
                    best_1 = x_1
                acc_postfix = {"best":best_acc[k],"act":act_acc}
                t.set_postfix(acc_postfix)
                t.update()

        inferences = np.zeros(val) - 1
        inferences_ibm = np.zeros(val) - 1
        for i in range(val_data.shape[0]):
            #set_trace()
            qclass = QOCClassifier(train_data[best_0,:], train_data[best_1,:], val_data[i,:])
            qclass.run_classification()
            dic_measure = get_res(qclass.circuito)
            if not '1' in dic_measure:
                dic_measure['1'] = 0
            if not '0' in dic_measure:
                dic_measure['0'] = 0
            if dic_measure['0'] > dic_measure['1']:
                inferences[i] = target_train[best_0]

            dic_measure_ibm = IBM_computer(qclass.circuito, backend, provider)
            if not '1' in dic_measure_ibm:
                dic_measure['1'] = 0
            if not '0' in dic_measure_ibm:
                dic_measure['0'] = 0
            if dic_measure_ibm['0'] > dic_measure_ibm['1']:
                    inferences_ibm[i] = target_train[x_0]

        val_acc[k] = accuracy(inferences, target_val, target_train[best_0])
        val_acc_ibm[k] = accuracy(inferences_ibm, target_val, target_train[best_0])
    return val_acc, val_acc_ibm, best_acc

def run_classifier(params):
    dataexp = ProcessData(name=params['dataset'], sample_len=params['split']*params['batch'], batch=params['batch'])
    if(params["show_data"]):
        dataexp.show_data( 55, 54, 61, all_data=True)

    if(params["circuit"]=="QOCC"):

        if params["train"]:
            # IMPLEMENT (CROSS?) VALIDATION
            val_acc_mean_c1 = []
            val_acc_mean_c2 = []
            val_ibm_mean_c1 = []
            val_ibm_mean_c2 = []
            for i in range(dataexp.split):
                print('Training C1')
                val_acc_c1, val_ibm_c1, best_acc_c1 = train(dataexp, i, c=1, batch=dataexp.batch, val=params['val'])
                print('Training C2')
                val_acc_c2, val_ibm_c2, best_acc_c2 = train(dataexp, i, c=2, batch=dataexp.batch, val=params['val'])
                val_acc_mean_c1 = np.append(val_acc_mean_c1, val_acc_c1)
                val_acc_mean_c2 = np.append(val_acc_mean_c2, val_acc_c2)
                val_ibm_mean_c1 = np.append(val_ibm_mean_c1, val_ibm_c1)
                val_ibm_mean_c2 = np.append(val_ibm_mean_c2, val_ibm_c2)
                print("Class 1 Best Accuracy: "+str(np.mean(best_acc_c1)))
                print("Class 1 Validation Accuracy: "+str(np.mean(val_acc_c1)))
                print("Class 1 IBM Accuracy: "+str(np.mean(val_ibm_c1)))
                print("Class 2 Best Accuracy: "+str(np.mean(best_acc_c2)))
                print("Class 2 Validation Accuracy: "+str(np.mean(val_acc_c2)))
                print("Class 2 IBM Accuracy: "+str(np.mean(val_ibm_c2)))
                print(" ")

            print("QOCC C1 accuracy: "+str(np.mean(val_acc_mean_c1)))
            print("QOCC C2 accuracy: "+str(np.mean(val_acc_mean_c2)))
            print("QOCC IBM C1 accuracy: "+str(np.mean(val_ibm_mean_c1)))
            print("QOCC IBM C2 accuracy: "+str(np.mean(val_ibm_mean_c2)))
            print_acc(dataexp, val=params['val'], split=dataexp.split)

        else:
            x_0_c0 = 8
            x_1_c0 = 23

            inferences = np.zeros(100) - 1
            for i in range(100):
                if(i != x_0_c0 and i != x_1_c0):
                    qclass = QOCClassifier(dataexp.norm[x_0_c0,:], dataexp.norm[x_1_c0,:], dataexp.norm[i,:])
                    qclass.run_classification()
                    dic_measure = get_res(qclass.circuito)
                    if not '1' in dic_measure:
                        dic_measure['1'] = 0
                    if not '0' in dic_measure:
                        dic_measure['0'] = 0
                    if dic_measure['0'] > dic_measure['1']:
                        inferences[i] = dataexp.Y[x_0_c0]

            act_acc = accuracy(inferences, dataexp.Y, dataexp.Y[x_0_c0])

    else:

        mini = DBQClassifier()
        print_res(mini.circuito)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--circuit', type=str, default='QOCC', help='Define what circuit will be used')
    parser.add_argument('--dataset', type=str, default='iris', help='Choose what dataset will be used')
    parser.add_argument('--show_data', type=bool, default=False, help='Plot the data distribution')
    parser.add_argument('--train', type=bool, default=True, help='Search for the best two samples')
    parser.add_argument('--batch', type=int, default=100, help='The size of batch')
    parser.add_argument('--val', type=int, default=30, help='The size of validation dataset')
    parser.add_argument('--split', type=int, default=1, help='The factor to split de dataset')

    params = vars(parser.parse_args())

    print('PARAMS:')
    print(params)
    run_classifier(params)
