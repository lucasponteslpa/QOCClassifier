import qiskit
from dataset import ProcessData
from qclassifiers import DBQClassifier, QOCClassifier
from qiskit.visualization import plot_histogram, circuit_drawer
from pdb import set_trace
import numpy as np
from tqdm import tqdm
from utils import accuracy, get_res, print_res
from sklearn import svm, tree, neighbors, linear_model

def SGD_classifier(data):
    clf = linear_model.SGDClassifier()
    clf.fit(data.X,data.Y)
    inferences = clf.predict(data.X)
    return accuracy(inferences, data.Y)

def KNN_classifier(data):
    clf = neighbors.KNeighborsClassifier()
    clf.fit(data.X,data.Y)
    inferences = clf.predict(data.X)
    return accuracy(inferences, data.Y)

def DTrees_classifier(data):
    clf = tree.DecisionTreeClassifier()
    clf.fit(data.X,data.Y)
    inferences = clf.predict(data.X)
    return accuracy(inferences, data.Y)

def SVM_classfier(data):
    clf = svm.SVC()
    clf.fit(data.X,data.Y)
    inferences = clf.predict(data.X)
    return accuracy(inferences, data.Y)

def train(dataexp):
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
                        dic_measure = get_res(qclass.circuito)
                        if not '1' in dic_measure:
                            dic_measure['1'] = 0
                        if not '0' in dic_measure:
                            dic_measure['0'] = 0
                        if dic_measure['0'] > dic_measure['1']:
                            inferences[i] = dataexp.Y[x_0_c0]

                act_acc = accuracy(inferences, dataexp.Y[:100], dataexp.Y[x_0_c0])
                if best_acc < act_acc:
                    best_acc = act_acc
                    best_0 = x_0_c0
                    best_1 = x_1_c0
                acc_postfix = {"best":best_acc,"act":act_acc}
                t.set_postfix(acc_postfix)
                t.update()

    return best_acc, best_0, best_1

def run_classifier(params):
    dataexp = ProcessData(name=params['dataset'])
    if(params["show_data"]):
        dataexp.show_data( 55, 54, 61, all_data=True)

    if(params["circuit"]=="QOCC"):

        if params["train"]:
            best_acc, x_0_c0, x_1_c0 = train(dataexp)
            print("Best Accuracy: "+str(best_acc))
            print("Sample 1 Choosed: "+str(x_0_c0))
            print("Sample 2 Choosed: "+str(x_1_c0))
            print("")

            svm_acc = SVM_classfier(dataexp)
            trees_acc = DTrees_classifier(dataexp)
            knn_acc = KNN_classifier(dataexp)
            sgd_acc = SGD_classifier(dataexp)

            print("QOCC accuracy: "+str(best_acc))
            print("SVM accuracy: "+str(svm_acc))
            print("DT accuracy: "+str(trees_acc))
            print("KNN accuracy: "+str(knn_acc))
            print("SGD accuracy: "+str(sgd_acc))
            print("")

        else:
            # skin C1 0 18
            # iris C1 8 23

            x_0_c0 = 8
            x_1_c0 = 23
            #x_0_c1 = 2
            #x_1_c1 = 34

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
            svm_acc = SVM_classfier(dataexp)
            trees_acc = DTrees_classifier(dataexp)
            knn_acc = KNN_classifier(dataexp)
            sgd_acc = SGD_classifier(dataexp)

            print("QOCC accuracy: "+str(act_acc))
            print("SVM accuracy: "+str(svm_acc))
            print("DT accuracy: "+str(trees_acc))
            print("KNN accuracy: "+str(knn_acc))
            print("SGD accuracy: "+str(sgd_acc))

    else:

        mini = DBQClassifier()
        print_res(mini.circuito)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--circuit', type=str, help='Define what circuit will be used')
    parser.add_argument('--dataset', type=str, default='iris', help='Choose what dataset will be used')
    parser.add_argument('--show_data', type=bool, default=False, help='Plot the data distribution')
    parser.add_argument('--train', type=bool, default=False, help='Search for the best two samples')
    parser.add_argument('--test', type=int, default=0, help='The test sample')

    params = vars(parser.parse_args())

    print('PARAMS:')
    print(params)
    run_classifier(params)
