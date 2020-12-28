import qiskit
from dataset import ProcessData
from qclassifiers import DBQClassifier, QOCClassifier
from qiskit.visualization import plot_histogram, circuit_drawer
from pdb import set_trace
import numpy as np
from tqdm import tqdm
from utils import accuracy, get_res, print_res
from sklearn import svm, tree, neighbors, linear_model

def classic_classifier(clf, data, batch=100, val=10):
    train_data = np.append(data.X_b[:(batch-val)//2,:],data.X_b[batch//2:batch-val//2,:], axis=0)
    train_target = np.append(data.Y_b[:(batch-val)//2],data.Y_b[batch//2:batch-val//2])
    val_data = np.append(data.X_b[(batch-val)//2:batch//2,:],data.X_b[batch-val//2:batch,:], axis=0)
    val_target = np.append(data.Y_b[(batch-val)//2:batch//2],data.Y_b[batch-val//2:batch])

    clf.fit(train_data,train_target)
    inferences = clf.predict(val_data)
    return accuracy(inferences, val_target)

def print_acc(data, q_acc, batch=100, val=10, split=10):
    svm_mean, trees_mean, knn_mean, sgd_mean = 0, 0, 0, 0
    for i in range(split):
        data.update_batch(i)
        svm_acc = classic_classifier(svm.SVC(), data, batch, val)
        trees_acc = classic_classifier(tree.DecisionTreeClassifier(), data, batch, val)
        knn_acc = classic_classifier(neighbors.KNeighborsClassifier(), data, batch, val)
        sgd_acc = classic_classifier(linear_model.SGDClassifier(), data, batch, val)
        svm_mean += svm_acc/10
        trees_mean += trees_acc/10
        knn_mean += knn_acc/10
        sgd_mean += sgd_acc/10


    print("QOCC accuracy: "+str(q_acc))
    print("SVM accuracy: "+str(svm_mean))
    print("DT accuracy: "+str(trees_mean))
    print("KNN accuracy: "+str(knn_mean))
    print("SGD accuracy: "+str(sgd_mean))

def train(dataexp, c=1, batch=100, val=10):
    batch_c = (batch-val)//2
    best_0 = 0
    best_1 = 0
    best_acc = 0

    train_data = np.append(dataexp.norm_b[:batch_c,:],dataexp.norm_b[batch_c+val//2:batch-val//2,:], axis=0)
    val_data = np.append(dataexp.norm_b[batch_c:batch_c+val//2,:],dataexp.norm_b[2*batch_c+val//2:batch,:], axis=0)
    target_train = np.append(dataexp.Y_b[:batch_c],dataexp.Y_b[(batch_c+val//2):(batch-val//2)])
    target_val = np.append(dataexp.Y_b[batch_c:batch_c+val//2],dataexp.Y_b[2*batch_c+val//2:batch])

    with tqdm(total=(batch_c*(batch_c-1)//2)) as t:
        for x_0_c0 in range(batch_c-1):
            for x_1_c0 in range(x_0_c0+1,batch_c):
                inferences = np.zeros(batch-val) - 1
                for i in range(2*batch_c):
                    if(i != x_0_c0 and i != x_1_c0):
                        qclass = QOCClassifier(train_data[x_0_c0,:], train_data[x_1_c0,:], train_data[i,:])
                        qclass.run_classification()
                        dic_measure = get_res(qclass.circuito)
                        if not '1' in dic_measure:
                            dic_measure['1'] = 0
                        if not '0' in dic_measure:
                            dic_measure['0'] = 0
                        if dic_measure['0'] > dic_measure['1']:
                            inferences[i] = target_train[x_0_c0]

                act_acc = accuracy(inferences, target_train, target_train[x_0_c0])
                if best_acc < act_acc:
                    best_acc = act_acc
                    best_0 = x_0_c0
                    best_1 = x_1_c0
                acc_postfix = {"best":best_acc,"act":act_acc}
                t.set_postfix(acc_postfix)
                t.update()

    inferences = np.zeros(val) - 1
    for i in range(val):
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

    val_acc = accuracy(inferences, target_val, target_train[best_0])

    return val_acc, best_acc, best_0, best_1

def run_classifier(params):
    dataexp = ProcessData(name=params['dataset'], sample_len=params['split']*params['batch'], batch=params['batch'])
    if(params["show_data"]):
        dataexp.show_data( 55, 54, 61, all_data=True)

    if(params["circuit"]=="QOCC"):

        if params["train"]:
            # IMPLEMENT (CROSS?) VALIDATION
            val_acc_mean = 0
            for i in range(dataexp.sample_len//dataexp.batch):
                dataexp.update_batch(i)
                val_acc, best_acc, x_0_c0, x_1_c0 = train(dataexp, batch=dataexp.batch, val=params['val'])
                val_acc_mean += val_acc
                print("Best Accuracy: "+str(best_acc))
                print("Validation Accuracy: "+str(val_acc))
                print("Sample 1 Choosed: "+str(x_0_c0))
                print("Sample 2 Choosed: "+str(x_1_c0))

            val_acc_mean /= 10
            print_acc(dataexp, val_acc_mean, batch=dataexp.batch, val=params['val'], split=dataexp.sample_len//dataexp.batch)

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
            print_acc(dataexp, act_acc)

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
    parser.add_argument('--batch', type=int, default=100, help='The size of batch')
    parser.add_argument('--val', type=int, default=10, help='The size of validation dataset')
    parser.add_argument('--split', type=int, default=1, help='The factor to split de dataset')

    params = vars(parser.parse_args())

    print('PARAMS:')
    print(params)
    run_classifier(params)
