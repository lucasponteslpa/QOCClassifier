import qiskit
from dataset import ProcessData
from qclassifiers import QClassifier
from qiskit.visualization import circuit_drawer
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
from pdb import set_trace
import numpy as np
from tqdm import tqdm
from utils import accuracy, get_res, print_res, split_batch, load_pairs, batch_shuffle, IBM_computer, inference_array, post_selec_sucess
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

def print_acc(data, res_file, val=10, split=10):
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

    res_file.write("SVM accuracy: "+str(np.mean(svm_mean))+'\n')
    res_file.write("DT accuracy: "+str(np.mean(trees_mean))+'\n')
    res_file.write("KNN accuracy: "+str(np.mean(knn_mean))+'\n')
    res_file.write("SGD accuracy: "+str(np.mean(sgd_mean))+'\n')

def train(data_shuffled, target_shuffled, provider_param, res_file, c=1, batch=100, val=10, n_samples=2 , n_pairs=30, name='QOCC'):

    # qiskit.IBMQ.load_account()
    # provider = qiskit.IBMQ.get_provider(hub='ibm-q')
    # backend = provider.get_backend('ibmq_ourense')
    qiskit.IBMQ.load_account()
    provider = qiskit.IBMQ.get_provider(hub='ibm-q-research',group='Adenilton-Silva')
    backend = provider.get_backend(provider_param)
    # provider = qiskit.IBMQ.load_account()
    # backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and not x.configuration().simulator and x.status().operational==True))

    kfold = batch//val
    batch_c = (batch-val)//2
    best_index = np.zeros(n_samples)
    best_acc = np.zeros(kfold)
    val_acc = np.zeros(kfold)
    val_acc_ibm = np.zeros(kfold)
    var_acc_ibm = np.zeros(kfold)

    #data_shuffled, target_shuffled = batch_shuffle(dataexp.norm_b[batch_index,:,:], dataexp.Y_b[batch_index,:])

    # ISSUE: THE SAMPLES OF TRAIN PROBABLY CONTAIN MEMBER OF ANOTHER CLASSE
    for k in range(kfold):
        train_data, target_train, val_data, target_val = split_batch(data_shuffled, target_shuffled, val, k)
        #set_trace()
        if name=='QOCC':
            pairs = load_pairs(n_pairs,np.where(target_train==c)[0])
            pairs = pairs[:n_samples*(len(pairs)//n_samples)].reshape(len(pairs)//n_samples,n_samples)
        else:
            pairs_c1 = load_pairs(n_pairs,np.where(target_train==1)[0],n=1)
            pairs_c2 = load_pairs(pairs_c1.shape[0],np.where(target_train==2)[0],n=1)
            pairs = np.zeros(2*len(pairs_c2))
            pairs[::2] = pairs_c1[:len(pairs_c2)]
            pairs[1::2] = pairs_c2
            pairs = pairs.reshape(len(pairs)//n_samples,n_samples).astype(int)


        with tqdm(total=len(pairs)) as t:
            for index in pairs:
                inferences = np.zeros(batch-val) - 1
                qclass_arr = []
                for i in range(2*batch_c):

                    train_vec = train_data[np.array(index)]
                    target_vec = target_train[np.array(index)] - 1
                    qclass = QClassifier(train_vec, target_vec, train_data[i,:], name=name)
                    qclass.preparation()
                    qclass_arr.append(qclass.circuito)

                dic_measure = get_res(qclass_arr, shots=1024)
                inferences = inference_array(dic_measure,c,name=name)
                #set_trace()
                act_acc = accuracy(inferences, target_train, c)

                if best_acc[k] < act_acc:
                    best_acc[k] = act_acc
                    best_index = index
                acc_postfix = {"best":best_acc[k],"act":act_acc}
                t.set_postfix(acc_postfix)
                t.update()

        #inferences = np.zeros(val) - 1
        #inferences_ibm = np.zeros(val) - 1
        train_vec = train_data[np.array(best_index)]
        target_vec = target_train[np.array(best_index)] - 1
        qclass_arr = []
        for i in range(val_data.shape[0]):
            #backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and not x.configuration().simulator and x.status().operational==True))
            qclass = QClassifier(train_vec, target_vec, val_data[i,:], name=name)

            qclass.preparation()
            qclass_arr.append(qclass.circuito)

        dic_measure = get_res(qclass_arr)
        inferences = inference_array(dic_measure, c, name=name)
        val_acc[k] = accuracy(inferences, target_val, c)
        aux_measure = []
        for _ in range(5):
            dic_measure_ibm = IBM_computer(qclass_arr, backend, provider)
            inferences_ibm = inference_array(dic_measure_ibm, c, name=name)

            aux_measure.append(accuracy(inferences_ibm, target_val, c))
        val_acc_ibm[k] = np.mean(aux_measure)
        var_acc_ibm[k] = np.var(aux_measure)
        #val_acc_ibm[k] = 0
        if name == 'QOCC':
            res_file.write('C'+str(c)+' Fold-'+str(k)+' Accuracy: '+str(val_acc[k])+'\n')
            res_file.write('C'+str(c)+' IBM Fold-'+str(k)+' Accuracy: '+str(val_acc_ibm[k])+'\n')
            res_file.write('\n')
            print('IBM Accuracy: '+str(val_acc_ibm[k]))
            print('Simulation Accuracy: '+str(val_acc[k]))

        else:
            res_file.write('DBQC'+' Fold-'+str(k)+' Accuracy: '+str(val_acc[k])+'\n')
            res_file.write('DBQC'+' IBM Fold-'+str(k)+' Accuracy: '+str(val_acc_ibm[k])+'\n')
            res_file.write('Post Selection Not Suceeds: '+str(post_selec_sucess(dic_measure_ibm)))
            res_file.write('\n')
            print('IBM Accuracy: '+str(val_acc_ibm[k]))
            print('Simulation Accuracy: '+str(val_acc[k]))

    return val_acc, val_acc_ibm, var_acc_ibm, best_acc

def run_classifier(params):
    res_file = open(params['out_file'],'w')
    dataexp = ProcessData(name=params['dataset'], sample_len=params['split']*params['batch'], batch=params['batch'])
    if(params["show_data"]):
        dataexp.show_data( 55, 54, 61, all_data=True)

    if(params["circuit"]=="QOCC"):

        if params["train"]:
            # IMPLEMENT (CROSS?) VALIDATION
            val_acc_mean_c1, val_ibm_mean_c1 = [], []
            val_acc_mean_c2, val_ibm_mean_c2 = [], []
            val_acc_mean_dbqc, val_ibm_mean_dbqc = [], []
            for i in range(dataexp.split):
                data_shuffled, target_shuffled = batch_shuffle(dataexp.norm_b[i,:,:], dataexp.Y_b[i,:])
                print('Training C1')
                val_acc_c1, val_ibm_c1, var_ibm_c1, best_acc_c1 = train(data_shuffled, target_shuffled, params['provider'], res_file, c=1, batch=dataexp.batch, n_samples=params['num_samples'],n_pairs=params['num_pairs'], val=params['val'])
                #val_acc_c1, val_ibm_c1, best_acc_c1 = 0,0,0
                print('Training C2')
                val_acc_c2, val_ibm_c2, var_ibm_c2, best_acc_c2 = train(data_shuffled, target_shuffled, params['provider'], res_file, c=2, batch=dataexp.batch, n_samples=params['num_samples'],n_pairs=params['num_pairs'], val=params['val'])
                #val_acc_c2, val_ibm_c2, best_acc_c2 = 0,0,0
                print('Training DBQC')
                val_acc_dbqc, val_ibm_dbqc, var_ibm_dbqc, best_acc_dbqc = train(data_shuffled, target_shuffled, params['provider'], res_file, c=2, batch=dataexp.batch, n_samples=params['num_samples'],n_pairs=params['num_pairs'],val=params['val'], name='DBQC')
                val_acc_mean_c1 = np.append(val_acc_mean_c1, val_acc_c1)
                val_acc_mean_c2 = np.append(val_acc_mean_c2, val_acc_c2)
                val_acc_mean_dbqc = np.append(val_acc_mean_dbqc, val_acc_dbqc)
                val_ibm_mean_c1 = np.append(val_ibm_mean_c1, val_ibm_c1)
                val_ibm_mean_c2 = np.append(val_ibm_mean_c2, val_ibm_c2)
                val_ibm_mean_dbqc = np.append(val_ibm_mean_dbqc, val_ibm_dbqc)

                res_file.write("Class 1 Best Accuracy: "+str(np.mean(best_acc_c1))+'\n')
                res_file.write("Class 1 Validation Accuracy: "+str(np.mean(val_acc_c1))+'\n')
                res_file.write("Class 1 IBM Accuracy: "+str(np.mean(val_ibm_c1))+'\n')
                res_file.write("Class 1 IBM Mean Variance: "+str(np.mean(var_ibm_c1))+'\n')
                res_file.write("Class 2 Best Accuracy: "+str(np.mean(best_acc_c2))+'\n')
                res_file.write("Class 2 Validation Accuracy: "+str(np.mean(val_acc_c2))+'\n')
                res_file.write("Class 2 IBM Accuracy: "+str(np.mean(val_ibm_c2))+'\n')
                res_file.write("Class 1 IBM Mean Variance: "+str(np.mean(var_ibm_c2))+'\n')
                res_file.write("DBQC Best Accuracy: "+str(np.mean(best_acc_dbqc))+'\n')
                res_file.write("DBQC Validation Accuracy: "+str(np.mean(val_acc_dbqc))+'\n')
                res_file.write("DBQC IBM Accuracy: "+str(np.mean(val_ibm_dbqc))+'\n')
                res_file.write("Class 1 IBM Mean Variance: "+str(np.mean(var_ibm_dbqc))+'\n')
                res_file.write('\n')

            res_file.write("QOCC C1 accuracy: "+str(np.mean(val_acc_mean_c1))+'\n')
            res_file.write("QOCC C2 accuracy: "+str(np.mean(val_acc_mean_c2))+'\n')
            res_file.write("QOCC IBM C1 accuracy: "+str(np.mean(val_ibm_mean_c1))+'\n')
            res_file.write("QOCC IBM C2 accuracy: "+str(np.mean(val_ibm_mean_c2))+'\n')
            res_file.write("DBQC accuracy: "+str(np.mean(val_acc_mean_dbqc))+'\n')
            res_file.write("DBQC IBM accuracy: "+str(np.mean(val_ibm_mean_dbqc))+'\n')
            print_acc(dataexp, res_file, val=params['val'], split=dataexp.split)
            res_file.close()

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

    parser = argparse.ArgumentParser(description='QOCC Experiments')
    parser.add_argument('--circuit', type=str, default='QOCC', help='Define what circuit will be used')
    parser.add_argument('--dataset', type=str, default='iris', help='Choose what dataset will be used')
    parser.add_argument('--show_data', type=bool, default=False, help='Plot the data distribution')
    parser.add_argument('--train', type=bool, default=True, help='Search for the best two samples')
    parser.add_argument('--batch', type=int, default=100, help='The size of batch')
    parser.add_argument('--val', type=int, default=30, help='The size of validation dataset')
    parser.add_argument('--split', type=int, default=1, help='The factor to split de dataset')
    parser.add_argument('--num_samples', type=int, default=2, help='Number os training samples to run in the circuit')
    parser.add_argument('--num_pairs', type=int, default=30, help='Number of pairs of samples')
    parser.add_argument('--out_file', type=str, default='results.txt', help='Define what circuit will be used')
    parser.add_argument('--provider', type=str, default='ibmq_athens', help='Define what circuit will be used')

    params = vars(parser.parse_args())

    print('PARAMS:')
    print(params)
    run_classifier(params)
