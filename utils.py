import numpy as np
from tqdm import tqdm
import qiskit
from qiskit.tools.monitor import job_monitor

def IBM_computer(circuit, backend, provider, shots=1024):
    job = qiskit.execute(circuit, backend=backend, shots=shots, optimization_level=3)
    job_monitor(job, interval = 2)
    results = job.result()
    answer = results.get_counts(circuit)

    return answer

def print_state(cq):
    backend = qiskit.Aer.get_backend('statevector_simulator')
    job = qiskit.execute(cq, backend)
    result = job.result()
    print(result.get_statevector())

def print_res(cq, shots=1024):
    backend = qiskit.Aer.get_backend('qasm_simulator')
    results = qiskit.execute(cq, backend=backend, shots=shots).result()
    answer = results.get_counts()
    print(answer)

def get_res(cq, shots=1024):
    backend = qiskit.Aer.get_backend('qasm_simulator')
    results = qiskit.execute(cq, backend=backend, shots=shots).result()
    answer = results.get_counts()

    return answer

def likelihood( mu, var, samples):
    lklh = (1/np.sqrt(2*np.pi*var))*np.exp((np.power(samples - mu,2))/var)
    return np.prod(lklh)

def accuracy(inferences, labels, target=None):
    #import pdb
    #pdb.set_trace()
    if target != None:
        mask_t = labels != target
        mask_dif = inferences != target
        mask = mask_dif == mask_t
        mask = mask != True
        np.append(mask,[False])
        matches = np.ma.array(np.ones(len(inferences+1)),mask=mask)
    else:
        mask = inferences != labels
        matches = np.ma.array(np.ones(len(inferences+1)),mask=mask)

    return matches.sum()/len(inferences) if matches.sum() is not np.ma.masked else 0.0

def ctrl_bin(state, level):

        state_bin = ''
        i = state
        while i//2 != 0:
            if(i>3):
                state_bin = state_bin + str(i%2)
                i = i//2
            else:
                state_bin = state_bin + str(i%2) + str(i//2)
                i = i//2

        #if level > len(state_bin):
        i = level - len(state_bin) - 1

        if state//2 == 0 and level > len(state_bin):
             state_bin = str(state%2)

        for _ in range(level-len(state_bin)):
            state_bin = state_bin + '0'

        return state_bin

def split_batch(X, Y, val, k_index):
    train_data = np.delete(X, range(k_index*val,(k_index+1)*val), axis=0)
    train_target = np.delete(Y, range(k_index*val,(k_index+1)*val))
    val_data = X[k_index*val:(k_index+1)*val,:]
    val_target = Y[k_index*val:(k_index+1)*val]

    return train_data, train_target, val_data, val_target

def batch_shuffle(X,Y):
    shuf = np.array(range(Y.shape[0]))
    np.random.shuffle(shuf)

    return X[shuf], Y[shuf]

def load_peers(l, pos, n=2):
    p = []
    for _ in range(n):
        p = np.append(p,np.random.choice(pos, size= l if pos.shape[0]>l else pos.shape[0] ))
    return p.astype(int)

def inference(dic_measure, target=1, name='QOCC'):
    if name=='QOCC':
        if not '1' in dic_measure:
            dic_measure['1'] = 0
        if not '0' in dic_measure:
            dic_measure['0'] = 0
        if dic_measure['0'] > dic_measure['1']:
            return target
        else:
            return -1
    else:
        if not '0 0' in dic_measure:
            dic_measure['0 0'] = 0
        if not '0 1' in dic_measure:
            dic_measure['0 1'] = 0
        if dic_measure['0 0'] > dic_measure['0 1']:
                return 1
        elif dic_measure['0 0'] <= dic_measure['0 1']:
            return 2