import numpy as np
from tqdm import tqdm
import qiskit

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

def get_res(cq):
    backend = qiskit.Aer.get_backend('qasm_simulator')
    results = qiskit.execute(cq, backend=backend, shots=1024).result()
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

def split_batch(X, Y, v):
    train_data = np.append( X[:(Y.shape[0]-v)//2,:], X[Y.shape[0]//2:Y.shape[0]-v//2,:], axis=0)
    train_target = np.append( Y[:(Y.shape[0]-v)//2], Y[Y.shape[0]//2:Y.shape[0]-v//2])
    val_data = np.append( X[(Y.shape[0]-v)//2:Y.shape[0]//2,:], X[Y.shape[0]-v//2:Y.shape[0],:], axis=0)
    val_target = np.append( Y[(Y.shape[0]-v)//2 : Y.shape[0]//2] ,Y[Y.shape[0]-v//2:Y.shape[0]])

    return train_data, train_target, val_data, val_target

def load_peers(L, l, init):
    return np.random.choice(range(init,init+L), size=l)