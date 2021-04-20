import numpy as np
from tqdm import tqdm
import qiskit
from qiskit.tools.monitor import job_monitor
from qiskit.compiler import assemble, transpile

def IBM_computer(circuit, backend, provider, shots=1024):
    transpiled_circs = transpile(circuit, backend=backend, optimization_level=3)
    qobjs = assemble(transpiled_circs, backend=backend)
    job_info = backend.run(qobjs)

    results = []

    for qcirc_result in transpiled_circs:
        results.append(job_info.result().get_counts(qcirc_result))


    return results

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
    transpiled_circs = transpile(cq, backend=backend, optimization_level=3)
    qobjs = assemble(transpiled_circs, backend=backend,shots=shots)
    job_info = backend.run(qobjs)

    results = [job_info.result().get_counts(qcirc_result) for qcirc_result in transpiled_circs ]

    return results

def likelihood( mu, var, samples):
    lklh = (1/np.sqrt(2*np.pi*var))*np.exp((np.power(samples - mu,2))/var)
    return np.prod(lklh)

def accuracy(inferences, labels, target=None):
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

def load_pairs(l, pos, n=2):
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
        if not '1 0' in dic_measure:
            dic_measure['1 0'] = 0
        if dic_measure['0 0'] > dic_measure['1 0']:
                return 1
        elif dic_measure['0 0'] <= dic_measure['1 0']:
            return 2

def inference_array(dic_measure, target=1, name='QOCC'):
    inf = [inference(dic,target,name) for dic in dic_measure]
    return np.array(inf)

def check_post(dic):
    if not '0 0' in dic:
            dic['0 0'] = 0
    if not '0 1' in dic:
        dic['0 1'] = 0
    if not '1 0' in dic:
            dic['0 0'] = 0
    if not '1 1' in dic:
        dic['1 1'] = 0

    if (dic['0 1'] + dic['1 1']) < (dic['0 0'] + dic['1 0']):
        return 1
    else:
        return 0

def post_selec_sucess(dic_measure):
    post_sucess = np.array([check_post(dic) for dic in dic_measure])
    return post_sucess.sum()
