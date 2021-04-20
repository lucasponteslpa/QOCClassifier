# Quantum One-class Classification With a Distance-based Classifier
[**[Paper]**](https://arxiv.org/abs/2007.16200)

**Abstract:** *The advancement of technology in Quantum Computing has brought possibilities for the execution of algorithms in real quantum devices. As a result, Quantum Machine Learning has grown due to the prospect of solving machine learning problems in quantum machines. However, the existing errors in the current quantum hardware and the low number of available qubits makes it necessary to use solutions that use fewer qubits and fewer operations, mitigating such obstacles. Hadamard Classifier (HC) is a simple distance-based quantum machine learning model for pattern recognition that aims to be minimal. However, HC can still be improved. We present a new classifier based on HC named Quantum One-class Classifier (QOCC) that consists of a minimal quantum machine learning model with fewer operations and qubits, thus being able to mitigate errors from NISQ (Noisy Intermediate-Scale Quantum) computers. Experimental results were obtained by running the proposed classifier on a quantum device provided by IBM Quantum Experience and show that QOCC has advantages over HC.*

## Replicate the experiments of the paper

    ./run.sh

## Python Dependencies

    python3 -m pip install -r requirements.txt

## Run the experiments

    python3 exp.py [-h] [--dataset DATASET] [--batch BATCH] [--val VAL] [--split SPLIT]

### Dataset Options

- `iris`: [Iris Data Set.](https://archive.ics.uci.edu/ml/datasets/iris)(default)
- `skin`: [Skin Segmentation Data Set](https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation)
- `Habermans`: [Haberman's Survival Data Set](https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival)

### Batch Option
Depending of the choosed dataset this parameter is fixed. If is greater than `400` for `Habermans` data set option, ocur an error, because the batch is greater than the data set resampled. For `iris` the batch is fixed to `100`. The default parameters of the batch and the split(next option) is based on the `skin` data set.

- `(int)`: The number of samples in the batch(`100` is the default).

### Split Option
This option delimiter the number of batches, given a data set. If the `iris` is choosed, this parameter is fixed to `1`.

- `(int)`: The number of batches(`10` is the default).

### Validation Option

- `(int)`: The number of samples in the validation data(`30` is the default).
