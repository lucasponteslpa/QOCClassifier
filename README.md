# Quantum One-class Classification With a Distance-based Classifier
[**[Paper]**](https://arxiv.org/abs/2007.16200)

**Abstract:** *Distance-based Quantum Classifier (DBQC) is a quantum machine learning model for pattern recognition. However, DBQC has a low accuracy on real noisy quantum processors. We present a modification of DBQC named Quantum One-class Classifier (QOCC) to improve accuracy on NISQ (Noisy Intermediate-Scale Quantum) computers. Experimental results were obtained by running the proposed classifier on a computer provided by IBM Quantum Experience and show that QOCC has improved accuracy over DBQC.*

## Python Dependencies

    python3 -m pip install -r requirements.txt

## Run the experiments

    python3 exp.py [-h] [--circuit CIRCUIT] [--dataset DATASET] [--show_data SHOW_DATA] [--train TRAIN] [--batch BATCH] [--val VAL] [--split SPLIT]

### Circuit Options

- `QOCC`: implemented(default).
- `DBQC`: not implemented completly.

### Dataset Options

- `iris`: [Iris Data Set.](https://archive.ics.uci.edu/ml/datasets/iris)(default)
- `skin`: [Skin Segmentation Data Set](https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation)
- `Habermans`: [Haberman's Survival Data Set](https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival)

### Show Data Option

- `True`: Plot the normalized data distributions of choosed data set.
- `False`: Don't plot the data(default).

### Train Option
The training run over each batch and is computed the mean accuracy over these batches.

- `True`: Run the simulation of the circuit to search for the training sample with higher accuracy (default).
- `False`: Execute a test with a given sample(not implemented).

### Batch Option
Depending of the choosed dataset this parameter is fixed. If is greater than `400` for `Habermans` data set option, ocur an error, because the batch is greater than the data set resampled. For `iris` the batch is fixed to `100`. The default parameters of the batch and the split(next option) is based on the `skin` data set.

- `(int)`: The number of samples in the batch(`100` is the default).

### Split Option
This option delimiter the number of batches, given a data set. If the `iris` is choosed, this parameter is fixed to `1`.

- `(int)`: The number of batches(`10` is the default).

### Validation Option

- `(int)`: The number of samples in the validation data(`30` is the default).

