import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from pdb import set_trace
import tensorflow.compat.v1 as tf
import pandas as pd
from utils import likelihood
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

class ProcessData():

    def __init__(self, name='iris', sample_len=1000, batch=100):
        self.sample_len = sample_len
        self.batch = batch
        np.random.seed(132)
        if name == 'skin':
            TRAIN_DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt'

            train_file_path = tf.keras.utils.get_file("segmentation.data.csv", TRAIN_DATA_URL)
            dataset = pd.read_csv(train_file_path).to_numpy()
            data = np.array([ i.split('\t') for i in dataset[:,0]]).astype(np.int32)
            self.X, self.Y = self.resample(data[:,0:3:2],data[:,3])

            c1s = Counter(self.Y)
            n_c1 = c1s[1]

            c1_index = np.random.choice(range(n_c1),size=sample_len//2)
            c2_index = np.random.choice(range(n_c1,self.X.shape[0]),size=sample_len//2)
            samples_index = np.append(c1_index,c2_index)
            self.X = self.X[samples_index]
            self.Y = self.Y[samples_index]

            self.X_b = np.append(self.X[:self.batch//2,:],self.X[self.sample_len//2:(self.sample_len+self.batch)//2, :], axis=0)
            self.Y_b = np.append(self.Y[:self.batch//2],self.Y[self.sample_len//2:(self.sample_len+self.batch)//2])

        elif name == 'Habermans':
            TRAIN_DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data'

            train_file_path = tf.keras.utils.get_file("haberman.data.csv", TRAIN_DATA_URL)
            self.dataset = pd.read_csv(train_file_path).to_numpy()
            self.X, self.Y = self.resample(self.dataset[:,0:3:2],self.dataset[:,3])

            classes = Counter(self.Y)

            self.sample_len = classes[1]+classes[2]
            self.batch = self.sample_len//(sample_len//batch)
            self.X_b = np.append(self.X[:self.batch//2,:],self.X[self.sample_len//2:(self.sample_len+self.batch)//2, :], axis=0)
            self.Y_b = np.append(self.Y[:self.batch//2],self.Y[self.sample_len//2:(self.sample_len+self.batch)//2])
        else:

            iris = datasets.load_iris()
            self.X = iris.data[:100, 1:3]
            self.Y = iris.target[:100]


        self.mean = np.array([np.mean(self.X[:,i]) for i in range(self.X.shape[1])])
        self.var = np.array([np.var(self.X[:,i]) for i in range(self.X.shape[1])])
        self.center = (self.X - np.reshape(self.mean,(1,self.X.shape[1])))/np.reshape(self.var,(1,self.X.shape[1]))

        a = np.array([np.linalg.norm(self.center[i,:]) for i in range(self.X.shape[0])])

        self.norm = np.zeros((self.X.shape[0],self.X.shape[1]))
        for i in range(self.X.shape[0]):
            self.norm[i,:] = self.center[i,:]/a[i]

        self.center_b = np.append(self.center[:batch//2,:],self.center[sample_len//2:(sample_len+batch)//2, :], axis=0)
        self.norm_b = np.append(self.norm[:batch//2,:],self.norm[sample_len//2:(sample_len+batch)//2, :], axis=0)

    def resample(self, X, Y, over_ratio=0.9, under_ratio=1.0):
        over = SMOTE(sampling_strategy=over_ratio)
        under = RandomUnderSampler(sampling_strategy=under_ratio)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        X_r, y_r = pipeline.fit_resample(X, Y)

        return X_r, y_r

    def update_batch(self, batch_index):
        self.X_b = np.append(self.X[(batch_index*self.batch)//2:((batch_index+1)*self.batch)//2,:],
                             self.X[(self.sample_len + batch_index*self.batch)//2:(self.sample_len + (batch_index+1)*self.batch)//2, :], axis=0)

        self.Y_b = np.append(self.Y[(batch_index*self.batch)//2:((batch_index+1)*self.batch)//2],
                             self.Y[(self.sample_len + batch_index*self.batch)//2:(self.sample_len + (batch_index+1)*self.batch)//2])

        self.norm_b = np.append(self.norm[(batch_index*self.batch)//2:((batch_index+1)*self.batch)//2,:],
                                self.norm[(self.sample_len + batch_index*self.batch)//2:(self.sample_len + (batch_index+1)*self.batch)//2, :], axis=0)

        self.center_b = np.append(self.center[(batch_index*self.batch)//2:((batch_index+1)*self.batch)//2,:],
                                self.center[(self.sample_len + batch_index*self.batch)//2:(self.sample_len + (batch_index+1)*self.batch)//2, :], axis=0)


    def show_data(self, x0, x1, xt, all_data=False):

        plt.figure(2, figsize=(8, 6))
        plt.clf()

        # Plot the training points
        if all_data:
            plt.scatter(self.norm[:, 0], self.norm[:, 1], c=self.Y)
        else:
            plt.scatter(np.array([self.norm[x0, 0], self.norm[x1, 0], self.norm[xt,0]]),
                        np.array([self.norm[x0, 1], self.norm[x1, 1], self.norm[xt,1]]),
                        c=np.array([self.Y[x0], self.Y[x1], self.Y[xt]]))
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        #plt.hist(self.X[:,0], bins=len(self.X[:,0])//50)

        plt.show()