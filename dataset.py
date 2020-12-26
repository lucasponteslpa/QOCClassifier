import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from pdb import set_trace
import tensorflow.compat.v1 as tf
import pandas as pd
from utils import likelihood

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

            mask_c1 = data[:,3] != 1
            c1s = np.ma.array(data[:,3], mask= mask_c1)
            n_c1 = c1s.sum()

            # PUT SEED TO FIX THE RANDOMIZATION
            c1_index = np.random.choice(range(n_c1),size=sample_len//2)
            c2_index = np.random.choice(range(n_c1,data.shape[0]),size=sample_len//2)
            samples_index = np.append(c1_index,c2_index)
            self.X = data[samples_index,0:2]
            self.Y = data[samples_index,3]

            self.X_b = np.append(self.X[:batch//2,:],self.X[sample_len//2:(sample_len+batch)//2, :], axis=0)
            self.Y_b = np.append(self.Y[:batch//2],self.Y[sample_len//2:(sample_len+batch)//2])

        elif name == 'Habermans':
            TRAIN_DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data'

            train_file_path = tf.keras.utils.get_file("haberman.data.csv", TRAIN_DATA_URL)
            self.dataset = pd.read_csv(train_file_path).to_numpy()

            mask_c1 = self.dataset[:,3] != 1
            c1s = np.ma.array(self.dataset[:,3], mask= mask_c1)
            n_c1 = c1s.sum()

            # PUT SEED TO FIX THE RANDOMIZATION
            c1_index = np.random.choice(range(n_c1),size=50)
            c2_index = np.random.choice(range(n_c1,self.dataset.shape[0]),size=50)
            samples_index = np.append(c1_index,c2_index)

            self.X = self.dataset[samples_index,0:3:2]
            self.Y = self.dataset[samples_index,3]

        else:
            iris = datasets.load_iris()
            self.X = iris.data[:100, 1:3]  # we only take the first two features.
            self.Y = iris.target[:100]
        #set_trace()
        self.mean = np.array([np.mean(self.X[:,i]) for i in range(self.X.shape[1])])
        self.var = np.array([np.var(self.X[:,i]) for i in range(self.X.shape[1])])
        self.center = (self.X - np.reshape(self.mean,(1,self.X.shape[1])))/np.reshape(self.var,(1,self.X.shape[1]))
        a = np.array([np.linalg.norm(self.center[i,:]) for i in range(self.X.shape[0])])
        #set_trace()
        self.norm = np.zeros((self.X.shape[0],self.X.shape[1]))
        for i in range(self.X.shape[0]):
            self.norm[i,:] = self.center[i,:]/a[i]

        self.norm_b = np.append(self.norm[:batch//2,:],self.norm[sample_len//2:(sample_len+batch)//2, :], axis=0)

    def update_batch(self, batch_index):
        self.X_b = np.append(self.X[(batch_index*self.batch)//2:((batch_index+1)*self.batch)//2,:],
                             self.X[(self.sample_len + batch_index*self.batch)//2:(self.sample_len + (batch_index+1)*self.batch)//2, :], axis=0)
        self.Y_b = np.append(self.Y[(batch_index*self.batch)//2:((batch_index+1)*self.batch)//2],
                             self.Y[(self.sample_len + batch_index*self.batch)//2:(self.sample_len + (batch_index+1)*self.batch)//2])
        self.norm_b = np.append(self.norm[(batch_index*self.batch)//2:((batch_index+1)*self.batch)//2,:],
                                self.norm[(self.sample_len + batch_index*self.batch)//2:(self.sample_len + (batch_index+1)*self.batch)//2, :], axis=0)


    def show_data(self, x0, x1, xt, all_data=False):

        plt.figure(2, figsize=(8, 6))
        plt.clf()

        # Plot the training points
        if all_data:
            plt.scatter(self.norm[:, 0], self.norm[:, 1], c=self.Y)
        else:
            plt.scatter(np.array([self.norm[x0, 0], self.norm[x1, 0], self.norm[xt,0]]), np.array([self.norm[x0, 1], self.norm[x1, 1], self.norm[xt,1]]), c=np.array([self.Y[x0], self.Y[x1], self.Y[xt]]))
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        #plt.hist(self.X[:,0], bins=len(self.X[:,0])//50)

        plt.show()