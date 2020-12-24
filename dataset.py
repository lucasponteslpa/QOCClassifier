import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from pdb import set_trace
import tensorflow.compat.v1 as tf
import pandas as pd
from utils import likelihood

class ProcessData():

    def __init__(self, name='iris'):
        if name == 'skin':
            TRAIN_DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt'

            train_file_path = tf.keras.utils.get_file("segmentation.data.csv", TRAIN_DATA_URL)
            dataset = pd.read_csv(train_file_path).to_numpy()
            data = np.array([ i.split('\t') for i in dataset[:,0]]).astype(np.int32)
            # mean = np.array([np.mean(data[:,i]) for i in range(data.shape[1])])
            # var = np.array([np.var(data[:,i]) for i in range(data.shape[1])])

            mask_c1 = data[:,3] != 1
            c1s = np.ma.array(data[:,3], mask= mask_c1)
            n_c1 = c1s.sum()

            # split_data = lambda dat,i,l: dat[i,0:2] if i<l else dat[i+n_c1,0:2]
            # split_class = lambda dat,i,l: dat[i,3] if i<l else dat[i+n_c1,3]
            # self.X = np.array([split_data(data,i,2000) for i in range(4000)])
            # self.Y = np.array([split_class(data,i,2000) for i in range(4000)])
            c1_index = np.random.choice(range(n_c1),size=50)
            c2_index = np.random.choice(range(n_c1,data.shape[0]),size=50)
            samples_index = np.append(c1_index,c2_index)
            self.X = data[samples_index,0:2]
            self.Y = data[samples_index,3]
            #print('Likelihood feature 1: '+str(likelihood(mean[0],var[0],self.X[:,0])))
            #print('Likelihood feature 2: '+str(likelihood(mean[1],var[1],self.X[:,1])))
            # print('Mean data: '+str(mean[0])+' Mean samples: '+str(np.mean(self.X[:,0])))
            # print('Var data: '+str(var[0])+' Var samples: '+str(np.var(self.X[:,0])))

        elif name == 'Habermans':
            TRAIN_DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data'

            train_file_path = tf.keras.utils.get_file("haberman.data.csv", TRAIN_DATA_URL)
            dataset = pd.read_csv(train_file_path).to_numpy()

            mask_c1 = dataset[:,3] != 1
            c1s = np.ma.array(dataset[:,3], mask= mask_c1)
            n_c1 = c1s.sum()

            c1_index = np.random.choice(range(n_c1),size=50)
            c2_index = np.random.choice(range(n_c1,dataset.shape[0]),size=50)
            samples_index = np.append(c1_index,c2_index)

            self.X = dataset[samples_index,0:3:2]
            self.Y = dataset[samples_index,3]
            # data = dataset[:,0:3:2]
            # self.X = data  # we only take the first two features.
            # self.Y = dataset[:,3]
            
        else:
            iris = datasets.load_iris()
            self.X = iris.data[:, 1:3]  # we only take the first two features.
            self.Y = iris.target
        #set_trace()
        self.mean = np.array([np.mean(self.X[:,i]) for i in range(self.X.shape[1])])
        self.var = np.array([np.var(self.X[:,i]) for i in range(self.X.shape[1])])
        self.center = (self.X - np.reshape(self.mean,(1,self.X.shape[1])))/np.reshape(self.var,(1,self.X.shape[1]))
        a = np.array([np.linalg.norm(self.center[i,:]) for i in range(self.X.shape[0])])
        #set_trace()
        self.norm = np.zeros((self.X.shape[0],self.X.shape[1]))
        for i in range(self.X.shape[0]):
            self.norm[i,:] = self.center[i,:]/a[i]

    def show_data(self, x0, x1, xt, all_data=False):

        plt.figure(2, figsize=(8, 6))
        plt.clf()

        # Plot the training points
        if all_data:
            plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y)
        else:
            plt.scatter(np.array([self.norm[x0, 0], self.norm[x1, 0], self.norm[xt,0]]), np.array([self.norm[x0, 1], self.norm[x1, 1], self.norm[xt,1]]), c=np.array([self.Y[x0], self.Y[x1], self.Y[xt]]))
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        plt.xlim(0, 100)
        plt.ylim(0, 100)

        #plt.hist(self.X[:,0], bins=len(self.X[:,0])//50)

        plt.show()