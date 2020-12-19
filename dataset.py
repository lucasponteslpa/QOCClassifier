import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from pdb import set_trace

class ProcessData():

    def __init__(self, name='iris'):
        self.data_name = name
        self.iris = datasets.load_iris()
        self.X = self.iris.data[:, :2]  # we only take the first two features.
        self.Y = self.iris.target

        self.mean = np.array([np.mean(self.X[:,i]) for i in range(self.X.shape[1])])
        self.var = np.array([np.var(self.X[:,i]) for i in range(self.X.shape[1])])
        self.center = (self.X - np.reshape(self.mean,(1,self.X.shape[1])))/np.reshape(self.var,(1,self.X.shape[1]))
        a = np.array([np.linalg.norm(self.center[i,:]) for i in range(self.X.shape[0])])
        #set_trace()
        self.norm = np.zeros((self.X.shape[0],self.X.shape[1]))
        for i in range(self.X.shape[0]):
            self.norm[i,:] = self.center[i,:]/a[i]

    def show_data(self, x0, x1, xt):

        plt.figure(2, figsize=(8, 6))
        plt.clf()

        # Plot the training points
        #plt.scatter(self.norm[42:58, 0], self.norm[42:58, 1], c=np.concatenate((self.Y[42:54],[self.Y[54]+4],self.Y[55:58])))
        #plt.scatter(self.norm[:100, 0], self.norm[:100, 1], c=self.Y[:100])
        plt.scatter(np.array([self.norm[x0, 0], self.norm[x1, 0], self.norm[xt,0]]), np.array([self.norm[x0, 1], self.norm[x1, 1], self.norm[xt,1]]), c=np.array([self.Y[x0], self.Y[x1], self.Y[xt]]))
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        plt.show()