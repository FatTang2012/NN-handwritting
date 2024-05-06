import numpy as np
from matplotlib import pyplot as plt
import math
from keras.datasets import mnist
import cv2
from tkinter import Tk, Label, PhotoImage, TOP
import os

class NN:
    def __init__(self, X_train, Y_train):
        self.W1 = np.random.randn(256, 784) * np.sqrt(2/784) # (256, 784)
        self.b1 = np.zeros((256, 1)) # (256, 1)
        self.W2 = np.random.randn(10, 256) * np.sqrt(2/256) # (10, 256)
        self.b2 = np.zeros((10, 1)) # (10, 1)
        self.X_train = X_train #(784, 60000)
        self.Y_train = Y_train #(60000,)

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def deriv_relu(self, Z):
        return (Z > 0).astype(int)
    
    def softmax(self, Z):
        e_z = np.exp(Z - np.max(Z))
        return e_z / e_z.sum(axis=0)

    def forward_prop(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1  # (256, 784) * (784, m) = (256, m)
        self.A1 = self.relu(self.Z1)  # (256, m)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2  # (10, 256) * (256, m) = (10, m)
        self.A2 = self.softmax(self.Z2)  # (10, m)

    def one_hot(self, y):
        n_classes = len(np.unique(self.Y_train)) # 10
        y_oh = np.eye(n_classes)[y] # (60000, 10)
        return y_oh.T # (10, 60000)
        
    def backward_prop(self, X, y):
        m = y.size # 60000
        y_oh = self.one_hot(y) # (10, 60000)
        dZ2 = self.A2 - y_oh # (10, 60000)
        self.dW2 = (1/m) * np.dot(dZ2, self.A1.T) # (10, 60000) * (60000, 256) = (10, 256)
        self.db2 = (1/m) * np.sum(dZ2, axis=1).reshape(-1, 1) # (10, 1)
        dZ1 = np.dot(self.W2.T, dZ2) * self.deriv_relu(self.Z1) # (256, 10) * (10, 60000) hadamard (256, 60000) = (256, 60000)
        self.dW1 = (1/m) * np.dot(dZ1, X.T) # (256, 60000) * (60000, 784) = (256, 784)
        self.db1 = (1/m) * np.sum(dZ1, axis=1).reshape(-1, 1) # (256, 1)

    def update_params(self, lr):
        self.W2 -= lr * self.dW2 # (10, 256)
        self.b2 -= lr * self.db2 # (10, 1)
        self.W1 -= lr * self.dW1 # (256, 784)
        self.b1 -= lr * self.db1 # (256, 1)

    def predict(self, X):
        self.forward_prop(X)
        return np.argmax(self.A2, axis=0) 

    def accuracy(self, prediction, y):
        return np.sum(prediction == y) / y.size 

    def random_mini_batches(self, X, y, mini_batch_size=64, seed=0):
        np.random.seed(seed)
        m = X.shape[1]
        mini_batches = []
    
        permutation = list(np.random.permutation(m)) # [0, 1, 2, ..., data_lenght-1] in random order
        X_shuffled = X[:, permutation]
        y_shuffled = y[permutation]
    
        n_batches = math.floor(m / mini_batch_size)
        for k in range(n_batches):
            mini_batch_X = X_shuffled[:, k*mini_batch_size:(k+1)*mini_batch_size]
            mini_batch_y = y_shuffled[k*mini_batch_size:(k+1)*mini_batch_size]
            mini_batches.append((mini_batch_X, mini_batch_y))
        if m % mini_batch_size != 0:
            mini_batch_X = X_shuffled[:, int(m/mini_batch_size)*mini_batch_size:]
            mini_batch_y = y_shuffled[int(m/mini_batch_size)*mini_batch_size:]
            mini_batches.append((mini_batch_X, mini_batch_y))  
        return mini_batches 
        
    def save_model(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savez(file_path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        
    def load_model(self, file_path):
        with np.load(file_path) as data:
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']

    def fit(self, X, y, mini_batch_size, lr, n_epochs, save_path):
        seed = 10
        for i in range(n_epochs):
            seed += 1
            minibatches = self.random_mini_batches(X, y, mini_batch_size, seed)
            for minibatch in minibatches:
                X_minibatch, y_minibatch = minibatch
                self.forward_prop(X_minibatch)
                self.backward_prop(X_minibatch, y_minibatch)
                self.update_params(lr)
            if i % 10 == 0:
                preds = self.predict(X)
                acc = self.accuracy(preds, y)
                print('Accuracy on training data after epoch %i: %f' % (i, acc))
    
        preds = self.predict(X)
        print('\nModel accuracy on training data:', self.accuracy(preds, y))

        self.save_model(save_path)

        
image_path = "D:\\Document move here\\Learning\\Hoc ki\\N2\\HK2 N2\\mang than kinh\\full set number\\ve.png"   

def train_model():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1).T / 255.

    nn = NN(X_train, y_train)
    nn.fit(X_train, y_train, 512, 0.01, 400, 'save_model/my_model')

if __name__ == "__main__":
    train_model()

