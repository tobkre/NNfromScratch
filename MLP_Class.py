# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 13:50:33 2018

@author: kretz01
"""

import numpy as np
from collections import OrderedDict
from Layer_Classes import HiddenLayer, OutputLayer

class MLP:
    """Object to define a multilayer perceptron (MLP).
    
    Args:
        n_input      (int): number of input nodes
        n_nodes      (int): number of nodes in Layer
        activation   (str): string to define the activation function of the Layer
        seed         (str, int, optional): seed for random initialization
        
    Attributes:
        W       (): Weight matrix of size
        b       (): Vector of biases 
        act_fcn (): activation function
        act_der (): derivative of actual activation function
    """
    def __init__(self, train_features, hidden_layer_sizes=(5,), n_out=1):
        self.n_hidden_layers = len(hidden_layer_sizes)
        self.n_nodes_input = train_features
        self.n_nodes_output = n_out
        self.n_nodes_hidden = hidden_layer_sizes
        self.n_layers = self.n_hidden_layers + 1
        self.Layers = OrderedDict()
        
        self.__init_model()
        
    def __init_model(self):
        n_in = self.n_nodes_input
        for i in range(self.n_hidden_layers):
            idf='hidden'+str(i)
            n_nodes = self.n_nodes_hidden[i]
            layer = HiddenLayer(n_input=n_in, n_nodes=n_nodes, activation='sigmoid')
            n_in = n_nodes
            self.Layers[idf] = layer
        out_layer = OutputLayer(n_input = n_in, n_nodes=self.n_nodes_output)
        self.Layers['out'] = out_layer
    
    def train_on_data(self, X_train, y_train, train_opts):
        epochs = train_opts['epochs']
        lr = train_opts['learning_rate']
        layer_output = np.zeros((self.n_layers,), dtype=np.object) 
        mse = np.zeros(epochs)
        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            print('Epoch', epoch)
            #forward propagation
            for (i,key) in enumerate(self.Layers):                
                if i==0:
                    layer_input = X_train
                else:
                    layer_input = layer_output[i-1]
#                print(key, 'Input')
#                print(layer_input)
#                print('W')
#                print(self.Layers[key].W)
#                print('b')
#                print(self.Layers[key].b)
#                print('_______________________________________')
                self.Layers[key].X = layer_input
                layer_output[i] = self.Layers[key].get_output()
                
            #backpropagation
            for (j, key) in enumerate(reversed(self.Layers)):
                print(key, 'Update')
                if key=='out':
                    dy = layer_output[-1] - y_train
                    prev_layer = self.Layers[key]
#                    print('delta_L')
#                    print(dy)
                    
                else:
#                    print(prev_layer.dy,'x',prev_layer.W.T)
                    dy =  np.dot(prev_layer.dy, prev_layer.W.T)*self.Layers[key].act_der(prev_layer.X)
#                    print('delta_l')
#                    print(dy)
                    prev_layer = self.Layers[key]
                
                if (j+1)==self.n_layers:
                    dW = np.dot(X_train.T, dy)/n_samples
                else:
#                    print('alpha_l-1')
#                    print(self.Layers[key].X.T)
#                    print('delta')
#                    print(dy)
                    dW = np.dot(self.Layers[key].X.T, dy)/n_samples
                db = np.mean(dy, axis=0)
                
                print('dW')
                print(dW)
                print('db')
                print(db)
                print('_______________________________________')
                self.Layers[key].update_params(eta=lr, dW=dW, db=db, dy=dy)
                
            mse[epoch] = np.mean(0.5*(y_train - layer_output[-1]).flatten()**2)
                    
                
        return mse
    
    def predict(self, X):
        layer_output = np.zeros((self.n_layers,), dtype=np.object)  
        for (i,key) in enumerate(self.Layers):
            layer = self.Layers[key]
            if i==0:
                layer_input = X
            else:
                layer_input = layer_output[i-1]
            layer_output[i] = layer.get_output(X_in=layer_input)
        return layer_output[-1]

def test_1(n_samples, n_features, n_epochs, lr):
    a = 0.5
    b = 6.
    c = 9.
    
    np.random.seed(seed=2)
    
    model = MLP(train_features=1, hidden_layer_sizes=(5,), n_out=1)
    
    X_train = np.random.uniform(1,10,size=(n_samples,n_features))
    y_train = a*X_train**2 - b*X_train + c
    #y_train = X_train**2
    
    train_opts = {'epochs':n_epochs, 'learning_rate':lr}
    
    mse = model.train_on_data(X_train=X_train, y_train=y_train, train_opts=train_opts)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(mse)
    
    idx_train = np.argsort(X_train, axis=0).flatten()
    X_train = X_train[idx_train, :]
    y_train = y_train[idx_train]
    
    plt.figure()
    plt.title('Train')
    plt.plot(X_train, y_train, '*')
    plt.plot(X_train, model.predict(X=X_train))
    
def test_2(n_epochs=5, lr=0.1):
    train_opts = {'epochs':n_epochs, 'learning_rate':lr}
    x = np.array([2,3]).reshape(2,1)
    y = x**2
    model = MLP(train_features=1, hidden_layer_sizes=(3,), n_out=1)
    
    mse = model.train_on_data(X_train=x, y_train=y, train_opts=train_opts)
    print()
    print(mse)
    
if __name__=='__main__':
    n_samples=100
    n_features=1
    n_epochs = 1000
    lr = 0.001
    #test_1(n_samples=n_samples, n_features=n_features, n_epochs=n_epochs, lr=lr)
    test_2()
    
   
    