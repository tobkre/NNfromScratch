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
    def __init__(self, train_features, hidden_layer_sizes=(5,), n_out=1, dropout_rate=0., learning_rate='constant', verbose=False):
        self.n_hidden_layers = len(hidden_layer_sizes)
        self.n_nodes_input = train_features
        self.n_nodes_output = n_out
        self.n_nodes_hidden = hidden_layer_sizes
        self.n_layers = self.n_hidden_layers + 1
        self.Layers = OrderedDict()
        self.dropout_rate = dropout_rate
        self.lr_rule = learning_rate
        self.verbose = verbose
        
        self.__init_model()
        
    def __init_model(self):
        n_in = self.n_nodes_input
        dropout_rate = self.dropout_rate
        for i in range(self.n_hidden_layers):
            idf='hidden'+str(i)
            n_nodes = self.n_nodes_hidden[i]
            layer = HiddenLayer(n_input=n_in, n_nodes=n_nodes, dropout_rate=dropout_rate, activation='sigmoid')
            n_in = n_nodes
            self.Layers[idf] = layer
        out_layer = OutputLayer(n_input = n_in, n_nodes=self.n_nodes_output)
        self.Layers['out'] = out_layer
    
    def train_on_data(self, X_train, y_train, train_opts):
        epochs = train_opts['epochs']
        lr = train_opts['learning_rate']
        lbda = train_opts['regularization']
        tol = train_opts['tolerance']
        layer_output = np.zeros((self.n_layers,), dtype=np.object) 
        dropout_rate = self.dropout_rate
        #mse = np.zeros(epochs)
        mse = []
        lrs = []
        n_samples = X_train.shape[0]
        for epoch in range(epochs):
#            print('Epoch', epoch)
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
#                print(key, 'Update')
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
                    dW = np.dot(X_train.T, dy)/n_samples + lbda * self.Layers[key].W
                else:
#                    print('alpha_l-1')
#                    print(self.Layers[key].X.T)
#                    print('delta')
#                    print(dy)
                    dW = np.dot(self.Layers[key].X.T, dy)/n_samples + lbda * self.Layers[key].W
                db = np.mean(dy, axis=0)
                
#                print('dW')
#                print(dW)
#                print('db')
#                print(db)
                
                self.Layers[key].update_params(eta=lr, dW=dW, db=db, dy=dy)
                if key!='out':
                    self.Layers[key].draw_dropout_sample(dropout_rate=dropout_rate)
#                print('_______________________________________')
#            mse[epoch] = np.mean(0.5*(y_train - layer_output[-1]).flatten()**2)
            #print(layer_output[-1].shape)
            #print(y_train.shape)
            #print()
            
            loss = np.mean(0.5*(y_train - layer_output[-1]).flatten()**2)
            if self.verbose:
                print('Iteration', epoch+1,', loss =',loss)
                
            mse.append(loss)
            lrs.append(lr)
            
            if epoch>=2 and mse[epoch]>mse[epoch-1] and mse[epoch-1]>mse[epoch-2] and self.lr_rule!='adaptive':
                lr = lr/2
                
#            if epoch>=2 and mse[epoch]<mse[epoch-1] and mse[epoch-1]<mse[epoch-2] and self.lr_rule=='adaptive':
#                lr = lr*1.5
               
            if epoch>=2 and abs(mse[epoch]-mse[epoch-1]) < tol and abs(mse[epoch-1]-mse[epoch-2]) < tol:
                print('Training stopped after', epoch, 'epochs. Loss less than tol=',tol)
                break                
        return mse, lrs
    
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
    
    def objective_topology(self, X, y):
        #yhat = self.predict(X)
        #loss = self.objective_loss(y, yhat)
        zetas = np.linspace(start=-1., stop=1.0, num=50)
        losses = np.zeros_like(zetas)
        for (i,zeta) in enumerate(zetas):
            for key in self.Layers:               
                dW = self.Layers[key].Wsave + zeta * self.Layers[key].W_sample
                db = self.Layers[key].bsave + zeta * self.Layers[key].b_sample
#                print(dW.shape)
                self.Layers[key].set_params(W=dW, b=db)
                yhat = self.predict(X)
            loss = self.objective_loss(y, yhat)
            losses[i] = loss
            
            for key in self.Layers:
                self.Layers[key].reset_params()
                
        return losses, zetas

    def objective_loss(self, y, yhat):
        return np.mean(0.5*(y - yhat).flatten()**2)
    
def test_1(n_samples, n_features, n_epochs, lr, lmbda, tol, dropout):
    a = 0.5
    b = 6.
    c = 9.
    scheduler = 'adaptive'

    full_model = MLP(train_features=1, hidden_layer_sizes=(20,), n_out=1,
                dropout_rate=0, learning_rate=scheduler, verbose=False)        

    sparse_model = MLP(train_features=1, hidden_layer_sizes=(20,), n_out=1,
                dropout_rate=dropout, learning_rate=scheduler, verbose=False)
    
#    X_train = np.random.uniform(1,10,size=(n_samples,n_features))
    X_train = np.linspace(start=1, stop=10, num=n_samples).reshape(n_samples, n_features)
    y_train = a*X_train**2 - b*X_train + c
    #y_train = X_train**2
    
    train_opts = {'epochs':n_epochs, 'learning_rate':lr, 'regularization':lmbda, 'tolerance':tol}
    
    print('Train fully connected net')
    mse_full, lrs_full = full_model.train_on_data(X_train=X_train, y_train=y_train, train_opts=train_opts)
    
    for key in full_model.Layers:
        W_init = full_model.Layers[key].W
        b_init = full_model.Layers[key].b
        full_model.Layers[key].safe_params(W=W_init, b=b_init)
    
    
    print('Train dropout net')
    mse_sparse, lrs_sparse = sparse_model.train_on_data(X_train=X_train, y_train=y_train, train_opts=train_opts)
    
    for key in sparse_model.Layers:
        W_init = sparse_model.Layers[key].W
        b_init = sparse_model.Layers[key].b
        sparse_model.Layers[key].safe_params(W=W_init, b=b_init)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title('MSE')
    plt.plot(mse_sparse, label='dropout')
    plt.legend(loc='best')
    
    plt.figure()
    plt.title('learning rate')
    plt.plot(lrs_sparse, label='dropout')
    plt.legend(loc='best')
    
    idx_train = np.argsort(X_train, axis=0).flatten()
    X_train = X_train[idx_train, :]
    y_train = y_train[idx_train]
    
    plt.figure()
    plt.title('Training Data')
    plt.plot(X_train, y_train, '*', label='Train Data')
    plt.plot(X_train, full_model.predict(X=X_train), label='full net')
    plt.plot(X_train, sparse_model.predict(X=X_train), label='with dropout')
    plt.legend(loc='best')
    
    print('Fully connected net error topology')
    full_top, zetas = full_model.objective_topology(X=X_train, y=y_train)
    plt.figure()
    plt.title('Topology')
    plt.plot(zetas, full_top)
    
    print('Dropout net error topology')
    sparse_top, zetas = sparse_model.objective_topology(X=X_train, y=y_train)
    plt.figure()
    plt.title('Topology')
    plt.plot(zetas, sparse_top)
    
    
def test_2(n_epochs=5, lr=0.1):
    train_opts = {'epochs':n_epochs, 'learning_rate':lr}
    x = np.linspace(start=1, stop=10, num=10).reshape(10,1)
    y = x**2
    model = MLP(train_features=1, hidden_layer_sizes=(3,), n_out=1)
    
    mse = model.train_on_data(X_train=x, y_train=y, train_opts=train_opts)
    print()
    print(mse)
    
if __name__=='__main__':
    n_samples=100#0
    n_features=1
    n_epochs = 10000000
    lr = 0.005 
    lmbda = 0
    tol = 1e-9
    
    dropout = 0.3
    test_1(n_samples=n_samples, n_features=n_features, n_epochs=n_epochs, lr=lr, lmbda=lmbda, tol=tol, dropout=dropout)
    #test_2()
    
    
   
    