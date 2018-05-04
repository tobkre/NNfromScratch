# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 13:48:13 2018

@author: kretz01
"""
import numpy as np

class Layer:
    """Object to define a fully connected layer of a neural net .
    
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
        R       (): dropout matrix (kept as 1 in output layer to disable dropout there)
    """
    
    def __init__(self, X_in, n_input, n_nodes, activation, seed):
        np.random.seed(seed=seed)
        self.W = np.random.uniform(low=-0.02, high=0.02, size=(n_input ,n_nodes))
        self.b = np.random.uniform(low=-0.02, high=0.02, size=(n_nodes))
#        self.W = np.ones((n_input ,n_nodes))
#        self.b = np.zeros((n_nodes))
        self.dy = None
        self.X = X_in
        self.R = 1
        
        if activation=='linear':
            self.act_fcn = lambda x: self.__lin_act(x)
            self.act_der = lambda x: self.__lin_der(x)
        elif activation=='relu':
            self.act_fcn = lambda x: self.__rel_act(x)
            self.act_der = lambda x: self.__rel_der(x)
        elif activation=='sigmoid':
            self.act_fcn = lambda x: self.__sig_act(x)
            self.act_der = lambda x: self.__sig_der(x)
        elif activation=='tanh':
            self.act_fcn = lambda x: self.__tanh_act(x)
            self.act_der = lambda x: self.__tanh_der(x)  
        else:
            raise ValueError('Unknown activation function:',activation)
        
    def update_params(self, eta, dW, db, dy):
        self.W = self.W - eta * (dW * self.R)
        self.b = self.b - eta * db
        self.dy = dy
    
    def fwd_prop(self, dropout=1):
        return np.dot(self.X, self.W*dropout) + self.b
    
    def __lin_act(self, arg):
        return arg
    
    def __lin_der(self, arg):
        return np.ones_like(arg)

    def __rel_act(self, arg):
        y = arg
        y[arg<0]=0
        return y
    
    def __rel_der(self, arg):
        y = np.ones_like(arg)
        y[arg<0]=0
        return y
    
    def __sig_act(self, arg):
        return 1/(1 + np.exp(-arg))
    
    def __sig_der(self, arg):
        return arg * (1-arg)
#        return self.__sig_act(arg) * (1-self.__sig_act(arg))
        
    def __tanh_act(self, arg):
        return np.tanh(arg)
    
    def __tanh_der(self, arg):
        return 1 - arg**2
        

class HiddenLayer(Layer):
    """Object to define a hidden layer of a FC neural net .
    
    Args:
        n_input      (int): number of input nodes
        n_nodes      (int): number of nodes in Layer
        dropout_rate (int)
        activation   (str): string to define the activation function of the Layer
        seed         (str, int, optional): seed for random initialization
        
    Attributes:
        W       (): Weight matrix of size
        b       (): Vector of biases 
        R       (): Dropout Matrix
        act_fcn (): activation function
        act_der (): derivative of actual activation function
    """
    def __init__(self, n_input, n_nodes, X_in=None, dropout_rate=0, activation='linear', seed=None):
        Layer.__init__(self, X_in=X_in, n_input=n_input, n_nodes=n_nodes, activation=activation, seed=seed)
        self.n_input = n_input
        self.n_nodes = n_nodes
        self.draw_dropout_sample(dropout_rate=dropout_rate)
    
    def draw_dropout_sample(self, dropout_rate):
        self.R = np.random.binomial(size=(self.n_input, self.n_nodes), n=1, p=1-dropout_rate)
        #self.R = np.random.binomial(size=(self.n_nodes), n=1, p=1-dropout_rate)
        
    def print_w(self):
        print(self.W)
        
    def get_output(self, X_in=None):
        if X_in is not None:
            self.X = X_in
        return self.act_fcn(self.fwd_prop(dropout=self.R))
        #return self.act_fcn(self.fwd_prop(dropout=self.R)) * self.R
        

class OutputLayer(Layer):
    """Object to define an output layer of a FC neural net .
    
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
    def __init__(self, n_input, n_nodes, X_in=None, activation='linear', seed=None):
        Layer.__init__(self, X_in=X_in, n_input=n_input, n_nodes=n_nodes, activation=activation, seed=seed)
        self.x=1
    
    def print_w(self):
        print(self.W)
        
    def get_output(self, X_in=None):
        if X_in is not None:
            self.X = X_in
        return self.act_fcn(self.fwd_prop())
