# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 08:25:01 2018

@author: kretz01
"""

"""Build a MLP with one hidden layer, one input layer (two values) and a single
neuron in the ooutput layer, to predict the outcome of the &-Function"""
 
import numpy as np
from Layer_Classes import Layer, HiddenLayer, OutputLayer

def calc_loss(model):
    return 1
    
def activation(x):
    #return 1/(1+np.exp(-x))
    return x

def deriv_activation(x):
    #return activation(x) + (1-activation(x))
    return np.ones_like(x) #activation(x) + (1-activation(x))
    
def  build_initmodel(n_input, n_hidden, n_output):
    model = {}
#    np.random.seed(1)
#    W1 = np.random.uniform(size=(n_hidden, n_input))
#    b1 = np.random.uniform(size=(n_hidden, 1))
#    W2 = np.random.uniform(size=(n_output, n_hidden))
#    b2 = np.random.uniform(size=(n_output, 1))
    W1 = np.ones((n_hidden, n_input))
    b1 = np.ones((n_hidden, 1))
#    b1[1] = 2
#    b1[2] = 3
    W2 = np.ones((n_output, n_hidden))
    b2 = np.zeros((n_output, 1))
    model['W1'] = W1
    model['W2'] = W2
    model['b1'] = b1
    model['b2'] = b2
    return model

def prop_fwd(W1, b1, Wout, bout, x):
    hiddenlayer_input = np.dot(W1, x) + b1.T
    hiddenlayer_output  = activation(hiddenlayer_input)
    outputlayer_input = np.dot(Wout, hiddenlayer_output.T) + bout
    yhat = activation(outputlayer_input)
    return yhat, hiddenlayer_output

def train(X, y, hidden, out, epochs, eta, print_loss=False):
    n_samples, n_features = X.shape
    error = np.zeros(epochs)
    for i in range(epochs):
        print('Epoch', i)
        #print(out.W)
        Wout_grad = np.zeros((n_samples, 3))
        bout_grad = np.zeros(n_samples)
        Whidden_grad = np.zeros((n_samples,3))
        bhidden_grad = np.zeros((n_samples,3))
#        for ind in range(n_samples):
#            hidden_layer_output = hidden.get_layer_output(X[ind])
#            y_net = out.get_layer_output(hidden_layer_output)
#            
#            delta = (y_net - y[ind])
#            out_err = delta * out.act_der(y_net)
#            Wout_grad[ind, :] = np.dot(out_err, hidden_layer_output)
#            bout_grad[ind] = out_err
#            #print(Wout_grad)
#            delta_1 = np.dot(out_err, out.W.T) * hidden.act_der(hidden_layer_output)
#            Whidden_grad[ind, :] = np.dot(delta_1.T, X[ind].reshape((1,1))).flatten()
#            bhidden_grad[ind, :] = delta_1.flatten()
#            
##       
#        dWout = np.reshape(np.sum(Wout_grad, axis=0)/n_samples, out.W.shape)
#        dbout = np.reshape(np.sum(bout_grad, axis=0)/n_samples, out.b.shape)
#        #print(dWout)
#        dWhidden = np.reshape(np.sum(Whidden_grad, axis=0)/n_samples, hidden.W.shape)
#        dbhidden = np.reshape(np.sum(bhidden_grad, axis=0)/n_samples, hidden.b.shape)
        
        
        hidden_layer_output = hidden.get_output(X)
        y_net = out.get_output(hidden_layer_output)
        
        delta = (y_net - y)
        
        #output_layer
        out_err = delta * out.act_der(y_net)
        Wout_grad = out_err * hidden_layer_output
        bout_grad = out_err
    
        dWout = np.reshape(np.sum(Wout_grad, axis=0)/n_samples, out.W.shape)
        dbout = np.reshape(np.sum(bout_grad, axis=0)/n_samples, out.b.shape)
        print('Out')
        print('dW')
        print(dWout)
        print('db')
        print(dbout)
        out.update_params(eta, dWout, dbout, dy=None)
        #hidden layer
        hidden_err = np.dot(out_err, out.W.T) * hidden.act_der(hidden_layer_output)
        Whidden_grad = hidden_err * X
        bhidden_grad = hidden_err
        
        dWhidden = np.reshape(np.sum(Whidden_grad, axis=0)/n_samples, hidden.W.shape)
        dbhidden = np.reshape(np.sum(bhidden_grad, axis=0)/n_samples, hidden.b.shape)
        print('Hidden')
        print('dW')
        print(dWhidden)
        print('db')
        print(dbhidden)
        
        hidden.update_params(eta, dWhidden, dbhidden, dy=None)
        error[i] = np.sum((y_net-y)**2, axis=0)/(2*n_samples)
    return error

def predict(X, hidden, out):
    hidden_layer_output = hidden.get_output(X)
    y_net = out.get_output(hidden_layer_output)
    return y_net


def test_0():
    n_epochs=5
    
    x = np.array([2,3]).reshape(2,1)
    y = x**2
    
   
    hidden_layer = HiddenLayer(n_input=n_features, n_nodes=3, X_in=None, dropout_rate=0, activation='sigmoid', seed=2)
    #print(hidden_layer.W)
    output_layer = OutputLayer(n_input=3, n_nodes=1, X_in=None, activation='linear', seed=2)
    
    mse = train(x, y, hidden_layer, output_layer, n_epochs, 0.1)
    return mse

if __name__=='__main__':
    import matplotlib.pyplot as plt
#    my_net = build_initmodel(1, 3, 1)
#    pred = predict(my_net, np.array([1,5]))
#    print(pred)
    mse = test_0()
    print('MSE', mse)
#    n_samples=100
#    n_features=1
#    n_epochs = 100000
#    
#    a = 0.5
#    b = 6.
#    c = 9.
#    
#    np.random.seed(seed=2)
#    x = np.random.uniform(1,10,size=(n_samples,n_features))
#    y = a*x**2 - b*x + c
#    #y = x**2
#    #x = np.reshape(np.array([2,4,1]), (3,1))
#    #y = np.reshape(np.array([4,16,1]), (3,1))
#    
#    hidden_layer = HiddenLayer(n_input=n_features, n_nodes=5, X_in=None, dropout_rate=0.3, activation='sigmoid', seed=2)
#    #print(hidden_layer.W)
#    output_layer = OutputLayer(n_input=5, n_nodes=1, X_in=None, activation='linear', seed=2)
#    
#    my_mse = train(x, y, hidden_layer, output_layer, n_epochs, 0.01)
#    
#    x_test = np.random.uniform(1,10,size=(25,1))
#    y_test = predict(x_test, hidden_layer, output_layer)
#    
#    idx_train = np.argsort(x, axis=0).flatten()
#    x = x[idx_train, :]
#    y = y[idx_train]
#    
#    idx_test = np.argsort(x_test, axis=0).flatten()
#    x_test = x_test[idx_test, :]
#    y_test = y_test[idx_test]
#    
#    y_true = a*x_test**2 - b*x_test + c
#    plt.figure()
#    plt.title('Error')
#    plt.plot(np.arange(n_epochs), my_mse)
#    
#    plt.figure()
#    plt.title('Train')
#    plt.plot(x, y, '*-')
#    plt.plot(x, predict(x, hidden_layer, output_layer))
#    
#    plt.figure()
#    plt.title('Test')
#    plt.plot(x_test, y_true, '*-')
#    plt.plot(x_test, y_test)
#    plt.show()
#    #pred = predict(trained_net, np.array([10,5]))
#    #print(pred)
        