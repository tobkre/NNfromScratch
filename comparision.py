# -*- coding: utf-8 -*-
"""
Created on Wed May  2 11:11:20 2018

@author: kretz01
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from MLP_Class import MLP
import warnings
warnings.filterwarnings('ignore')

if __name__=='__main__':
    a = 0.5
    b = 6.
    c = 9.
    n_samples = 10
    n_features = 1
        
#    X_train = np.random.uniform(1,10,size=(n_samples,n_features))
    X = np.linspace(start=1, stop=10, num=n_samples).reshape(n_samples,n_features)
    y = a*X**2 - b*X + c
    y = y.ravel()
    
    n_epochs = 300
    lr = 0.01
    lmbda = 1e-5
    tol=1e-4
    
    train_opts = {'epochs':n_epochs, 'learning_rate':lr, 'regularization':lmbda, 'tolerance':tol}
    
    print('Sklearn')
    sk_clf = MLPRegressor(solver='sgd', alpha=lmbda, hidden_layer_sizes=(5, 2),
                          verbose=True, random_state=1, momentum=0,
                          max_iter=n_epochs, learning_rate='adaptive', learning_rate_init=lr)
    sk_clf.fit(X, y)
    
    print('Own')
    my_clf = MLP(train_features=n_features, hidden_layer_sizes=(5, 2), n_out=1, dropout_rate=0.4, learning_rate='constant', verbose=True)
    my_clf.train_on_data(X, y, train_opts=train_opts)