# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 08:52:12 2018

@author: kretz01
"""
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor


if __name__=='__main__':
    import matplotlib.pyplot as plt
#    my_net = build_initmodel(1, 3, 1)
#    pred = predict(my_net, np.array([1,5]))
#    print(pred)
    n_samples=1000
    n_features=1
    n_epochs = 7000
    a = 0.5
    b = 6.
    c = 9.
    
    np.random.seed(seed=2)
    x = np.random.uniform(1,10,size=(n_samples,n_features))
    y = (a*x**2 - b*x + c).flatten()


    my_model = MLPRegressor(
        # try some layer & node sizes
        hidden_layer_sizes=(10,5,5), 
        solver='sgd',
        # find a learning rate?
        learning_rate='constant',
        learning_rate_init=0.001,
        shuffle=False,
        # activation functions (relu, tanh, identity)
        activation='tanh',
        epsilon=1e-11,
        max_iter=800000)
    #x = np.reshape(np.array([2,4,1]), (3,1))
    #y = np.reshape(np.array([4,16,1]), (3,1))
    
    my_model.fit(x, y)

    pd.DataFrame(my_model.loss_curve_).plot()
    
    x_test = np.random.uniform(1,10,size=(10,1))
    x_test = np.sort(x_test, axis=0)
    y_test = my_model.predict(x_test)
    
    x = np.sort(x, axis=0)
    new_x = x.flatten()
    plt.figure()
    plt.plot(new_x, my_model.predict(x), 'o--')
    plt.plot(new_x, a*new_x**2 - b*new_x + c, 'o--')
    plt.show()
    #pred = predict(trained_net, np.array([10,5]))
    #print(pred)