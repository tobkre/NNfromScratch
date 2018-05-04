# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:01:07 2018

@author: kretz01
"""

import numpy as np
import tensorflow as tf

#TODO: Why does the optimal net does not converge??
if __name__=='__main__':
    a = 0.5
    b = 6.
    c = 9.
    n_samples = 100
    n_features = 1
        
#    X_train = np.random.uniform(1,10,size=(n_samples,n_features))
    batch_x = np.linspace(start=1, stop=10, num=n_samples).reshape(n_samples,n_features)
    batch_y = a*batch_x**2 - b*batch_x + c
    batch_y = batch_y.ravel()
    
    n_epochs = 30000
    lr = 0.005
    lmbda = 1e-5
    tol=1e-4
    
    weights = {
            'hidden1': tf.Variable(tf.random_normal([1, 20])),
#            'hidden2': tf.Variable(tf.random_normal([20, 10])),
            'out': tf.Variable(tf.random_normal([20, 1]))
            }
    
    biases = {
            'hidden1': tf.Variable(tf.random_normal([20])), 
#            'hidden2': tf.Variable(tf.random_normal([10])), 
            'out': tf.Variable(tf.random_normal([1]))
            }
    
    x = tf.placeholder("float", [n_samples, n_features])
    y = tf.placeholder("float", [n_samples])
    
    layer1 = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])
    layer1 = tf.nn.tanh(layer1)
    layer1 = tf.nn.dropout(layer1, keep_prob=0.8)
#    layer2 = tf.add(tf.matmul(layer1, weights['hidden2']), biases['hidden2'])
#    layer2 = tf.nn.tanh(layer2)
    yhat = tf.add(tf.matmul(layer1, weights['out']), biases['out'])
    
    cost = tf.reduce_mean(tf.multiply(0.5, tf.square(y-yhat)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        losses = []
        sess.run(init)
        for epoch in range(n_epochs):
            _, loss_val = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            #print('Epoch',epoch, ' . Loss =',loss_val)
            losses.append(loss_val)
        pred = sess.run(yhat, feed_dict={x: batch_x})
        
    import matplotlib.pyplot as plt
    plt.figure(5)
    plt.clf()
    plt.plot(batch_x, batch_y, '*', label='original')
    plt.plot(batch_x, pred, '.', label='predicted')
    plt.legend(loc='best')
    
    plt.figure(4)
    plt.clf()
    plt.plot(losses)