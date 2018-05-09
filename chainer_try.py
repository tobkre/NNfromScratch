# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:35:37 2018

@author: kretz01
"""
import argparse
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers, training, Variable
from chainer.training import extensions

class SLN(chainer.Chain):
    def __init__(self):
        super(SLN, self).__init__(
                fc1 = L.Linear(None, 20),
                fco = L.Linear(None, 1)
                )        
    
#    def __call__(self, data, label):
#        print()
#        h = F.sigmoid(self.fc1(data))
#        h = F.dropout(h, ratio=0.5)
#        y = self.fco(h)
#        mse = 0.5*F.mean_squared_error(y, label)
#        return y
    def forward(self, x_data, y_data, train=True, n_patches=32):
        if not isinstance(x_data, Variable):
            x = Variable(x_data)
        else:
            x = x_data
            x_data = x.data
        self.n_images = y_data.shape[0]
        self.n_patches = x_data.shape[0]
        self.n_patches_per_image = self.n_patches / self.n_images

        h = F.dropout(F.sigmoid(self.fc1(x)), ratio=0.5)
        h = self.fco(h)
        
        if train:
            return self.loss
        else:
            return self.loss, self.y

def main():
    a = 0.5
    b = 6.
    c = 9.
    
    n_samples = 100
    n_features = 1
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()
    
    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    
    model = SLN()
#    optimizer = optimizers.SGD().setup(model)
    
    X_train = np.linspace(start=1, stop=10, num=n_samples).reshape(n_samples, n_features).astype(np.float32)
    y_train = a*X_train**2 - b*X_train + c
    
    model.forward(X_train, y_train, False, X_train.shape[0])
#    train = X_train
#    label = y_train
#    dataset = chainer.datasets.TupleDataset(train, label)
#    #train, test = chainer.datasets.get_mnist()
#    
#    train_iter = chainer.iterators.SerialIterator(dataset, 10)
#    #test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)
#    
#    updater = training.updaters.StandardUpdater(train_iter, optimizer)
#    
#    trainer = training.Trainer(updater, (100, 'epoch'), out='result')
#    # Print a progress bar to stdout
##    trainer.extend(extensions.ProgressBar())
#    
#    trainer.run()

if __name__=='__main__':
    main()
    