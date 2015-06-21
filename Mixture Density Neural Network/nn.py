# -*- coding: utf-8 -*-


def logistic_sigmoid(X):
    '''
    Sigmoid function
    '''
    return 1.0/(1 + np.exp(-1*X))
    
def d_logistic_sigmoid(X):
    '''
    Derivative of sigmoid function
    '''
    n,m = np.shape(X)
    H   = logistic_sigmoid(X) 
    return H*(np.ones([n,m]) - H)

    
    
from scipy.optimize import fmin_l_bfgs_b
import numpy as np

class NeuralNetwork(object):
    
    '''
    Simple implementation of neural network algorithm
    
    Input:
    -------
    
    n_input    -   dimensionality of input data, (number of nodes in input layer)
    n_hidden   -   number of nodes in hidden layer
    n_output   -   number of nodes in output layer
    activ      -   type of activation function (either "logistic-sigmoid", "tanh")
    
    
    '''
    
    
    def __init__(self, n_input, n_hidden, n_output,minibatch = 100):
        
        # number of nodes in input, hidden and output layers
        self.in_n      =  n_input  + 1                      #
        self.hi_n      =  n_hidden + 1                      
        self.out_n     =  n_output
        
        # weighting matrices from input layer to hidden and from hidden to output
        self.W_in_hi   =  np.random.random([self.in_n,n_hidden])    
        self.W_hi_out  =  np.random.random([self.hi_n,self.out_n])

        
        
    def forward_propagation(self,X):
        '''
        Forward propagation 
        '''
        n,m      =  np.shape(X) 
        
        # augment X with column of ones (to account for bias term)
        bias_in  =  np.ones([n,1])
        X        =  np.concatenate((X,bias_in), axis = 1)
        
        # hidden unit activations
        h_act    =  np.dot(X,self.W_in_hi)        
        # hidden layer value
        H        =  logistic_sigmoid(h_act)
        # augment hidden layer values with columns of ones (bias term)
        bias_hi  =  np.ones([n,1])        
        H        =  np.concatenate((H,bias_hi), axis = 1)
        
        # output units activations
        o_act    =  np.dot(H,self.W_hi_out)
        return o_act
        
    
    def _backprop_train(self,X,Y):
        '''
        Training Neural Network using Backpropagation
        '''
        n,m       =  np.shape(X) 
        bias_in   =  np.ones([n,1])
        X         =  np.concatenate((X,bias_in), axis = 1)        
        h_act     =  np.dot(X,self.W_in_hi)        
        H         =  logistic_sigmoid(h_act)
        bias_hi   =  np.ones([n,1])        
        H         =  np.concatenate((H,bias_hi), axis = 1)
        o_act     =  np.dot(H,self.W_hi_out)
        
        # errors in output layer
        err_out    =  ( Y - o_act ).T
        # exclude bias term to propagate error to hidden layer
        W_hi_out_b =  self.W_hi_out[0:self.hi_n-1,:]
        # error  of hidden layer
        print np.shape(d_logistic_sigmoid(h_act).T)
        print np.shape(np.dot(W_hi_out_b.T,err_out))
        err_hi     =  d_logistic_sigmoid(h_act).T * np.dot(W_hi_out_b.T,err_out)
        # gradient for weights from hidden to output layer
        grad_hi_out  =  np.dot(err_out,H)
        # gradient for weights from input to hidden layer
        grad_in_hi   =  np.dot(err_hi,X)
        print grad_hi_out
        print grad_in_hi
        
        
if __name__=="__main__":
    nnet = NeuralNetwork(2,2,1)
    X = np.zeros([60,2])
    X[:,0] = np.linspace(0,4,60)
    X[:,1] = np.random.random(60)
    Y = np.sin(X)
    #print nnet.forward_propagation(X)
    nnet._backprop_train(X,Y)
        
        
        
        
        
        
        
        
    
        
        