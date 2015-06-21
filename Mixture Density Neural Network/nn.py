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
        self.W_in_hi   =  np.zeros([self.in_n,n_hidden])    
        self.W_hi_out  =  np.zeros([self.hi_n,self.out_n])
        
        # activation functions
        if   activ == "logistic-sigmoid":
            self.activation = logistic_sigmoid
        elif activ == "tahn":
            self.activation = tahn
            
        # 
        
        
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
        H        =  self.activation(h_act)
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
        H         =  self.activation(h_act)
        bias_hi   =  np.ones([n,1])        
        H         =  np.concatenate((H,bias_hi), axis = 1)
        
        # errors in output layer
        err_out   = ( Y - H ).T
        err_hi    =  d_logistic_sigmoid(h_act).T * np.dot(self.W_hi_out,err_out)
        
        
        
        
        
    
        
        