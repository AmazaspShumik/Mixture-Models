# -*- coding: utf-8 -*-


from scipy.optimize import fmin_l_bfgs_b
import numpy as np

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
    
    
    def __init__(self, n_input, n_hidden, n_output,minibatch = 100, conv_tol= 1e-5, max_iter = 1000):
        
        # number of nodes in input, hidden and output layers
        self.in_n      =  n_input  + 1                      #
        self.hi_n      =  n_hidden + 1                      
        self.out_n     =  n_output
        
        # weighting matrices from input layer to hidden and from hidden to output
        self.W_in_hi   =  0.001*np.random.random([self.in_n,n_hidden])    
        self.W_hi_out  =  0.001*np.random.random([self.hi_n,self.out_n])
        
        # parameters for convergence
        self.conv_tol  = conv_tol
        self.max_iter  = max_iter

        
    def forward_propagation(self,X):
        '''
        Forward propagation 
        '''
        n,m      =  np.shape(X) 
        
        # augment X with column of ones (to account for bias term)
        bias_in  =  np.ones([n,1])
        X        =  np.concatenate((X,bias_in), axis = 1)
        #print self.W_in_hi
        
        # hidden unit activations
        h_act    =  np.dot(X,self.W_in_hi)
        # hidden layer value
        H        =  logistic_sigmoid(h_act)
        # augment hidden layer values with columns of ones (bias term)
        bias_hi  =  np.ones([n,1])        
        H        =  np.concatenate((H,bias_hi), axis = 1)
        
        #print H
        # output units activations
        o_act    =  np.dot(H,self.W_hi_out)
        #print o_act

        return o_act
        
    
    def _backprop_train(self,X,Y,W):
        '''
        Training Neural Network using Backpropagation
        '''
        # reshape weights from vector format to matrix format
        W_in_hi,W_hi_out,in_hi,hi_out = self.weight_vec_to_mat_transform(W, dim_params = True)
        
        # forward propagate
        n,m       =  np.shape(X) 
        bias_in   =  np.ones([n,1])
        X         =  np.concatenate((X,bias_in), axis = 1)        
        h_act     =  np.dot(X,W_in_hi)        
        H         =  logistic_sigmoid(h_act)
        bias_hi   =  np.ones([n,1])        
        H         =  np.concatenate((H,bias_hi), axis = 1)
        o_act     =  np.dot(H,W_hi_out)
        
        # errors in output layer
        err_out         =  ( Y - o_act ).T
        print err_out
        #print err_out
        # exclude bias term to propagate error to hidden layer
        W_hi_out_b      =  W_hi_out[0:(self.hi_n-1),:]
        # error  of hidden layer
        err_hi          =  d_logistic_sigmoid(h_act).T * np.dot(W_hi_out_b,err_out)
        # gradient for weights from hidden to output layer
        grad_hi_out     =  np.dot(err_out,H)
        # gradient for weights from input to hidden layer
        grad_in_hi      =  np.dot(err_hi,X)
        cost            =  np.dot(err_out,err_out.T)
        grad_hi_layer   =  np.reshape(grad_in_hi,(in_hi,))
        grad_out_layer  =  np.reshape(grad_hi_out,(hi_out,))
        # put gradient in weight vector (no need to create new arrays)
        W[0:in_hi]      =  grad_hi_layer
        W[in_hi:(in_hi+hi_out)] = grad_out_layer        
        return [cost[0][0],W]
        
        
    def training(self,X,Y,W):
        fitter           = lambda W: self._backprop_train(X,Y,W)
        W_opt,J,D        = fmin_l_bfgs_b(fitter,W,fprime = None,
                                     pgtol = self.conv_tol,
                                     maxiter = self.max_iter)
        print J,D
        self.W_in_hi, self.W_hi_out, = self.weight_vec_to_mat_transform(W_opt)
        
                
        
    #------------------------------- Helper methods ------------------------------------#    
        
        
    def weight_vec_to_mat_transform(self,W,dim_params = False):
        # hidden layer parameters excluding bias term
        hi        =  self.hi_n -1     
        # number of elements in weight matrix (in -> hi)                           
        in_hi     =  hi*self.in_n       
        # number of elements in weight matrix (hi -> out)                        
        hi_out    =  self.hi_n*self.out_n       
        W_in_hi   =  np.reshape(W[0:in_hi], (self.in_n, hi))
        W_hi_out  =  np.reshape(W[in_hi: in_hi+hi_out], (self.hi_n,self.out_n))
        if dim_params is False:
           return [W_in_hi,W_hi_out]
        return [W_in_hi,W_hi_out, in_hi,hi_out]
        
        
        
if __name__=="__main__":
    nnet = NeuralNetwork(2,4,1)
    X = np.zeros([60,2])
    X[:,0] = np.linspace(0,4,60)
    X[:,1] = np.random.random(60)
    Y = np.zeros([60,1])
    Y[:,0] = X[:,0]+np.random.normal(60)
    #print nnet.forward_propagation(X)
    #print nnet.W_hi_out
    #print nnet.W_in_hi
    #nnet._backprop_train(X,Y,W)
    nnet.training(X,Y,np.random.random(17))
    #nnet._backprop_train(X,Y,np.random.random(17))
    Y_hat = nnet.forward_propagation(X)
        
        
        
        
        
        
        
        
    
        
        