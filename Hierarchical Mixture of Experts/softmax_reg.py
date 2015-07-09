# -*- coding: utf-8 -*-


import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import label_binariser as lb
from helpers import *


def softmax(Theta,X):
    '''
    Calculates value of softmax function. Output can be interpreted as matrix
    of probabilities, where  each entry of column i is probability that corresponding 
    input belongs to class 'i'.
    

    Parameters:
    -----------
    
    Theta: numpy array of size 'm x k'
             Matrix of coefficients
             
    X: numpy array of size 'n x m'
             Explanatory variables
    
    Returns:
    --------
     : numpy array of size 'n x k'
             Matrix of probabilities

    '''
    n,m = np.shape(X)
    m,k = np.shape(Theta)
    max_vec = np.max( np.dot(X,Theta), axis = 1)                       # substract max element, prevents overflow
    X_Theta = np.dot(X,Theta) - np.outer(max_vec,np.ones(k))
    return np.exp( X_Theta )/np.outer(np.sum( np.exp( X_Theta ) , axis = 1),np.ones(k))
    
    

    
def cost_grad(Theta,Y,X,k,weights, underflow_tol = 1e-20):
    '''
    Calculates negative log likelihood and gradient of negative log likelihood of multinomial
    distribution together. Reusing intermediate values created in process of likelihood
    and estimation makes this function more efficient than calls to two separate 
    function.
    
    Parameters:
    ----------
    
    Theta: numpy array of size 'm x k', 
             Matrix of coefficients
             
    Y:  numpy array of size 'n x k'
             Ground Truth Matrix
             
    X: numpy array of size 'n x m'
             Explanatory variables
    
    k: int 
             Number of classes
             
    underflow_tol: float
             Threshold for preventing underflow
    
    Returns:
    --------
    
    tuple(int, np.array): tuple, of size 2
            First element is value of cost function, second element is numpy array 
            of size 'm x 1', representing gradient
    '''
    n,m         =  np.shape(X)
    Theta       =  np.reshape(Theta,(m,k))                                         
    P           =  bounded_variable(softmax(Theta,X), underflow_tol,1)
    unweighted  =  np.sum(Y*np.log(P), axis = 1)
    cost        =  -1.0/n*np.dot(weights,unweighted)
    resid       =  (Y-P)
    grad        =  -1.0/n*np.dot(np.dot(X.T,np.diagflat(weights)),resid)
    return (cost, np.reshape(grad,(m*k,)))
    

#----------------------------------------  Softmax Regression  --------------------------------------------#
    
class SoftmaxRegression(object):
    '''
    Softmax classifier using l-bfgs-b optimization procedure.
    (Bias term is not added in the process of computation, so  it needs to be in
    data matrix)

    Parameters:
    -----------
    
    max_iter: int 
                Maximum nmber of iterations (default = 1000)
                
    tolerance: float 
                Precision threshold for convergence (default = 1e-10)
                
    underflow_tol: float (default: 1e-20)
                Threshold for preventing underflow
                
                
    '''
    
    def __init__(self,tolerance = 1e-10, max_iter = 80, underflow_tol = 1e-20):
        self.tolerance              = tolerance
        self.max_iter               = max_iter
        self.underflow_tol          = underflow_tol

        
    def init_params(self,m,k):
        '''
        
        Parameters:
        -----------
        m: int
           Dimensionality of data        
        
        k: int
           Number of classes in classification problem
        '''
        self.theta = np.random.random([m,k])
        
        
    def _pre_processing_targets(self,Y,k):
        ''' 
        Preprocesses data, transforms from vector Y to ground truth matrix Y
        '''
        self.binarisator = lb.LabelBinariser(Y,k)
        return self.binarisator.convert_vec_to_binary_matrix()

        
    def fit(self,Y_raw,X,k,weights, preprocess_input = False):
        '''
        Fits parameters of softmax regression using l-bfgs-b optimization procedure
                
        Parameters:
        -----------
        
        X: numpy array of size 'n x m'
            Expalanatory variables
            
        Y_raw: numpy array of size 'n x 1'
            Dependent variables that need to be approximated
            
        k: int
            Number of classes
            
        weights: numpy array of size 'n x 1'
            Weighting for each observation
            
        preprocess_input: bool
            If true Y_raw is assumed to be vector, and is transformed into
            ground truth matrix of zeros and ones of size 'n x k', where k is
            number of classes
        
        '''
        if preprocess_input is True:
            Y            =  self._pre_processing_targets(Y_raw,k)
        else:
            Y            =  Y_raw
        fitter          = lambda theta: cost_grad(theta,Y,X,k,weights,self.underflow_tol)
        n,m             = np.shape(X)
        self.k          = k
        theta_initial   = 0.1*np.random.random(m*k)
        theta,J,D       = fmin_l_bfgs_b(fitter,
                                        theta_initial,
                                        fprime = None,
                                        pgtol = self.tolerance,
                                        approx_grad = False,
                                        maxiter = self.max_iter)
        self.theta      = np.reshape(theta,(m,k))
        
        
    def predict_probs(self,X_test):
        '''
        Calculates matrix of probabilities for given data matrix
        
        Parameters:
        -----------
        
        X_test: numpy array of size 'uknown x m' 
             Explanatory variables of test set
        
        Returns:
        -------
                 
        P: numpy array of size 'uknown x k'
             Matrix of probabilities, showing probabilty of observation belong
             to particular class
        '''
        P = softmax(self.theta,X_test)
        return P
    
    
    def predict(self,X_test):
        '''
        For each observation in X predicts class to which it belongs
        This method can be used if in 'fit' method preprocess_input was True
        
        Parameters:
        -----------
        
        X: numpy array of size 'unknown x m'
            Expalanatory variables
            
        Returns:
        --------
        
        prediction: numpy array of size 'unknown x m'
            Estimated target value for each observation
            
        '''
        p          = self.predict_probs(X_test)
        prediction = self.binarisator.convert_prob_matrix_to_vec(p)
        return prediction
        
        
    def log_likelihood(self,X,Y,weights, preprocess = False):
        '''
        Returns log likelihood for softmax regression
        
        Parameters:
        -----------
        
        X: numpy array of size 'n x m'
             Explanatory variables
        
        Y: numpy array of size 'n x 1'
             Target variable can take only values 0 or 1
         
        weights: numpy array of size 'n x 1'
             Weights for observations
             
        preprocess: bool
             If True transforms vector to ground truth matrix
             
        Returns:
        --------
        
        weighted_log_likelihood: float
             log likelihood
        
        '''
        if preprocess is True:
            Y = self.binarisator.convert_vec_to_binary_matrix()
        weighted_log_likelihood = -1*cost_grad(self.theta,Y,X,self.k,weights,self.underflow_tol)[0]
        return weighted_log_likelihood
        

