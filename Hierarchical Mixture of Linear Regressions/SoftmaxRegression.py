# -*- coding: utf-8 -*-
"""
Softmax Regression for Gating Network in HME model

 m - dimensionality of input (i.e. length of row in matrix X)
 k - number of classes in multinomial distribution
 n - number of observations
    
Uses multinomial likelihood with softmax function to find parameters of
gating network

"""


import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn import preprocessing
from scipy.sparse import csr_matrix



# ---------------------------  Multinomial Likelihood (with softmax function) ---------------------#

def softmax(Theta,X):
    '''
    Calculates value of softmax function. Output can be interpreted as matrix
    of probabilities, where is each entry of columns i is probability that corresponding 
    input belongs to class 'i'.
    

    Input:
    ------
    Theta      - numpy array of size 'm x k' (or of size 'm*k x 1'), where each column 
                 represents vector of coefficients corresponding to some class
    X          - numpy array of size 'n x m', vector of input
    
    Output:
    ------
               - numpy array of size 'n x k', matrix of probabilities
               
               
    Example:
    --------
    sr = SoftmaxRegression(Y,X,weights,2)
    sr.fit()
    P = sr.predict_probs(X)
    th = sr.get_fitted_params()
            
    '''
    n,m = np.shape(X)
    m,k = np.shape(Theta)
    max_vec = np.max( np.dot(X,Theta), axis = 1)                       # substract max element, prevents overflow
    X_Theta = np.dot(X,Theta) - np.outer(max_vec,np.ones(k))
    return np.exp( X_Theta )/np.outer(np.sum( np.exp( X_Theta ) , axis = 1),np.ones(k))
    
    
def negative_log_multinomial_gradient(Theta, Y, X, k, vec_to_mat = False):
    '''
    Calculates gradient for multinomial model.
    
    Input:
    ------
    Theta   - numpy array of size 'm x k', where each column represents vector of coefficients
    Y       - scipy.sparse.csr_matrix (sparse matrix) of size 'n x k'
    X       - numpy array of size 'n x m', vector of input
    k       - int, number of classes
    
    Output:
    ------
    np.array - numpy array of size 'n*k x 1', matrix of gradients tansformed to vector
    '''
    n,m       = np.shape(X)
    # transform Theta to matrix if required
    if vec_to_mat is True:
        Theta = np.reshape(Theta,(m,k))
    m,k = np.shape(Theta)
    P             = softmax(Theta,X)                                    # matrix of class probabilities
    resp          = (Y - P)                                             # dim(resp) = n x k
    gradient      = np.dot(X.T, resp)                                   # dim(gradient)      = m x k 
    gradient      = np.reshape(np.array(gradient),m*k)
    return -1.0/n*gradient


def negative_log_multinomial_likelihood(Theta, Y, X,k, vec_to_mat = False):
    '''
    Calculates negative log likelihood.
    
    Input:
    ------
    Theta   - numpy array of size 'm x k', where each column represents vector of coefficients
    Y       - scipy.sparse.csr_matrix (sparse matrix) of size 'n x k'
    X       - numpy array of size 'n x m', vector of input
    k       - int, number of classes
    
    Output:
    ------
    int     -  negative log likelihood
    '''
    n,m = np.shape(X)
    # transform Theta to matrix if required
    if vec_to_mat is True:
        Theta = np.reshape(Theta,(m,k))
    m,k             = np.shape(Theta)
    P               = softmax(Theta,X)
    likelihood      = np.sum(Y.dot(np.eye(k))*np.log(P))
    return -1.0/n*likelihood
    
    
def cost_grad(Theta,Y,X,k,weights):
    '''
    Calculates negative log likelihood and gradient of negative log likelihood of multinomial
    distribution together. Reusing intermediate values created in process of likelihood
    and estimation makes this function more efficient than calls to two separate 
    function.
    
    Input:
    ------
    Theta   - numpy array of size 'm x k', where each column represents vector of coefficients
    Y       - scipy.sparse.csr_matrix (sparse matrix) of size 'n x k'
    X       - numpy array of size 'n x m', vector of input
    k       - int, number of classes
    
    Output:
    -------
    
    tuple(int, np.array) - tuple, where first element is value of cost function,
                           second element is numpy array, representing gradient
    '''
    n,m     = np.shape(X)
    Theta   = np.reshape(Theta,(m,k))                                         # Theta transformed to matrix
    P       = softmax(Theta,X)                                                # Matrix of probabilities
    unweighted = np.sum(Y.dot(np.eye(k))*np.log(P), axis = 1)
    cost    = -1.0/n*np.dot(weights,unweighted)                       # E y_i*w_i*log(p_i) (weighted cost)
    resid   = (Y.dot(np.eye(k))-P)
    grad    = -1.0/n*np.dot(np.dot(X.T,np.diagflat(weights)),resid)
    grad    = np.array(grad)                                                   # gradient matrix of size m x k (make Fortran contigious)
    return (cost, np.reshape(grad,(m*k,)))
    

    
#------------------------------------------ Target Preprocessing -----------------------------------------#


def target_preprocessing(Y,k):
    '''
    Transforms vector of class labels into Compressed Sparse Row Matrix.
    (Ground truth matrix)
    
    Input:
    ------
          Y - numpy array of size 'n x 1'
          k - number of classes
    
    Output:
    -------
           - scipy.sparse.csr_matrix (sparse matrix of size 'n x k')
    '''
    if k >2:
       lb = preprocessing.LabelBinarizer(sparse_output = True)
       return (csr_matrix(lb.fit_transform(Y)),lb)
    out = np.zeros([len(Y),2])
    el = Y[0]
    out[:,0] = 1*(Y==el)
    out[:,1] = 1*(Y!=el)
    return csr_matrix(out)

    
#----------------------------------------  Softmax Regression  --------------------------------------------#
    
class SoftmaxRegression(object):
    '''
    Softmax classifier using l-bfgs-b optimization procedure.
    (Bias term is not added in the process of computation, so  it needs to be in
    data matrix)

    Parameters:
    ----------
    Y            -   numpy array of size 'n x 1', dependent variable
    X            -   numpy array of size 'n x m', explanatory variable
    K            -   int, number of classes for classification
    max_iter     -   int, maximum nmber of iterations            (default=1000)
    tolerance    -   float, precision threshold for convergence  (default=1e-5)

    '''
    
    def __init__(self,tolerance = 1e-3, max_iter = 100):
        self.tolerance              = tolerance
        self.max_iter               = max_iter
        self.theta                  = 0
    
    def fit_vector_output(self,Y,X,K,weights):
        '''
        Fits parameters of softmax regression l-bfgs-b optimization procedure
        '''
        # ground truth matrix
        Y_gt                       = target_preprocessing(Y,K) 
        self._fit(Y_gt,X,K,weights)
        
    def fit_matrix_output(self,Y,X,weights):
        n,k   =  np.shape(Y)
        self._fit(Y,X,k,weights)
        
        
    def _fit(self,Y_gt,X,k, weights):
        '''
        Fits parameters of softmax regression l-bfgs-b optimization procedure
        '''
        fitter          = lambda theta: cost_grad(theta,Y_gt,X,k,weights)
        n,m             = np.shape(X)
        theta_initial   = np.zeros(m*k, dtype = np.float)
        theta,J,D       = fmin_l_bfgs_b(fitter,theta_initial,fprime = None,pgtol = self.tolerance,maxiter = self.max_iter)
        self.theta      = np.reshape(theta,(m,k))
        
        
    def predict_probs(self,X_test):
        '''
        Calculates matrix of probabilities for given data matrix
        
        Input:
        ------
        X_test    -  numpy array of size 'u x m' (where u is unknown size parameter)
        
        Output:
        -------
                  - numpy array of size 'u x k' (where u is unknown size parameter)
        '''
        P = softmax(self.theta,X_test)
        return P
    
    
    def get_fitted_params(self):
        '''
        Returns learned parameters of softmax regression
        
        Output:
        -------
                 - numpy array of size ' m x k'

        '''
        return self.theta

  