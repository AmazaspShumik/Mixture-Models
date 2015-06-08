# -*- coding: utf-8 -*-
"""
Discriminative Gating Network.


 m - dimensionality of input (i.e. length of row in matrix X)
 k - number of classes in multinomial distribution
 n - number of observations
    
Uses multinomial likelihood with softmax function to find parameters of
gating network

"""


import numpy as np
from scipy.optimize import fmin_bfgs
from scipy.optimize import check_grad
from scipy.optimize import approx_fprime
from sklearn import preprocessing
from scipy.sparse import csr_matrix



# ---------------------------  Multinomial Likelihood (with softmax function) ---------------------#

def softmax(Theta,X,i, vec_to_mat = False):
    '''
    Calculates value of softmax function. Output can be interpreted as vector
    of probabilities, where is each entry is probability that corresponding 
    input belongs to class 'i'.
    

    Input:
    ------
    Theta      - numpy array of size 'm x k' (or of size 'm*k x 1'), where each column 
                 represents vector of coefficients corresponding to some class
    i          - int, representing class of multinomial distribution
    X          - numpy array of size 'n x m', vector of input
    vec_to_mat - boolean, if True 'Theta' is of size 'm*k x 1' (for optimising with bfgs)
    
    Output:
    ------
               - numpy array of size 'n x 1', vector of probabilities
            
    '''
    n,m = np.shape(X)
    if vec_to_mat is True:
        k = np.shape(Theta)[0] / m
        Theta = np.reshape(Theta,(m,k))
    # to prevent numerical overflow substract max element
    max_vec = np.max( np.dot(X,Theta), axis = 0)
    X_Theta = np.dot(X,Theta) - np.outer(max_vec,np.ones(n)).T
    return np.exp( X_Theta[:,i] )/np.sum( np.exp( X_Theta ) , axis = 1) 
    
    
def negative_log_multinomial_gradient(Theta, Y, X, weights, vec_to_mat = False):
    '''
    Calculates gradient for multinomial model.
    
    Input:
    ------
    Theta   - numpy array of size 'm x k', where each column represents vector of coefficients
    Y       - scipy.sparse.csr_matrix (sparse matrix) of size 'n x k'
    X       - numpy array of size 'n x m', vector of input
    weights - numpy array of size 'n x 1', vector of weightings for observations 
    
    Output:
    ------
            - numpy array of size 'n*k x 1', matrix of gradients tansformed to vector
    '''
    n,m       = np.shape(X)
    if vec_to_mat is True:
        k = np.shape(Theta)[0] / m
        Theta = np.reshape(Theta,(m,k))
    m,k = np.shape(Theta)
    P = np.array([softmax(Theta,X,i) for i in range(k)]).T  # matrix of class probabilities
    resp          = (Y - P)                                 # dim(resp) = n x k
    weighted_obs  = np.dot(X.T,np.diagflat(weights))        # dim(weighted_obs)  = m x n
    gradient      = np.dot(weighted_obs, resp)              # dim(gradient)      = m x k 
    return -1*gradient


def negative_log_multinomial_likelihood(Theta, Y, X, weights, vec_to_mat = False):
    '''
    Calculates negative log likelihood.
    
    Input:
    ------
    Theta   - numpy array of size 'm x k', where each column represents vector of coefficients
    Y       - scipy.sparse.csr_matrix (sparse matrix) of size 'n x k'
    X       - numpy array of size 'n x m', vector of input
    weights - numpy array of size 'n x 1', vector of weightings for observations 
    
    Output:
    ------
            - int, negative log likelihood
    '''
    n,m = np.shape(X)
    if vec_to_mat is True:
        k = np.shape(Theta)[0] / m
        Theta = np.reshape(Theta,(m,k))
    m,k = np.shape(Theta)
    P = np.array([softmax(Theta,X,i) for i in range(k)]).T
    likelihood_vec = np.sum(Y.dot(np.eye(k))*np.log(P), axis = 1)
    likelihood = np.dot(weights,likelihood_vec)
    return -1*likelihood
    
    
#-------------------------------------- Optimization with BFGS --------------------------------#
    
    
def optimize(Y,X,Theta,weights):
    
    # cost function decorator for bfgs optimization 
    def cost_decorator(Theta):
        pass
        # convert Theta from vector to matrix
        #theta_vec = 
        #return negative_log_multinomial_likelihood(Y,X,theta_vec,weights)
    # gradient decorator for bfgs optimization
    def gradient_decorator(Theta):
        pass
        # convert Theta from vector to matrix
        #grad = negative_log_multinomial_gradient(Y,X,Theta,weights)
        # convert gradient from matrix to vector
        #return grad
        
    pass



    
#------------------------------------------ Helper functions -----------------------------------#


def target_preprocessing(Y,k):
    '''
    Transforms vector of class labels into Compressed Sparse Row Matrix
    
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
        return csr_matrix(lb.fit_transform(Y))
    out = np.zeros([len(Y),2])
    el = Y[0]
    out[:,0] = 1*(Y==el)
    out[:,1] = 1*(Y!=el)
    return csr_matrix(out)  
    
    

if __name__=="__main__":
    Theta = np.zeros([2,2])
    X = np.random.random([8,2])
    Y = target_preprocessing(np.array([1,1,1,0,0,1,1,1]),2)
    weights = np.ones(8)
    print negative_log_multinomial_gradient(Theta,Y,X,np.ones(8))
    print negative_log_multinomial_likelihood(Theta,Y,X, weights)
    #print negative_log_multinomial_gradient(np.reshape(Theta,(4,1)),Y,X,np.ones(8),vec_to_mat = True)
    #print negative_log_multinomial_likelihood(np.reshape(Theta,(4,1)),Y,X, weights, vec_to_mat = True)

    