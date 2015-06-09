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
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import fmin_bfgs
from sklearn import preprocessing
from scipy.sparse import csr_matrix



# ---------------------------  Multinomial Likelihood (with softmax function) ---------------------#

def softmax(Theta,X,i):
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
    max_vec = np.max( np.dot(X,Theta), axis = 0)                       # substract max element, prevents overflow
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
    np.array - numpy array of size 'n*k x 1', matrix of gradients tansformed to vector
    '''
    n,m       = np.shape(X)
    # transform Theta to matrix if required
    if vec_to_mat is True:
        k = np.shape(Theta)[0] / m
        Theta = np.reshape(Theta,(m,k))
    m,k = np.shape(Theta)
    P             = np.array([softmax(Theta,X,i) for i in range(k)]).T  # matrix of class probabilities
    resp          = (Y - P)                                             # dim(resp) = n x k
    weighted_obs  = np.dot(X.T,np.diagflat(weights))                    # dim(weighted_obs)  = m x n
    gradient      = np.dot(weighted_obs, resp)                          # dim(gradient)      = m x k 
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
    int     -  negative log likelihood
    '''
    n,m = np.shape(X)
    # transform Theta to matrix if required
    if vec_to_mat is True:
        k     = np.shape(Theta)[0] / m
        Theta = np.reshape(Theta,(m,k))
    m,k             = np.shape(Theta)
    P               = np.array([softmax(Theta,X,i) for i in range(k)]).T
    likelihood_vec  = np.sum(Y.dot(np.eye(k))*np.log(P), axis = 1)
    likelihood      = np.dot(weights,likelihood_vec)
    return -1*likelihood
    
    
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
    weights - numpy array of size 'n x 1', vector of weightings for observations 
    
    Output:
    -------
    
    tuple(int, np.array) - tuple, where first element is value of cost function,
                           second element is numpy array, representing gradient
    '''
    n,m   = np.shape(X)
    Theta = np.reshape(Theta,(m,k))                                         # Theta transformed to matrix
    P     = np.array([softmax(Theta,X,i) for i in range(k)]).T              # Matrix of probabilities
    cost  = -1*np.dot(weights,np.sum(Y.dot(np.eye(m))*np.log(P),axis = 1))  # E y_i*w_i*log(p_i) (weighted cost)
    X_w   = np.dot(X.T,np.diagflat(weights))                                # weighted observations
    grad  = -1*np.dot(X_w,(Y-P))
    grad = np.array(grad)                                                   # gradient matrix of size m x k (make Fortran contigious)
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
    return tuple([csr_matrix(out), el])  
    
    
    
#----------------------------------------  Softmax Regression  --------------------------------------------#
    
class SoftmaxRegression(object):
    '''
    
    
    
    '''
    
    def __init__(self,Y,X,weights,K, tolerance, max_iter):
        self.labels                 = set(list(Y)) 
        self.X                      = X
        self.Y, self.transformer    = target_preprocessing(Y,K)
        self.weights                = weights
        self.tolerance              = tolerance
        self.max_iter               = max_iter
        self.k                      = K
        self.theta                  = 0
    
    def fit(self):
        fitter          = lambda theta: cost_grad(theta,self.Y,self.X,self.k,self.weights)
        n,m             = np.shape(X)
        theta_initial   = np.zeros([m*self.k,1], dtype = np.float)
        theta,J,D       = fmin_l_bfgs_b(fitter,theta_initial,fprime = None,pgtol = self.tolerance,maxiter = self.max_iter)
        print J, D
        self.theta      = np.reshape(theta,(m,self.k))
        
    def predict_probs(self,X_test):
        '''
        Returns matrix of probabilities for input 'X_test'
        '''
        P = np.array([softmax(self.theta,self.X,i) for i in range(self.k)])
        print P        
        return P
    
    def predict(self,X_test):
        P      =  self.predict_probs(X_test)                # find probability for each class
        M      =  np.max(P, axis = 1)
        Y      =  np.array([1*(P[:,i]==M) for i in range(self.k)])
        if self.k > 2 :
           labels =  self.transformer.inverse_transform(Y)  # corresponding labels
           return labels
        else:
           label_one                 = self.transformer
           label_two                 = list(self.labels)[0]
           if label_one==label_two:
               label_two = list(self.labels)[1]
           print label_two
           Y_labeled                 = np.array([label_one for i in range(np.shape(X_test)[0])])
           Y_labeled[1*(Y[:,1]==1)]  = label_two
        print Y_labeled
        
    def get_fitted_params(self):
        '''
        returns learned parameters
        '''
        return self.theta
        
        
if __name__=="__main__":
    Theta = np.zeros([2,3])
    X = np.random.random([150,2])
    X[0:50,:] = X[0:50,:] + 10
    X[50:100,:] = X[50:100,:] + 20
    Y = np.zeros(150)
    Y[0:50] = np.ones(50)
    Y[50:100] = np.ones(50)*2
    weights = np.ones(150)

    print "----------------------------------------"
    sr = SoftmaxRegression(Y,X,weights,2,0.001,10000)
    sr.fit()
    sr.predict_probs(X)
    sr.predict(X)
    