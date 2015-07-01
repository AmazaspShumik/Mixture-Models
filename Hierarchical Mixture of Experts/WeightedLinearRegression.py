# -*- coding: utf-8 -*-
"""
Weighted Linear Regression , Expert in HME model

   m - dimensionality of input (i.e. length of row in matrix X)
   n - number of observations
"""

import numpy as np

def cholesky_solver_least_squares(part_one, part_two):
    '''
    Solves least squares problem using cholesky decomposition
    
    part_one - numpy array of size 'm x m', equals X.T * X
    part_two - numpy array of size 'm x 1', equals X.T * Y
    
    '''
    # R*R.T*Theta = part_two
    R = np.linalg.cholesky(part_one)
    # R*Z = part_two
    Z     = np.linalg.solve(R,part_two)
    # R.T*Theta = Z
    Theta = np.linalg.solve(R.T,Z)
    return Theta
    
    
def norm_pdf(theta,y,x,sigma_2):
    '''
    Calculates probability of observing Y given Theta and sigma
    
    
    Input:
    ------
    
    Theta    -  numpy array of size 'm x k', matrix of parameters
    Y        -  numpy array of size 'n x 1', vector of dependent variables
    X        -  numpy array of size 'n x m', matrix of inputs 
    sigma_2  -  numpy array of size 'm x 1', vector of variances
    
    Output:
    -------
    prob: numpy array of size 'n x 1'
          Probability of observing y given theta and X
    
    '''
    normaliser = 1.0/np.sqrt(2*np.pi*sigma_2)
    u          = y - np.dot(x,theta)
    prob       = normaliser* np.exp( -0.5 * u*u / sigma_2 )
    return prob
    
    


class WeightedLinearRegression(object):
    '''
    Weighted Linear Regression
    
    Fits weighted  regression 
        
    Input:
    ------
    
    Y                        - numpy array of size 'n x 1', vector of dependent variables
    X                        - numpy array of size 'n x m', matrix of inputs
    weights                  - numpy array of size 'n x 1', vector of observation weights
    
    '''
    
    def __init__(self):
        self.theta        = 0             # coefficients excluding bias term
        self.mse          = 0             # mean squared error
        self.var          = 0             # fitted variance

    def init_weights(self,m):
        self.theta = np.zeros(m)
        self.var   = 1 

    def fit(self,X,Y,weights):
        ''' Fits weighted regression '''
        n,m         =  np.shape(X)
        W           =  np.diagflat(weights)
        part_one    =  np.dot(np.dot(X.T,W),X)
        part_two    =  np.dot(np.dot(X.T,W),Y)
        self.theta  =  cholesky_solver_least_squares(part_one, part_two)
        vec_1       =  (Y - np.dot(X,self.theta))
        self.var    =  np.dot(vec_1,np.dot(vec_1,W))/np.sum(W)
        
    def predict(self,X):
        return np.dot(X,self.theta)
        


