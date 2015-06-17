# -*- coding: utf-8 -*-
"""
Weighted Linear Regression , Expert in HME model
"""

import numpy as np

def cholesky_solver_least_squares(part_one, part_two):
    '''
    Solves least squares problem using cholesky decomposition
    
    
    
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
    sigma_2  -  vector of variances 'm x 1', vector of variances
    
    Output:
    -------
    
             - int
    
    '''
    normaliser = 1.0/np.sqrt(2*np.pi*sigma_2)
    u          = y - np.dot(x,theta)
    return normaliser* np.exp( -0.5 * u*u / sigma_2 )
    
    


class WeightedLinearRegression(object):
    
    def __init__(self,X,Y,weights):
        self.theta        = 0             # coefficients excluding bias term
        self.weights      = weights       # weighting of observations
        self.mse          = 0             # mean squared error
        self.X            = X
        self.Y            = Y
        self.var          = 0             # fitted variance


    def fit(self):
        '''
        Fits weighted ridge regression 
        '''
        n,m         =  np.shape(self.X)
        W           =  np.diagflat(self.weights)
        part_one    =  np.dot(np.dot(self.X.T,W),self.X)
        part_two    =  np.dot(np.dot(self.X.T,W),self.Y)
        self.theta  =  cholesky_solver_least_squares(part_one, part_two)
        vec_1       =  (self.Y - np.dot(self.X,self.theta))
        self.var    =  np.dot(vec_1,np.dot(vec_1,W))/np.sum(W)
        

    def predict(self,X_test):
        '''
        Predicts target values for X_test
        '''
        fitted = np.dot(X_test,)
        return fitted


if __name__=="__main__":
    X = np.ones([100,3])
    X[:,0] = np.linspace(0,10,100)+np.random.random(100)
    X[:,1] = np.linspace(0,10,100)+np.random.random(100)
    Y = 2*X[:,0] + 4*X[:,1] + np.random.normal(0,1,100) -  2*np.ones(100)
    weights = np.ones(100)
    wlr = WeightedLinearRegression(X,Y,weights)
    wlr.fit()
