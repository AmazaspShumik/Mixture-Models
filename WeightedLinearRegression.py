# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:27:28 2015

@author: amazaspshaumyan
"""

import numpy as np

def cholesky_solver_least_squares(part_one, part_two):
    '''
    
    '''
    # R*R.T*Theta = part_two
    R = np.linalg.cholesky(part_one)
    # R*Z = part_two
    Z     = np.linalg.solve(R,part_two)
    # R.T*Theta = Z
    Theta = np.linalg.solve(R.T,Z)
    return Theta




class WeightedRidgeLinearRegression(object):
    
    def __init__(self,X,Y,weights):
        self.theta        = 0             # coefficients excluding bias term
        self.beta_0       = np.mean(Y)    # bias term
        self.lambda_ridge = 0             # regularisation parameter
        self.weights      = weights       # weighting of observations
        self.mse          = 0             # mean squared error
        self.X            = X
        self.Y            = Y


    def fit(self,X,Y,weights, lambda_ridge =  0):
        '''
        Fits weighted ridge regression 
        '''
        n,m               = np.shape(X)
        self.lambda_ridge = lambda_ridge
        # center data (since bias term is included see excercise 3.5 from ESL)
        Y_c        =  Y - self.beta_0                                                # centered Y vector
        mu         =  np.mean(X,axis = 0)
        X_c        =  X - np.outer(mu,np.ones(m)).T                                  # centered X matrix
        W          =  np.diagflat(weights)
        part_one   =  np.dot(np.dot(X_c.T,W),X_c) + lambda_ridge*np.eye(m)
        part_two   =  np.dot(np.dot(X_c.T,W),Y_c)
        Theta      =  cholesky_solver_least_squares(part_one, part_two)              # solves least squares with cholesky decomposition
        mse        =  1/n* np.sqrt(np.dot((Y_c - X_c*Theta).T, ( Y_c - X_c*Theta)))  # mean squared error
        self.theta =  Theta
        self.mse   =  mse
    

    def predict(self,X_test):
        '''
        Predicts target values for X_test
        '''
        m,n                 = np.shape(X_test)
        # add bias term
        X_test_aug          = np.zeros([n,(m+1)])
        X_test_aug[:,m+1]   = np.ones(n)
        X_test_aug[:,0:m+1] = X_test
        Theta_bias          = list(self.theta)
        Theta_bias          = Theta_bias.append(self.beta_0)
        # fit
        fitted              = np.dot(X_test_aug,Theta_bias)
        return fitted
        
        
        
        