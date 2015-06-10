# -*- coding: utf-8 -*-
"""
Weighted Linear Regression , Expert in HME model
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
        self.weights      = weights       # weighting of observations
        self.mse          = 0             # mean squared error
        self.X            = X
        self.Y            = Y


    def fit(self,X,Y,weights, lambda_ridge =  0):
        '''
        Fits weighted ridge regression 
        '''
        W = np.diagflat(weights)
        part_one    = np.dot(np.dot(X.T,W),X)
        part_two    = np.dot(np.dot(X.T,W),Y)
        self.theta  = cholesky_solver_least_squares(part_one, part_two)
        self.mse    =  np.dot((Y - np.dot(X,self.theta)).T, (Y - np.dot(X,self.theta)))
        

    def predict(self,X_test):
        '''
        Predicts target values for X_test
        '''
        fitted = np.dot(X_test,)
        return fitted
        
        
    def get_fitted_params(self):
        
        
        
        
        
        