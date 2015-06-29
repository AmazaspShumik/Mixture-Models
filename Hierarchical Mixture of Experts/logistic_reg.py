# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import label_binariser as lb

#----------------------------- Helper Methods ---------------------------------------#

def logistic_sigmoid(M):
    '''
    Sigmoid function
    '''
    1.0/( 1 + np.exp(-1*M ))
    
    
def logistic_cost_grad(theta,Y,X,weights):
    '''
    Calculates cost and gradient for logistic cost function
    
    Parameters:
    -----------
    
    theta: numpy array of size 'm x 1'
         Vector of model parameters
         
    Y: numpy array of size 'n x 1'
         Target variable can take only values 0 or 1
         
    X: numpy array of size 'n x m'
         Explanatory variables
         
    weights: numpy array of size 'n x 1'
         Weights for observations
         
    Returns:
    --------
    
    cost_grad: list of size 2
         First element of list is value of cost function, second element
         is gradient
    
    '''
    n,m       =  np.shape(X)
    p_one     =  logistic_sigmoid(np.dot(X,theta))
    p_zero    =  np.ones(n) - p_one
    cost      =  np.dot( weights *Y , np.log(p_one) ) - np.dot( weights * (np.ones(n) - Y) , p_zero )
    grad      =  np.dot( X , p_zero - Y)
    cost_grad =  [cost, grad]
    return cost_grad
    
def logistic_pdf(theta,Y,X):
    '''
    Calculates probability of observing Y given theta and X
    
    Parameters:
    -----------
    
    theta: numpy array of size 'm x 1'
         Vector of model parameters
         
    Y: numpy array of size 'n x 1'
         Target variable can take only values 0 or 1
         
    X: numpy array of size 'n x m'
         Explanatory variables
         
    Returns:
    --------
    probs:  numpy arrays of size 'n x 1'
    
    
    '''
    n            = np.shape(Y)[0]
    probs        = logistic_sigmoid(np.dot(X,theta))
    probs[Y==1]  = np.ones(n) - probs
    return probs
    
# ------------------------------- Logistic Regression Class --------------------------------------#

class LogisticRegression(object):
    
    
    def __init__(self, p_tol, max_iter):
        self.p_tol   =  p_tol
        self.maxiter =  max_iter
        
    def _preprocessing_targets(self,Y):
        self.binarisator = lb.LabelBinariser(Y,2)
        
        
    def fit(self,Y_raw,X,weights):
        n,m          = np.shape(X)
        fitter       = lambda theta: logistic_cost_grad(theta,Y,X,weights)
        theta_init   = 0.1*np.random.random(m)
        theta,J,D    = fmin_l_bfgs_b(fitter, theta_init, fprime = None, pgtol = self.p_tol, maxiter = self.maxiter)
        self.theta   = theta
        

        
    def predict_probs(self):
        pass
    
    def predict(self):
        pass
    
if __name__ == "__main__":
    pass
        