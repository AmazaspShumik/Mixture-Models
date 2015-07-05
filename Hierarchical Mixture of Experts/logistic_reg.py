# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import label_binariser as lb
import matplotlib.pyplot as plt

#----------------------------- Helper Methods ---------------------------------------#

def logistic_sigmoid(M):
    '''
    Sigmoid function
    '''
    return 1.0/( 1 + np.exp(-1*M ))
    
    
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
    H         =  logistic_sigmoid(np.dot(X,theta))
    cost_vec  =  Y*np.log(H) + (1 - Y)*np.log(1 - H)
    cost      =  -1.0*np.sum(weights*cost_vec)
    grad      =  np.dot( np.dot(X.T,np.diagflat(weights)) , H - Y)
    cost_grad =  (cost, grad.T)
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
    probs[Y==0]  = np.ones(n)[Y==0] - probs[Y==0]
    return probs
    
# ------------------------------- Logistic Regression  -----------------------------------------#

class LogisticRegression(object):
    '''
    Logistic Regression, linear classification model
    
    Parameters:
    -----------
    
    p_tol: float
           Convergence threshold for Iterative Reweighted Least Squares
           
    max_iter: int
            Maximum number of iterations of IRLS
    
    '''
    
    def __init__(self, p_tol = 1e-5, max_iter = 40):
        self.p_tol   =  p_tol
        self.maxiter =  max_iter
        
    def init_params(self,m):
        self.theta = np.random.normal(0,1,m)
        
        
    def _preprocessing_targets(self,Y):
        '''
        Preprocesses target vaector into 0-1 encoded vector
        '''
        self.binarisator = lb.LabelBinariser(Y,2)
        return self.binarisator.logistic_reg_direct_mapping()
        
        
    def fit(self,X,Y_raw,weights = None, preprocess_input = False):
        '''
        Fits model, finds best parameters.
        
        Parameters:
        -----------
        
        Y_raw: numpy array of size 'n x 1'
              Vector of targets, that shouyld be approxiamted
              
        X: numpy array of size 'n x m'
              Matrix of inputs, training set
              
        weights: numpy array of size 'n x 1'
              Weighting of observations
              
        preprocess_input: bool
              True if target variable is not vector of zeros and ones
        
        '''
        # preprocess target variables if needed
        if preprocess_input is True:
            Y        = self._preprocessing_targets(Y_raw)
        else:
            Y        = Y_raw
        n,m          = np.shape(X)
        # default weighting
        if weights is None:
            weights = np.ones(n)
        fitter       = lambda theta: logistic_cost_grad(theta,Y,X,weights)
        theta_init   = np.zeros(m)
        theta,J,D    = fmin_l_bfgs_b(fitter, theta_init, fprime = None, pgtol = self.p_tol, maxiter = self.maxiter)
        self.theta   = theta
        
    
    def predict_probs(self,X, theta = None):
        '''
        Predicts probability of belonging to one of two classes for test set data
        
        Parameters:
        -----------
        
        X: numpy array of size 'unknown x m'
              Matrix of inputs, test set
              
        theta: numpy array of size 'm x 1'
              Vector of parameters
              
        Returns:
        --------
        
        probs: numpy array of size 'n x 1'
              Vector of probabilities
        '''
        # default parameters are for fitted model
        if theta is None:
            theta = self.theta
        probs        = logistic_sigmoid(np.dot(X,theta))
        return probs
        
        
    def predict(self,X, theta = None, postprocess_output = True):
        '''
        Predicts target class to which observation belong
        
        Parameters:
        -----------
        
        X: numpy array of size 'unknown x m'
              Matrix of inputs, test set
              
        theta: numpy array of size 'm x 1'
              Vector of parameters
              
        postprocess_output: bool
             If True transforms vector of zeros and one to original format class
              
        Returns:
        --------
        
        pr: numpy array of size 'n x 1'
              Vector of target classes (belongs to one )
        '''
        pr           = self.predict_probs(X,theta)
        pr[pr>0.5]   = 1
        pr[pr<0.5]   = 0
        if postprocess_output is True:
            return self.binarisator.logistic_reg_inverse_mapping(pr)
        return pr
        

if __name__ == "__main__":
#    X = np.ones([50,3])
#    X[:,1] = np.random.normal(0,1,50)
#    X[:,2] = np.random.normal(0,1,50)
#    X[25:50,1:3] = X[25:50,1:3] + 10
#    Y = np.array(["y" for i in range(50)])
#    Y[25:50] = "n"
#    lr = LogisticRegression()
#    lr.fit(Y,X,np.ones(50))
#    Y_hat = lr.predict_probs(X)
#    Y_est = lr.predict(X)
#    # plot decision boundary
#    x1 = np.linspace(-3,10,100)
#    x2 = -1*(lr.theta[0]+x1*lr.theta[1])/lr.theta[2]
#    plt.plot(x2,x1,"g+")
#    plt.plot(X[Y=="n",1],X[Y=="n",2],"b+")
#    plt.plot(X[Y=="y",1],X[Y=="y",2],"c+")
#    plt.show()
    
        