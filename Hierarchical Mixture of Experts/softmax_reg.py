# -*- coding: utf-8 -*-

# THINGS TO DO

# TODO: Implement function analytically computing hessian for softmax regression 
#       without overparametrisation (for newton-cg)
# 


import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import logsumexp



def log_softmax(Theta,X):
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
     probs: numpy array of size 'n x k'
             Matrix of probabilities

    '''
    n,m           = np.shape(X)
    m,k           = np.shape(Theta)
    X_Theta       = np.dot(X,Theta)
    norm          = logsumexp(X_Theta, axis = 1)
    norm          = np.outer(norm, np.ones(k))
    log_softmax   = X_Theta - norm
    return log_softmax


    
def cost_grad(Theta,Y,X,k,weights):
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

    Returns:
    --------
    
    tuple(int, np.array): tuple, of size 2
            First element is value of cost function, second element is numpy array 
            of size 'm x 1', representing gradient
    '''
    n,m                   =  np.shape(X)
    Theta                 =  np.reshape(Theta,(m,k-1))
    Theta                 =  np.concatenate([np.zeros([m,1]),Theta], axis = 1)
    log_P                 =  log_softmax(Theta,X)
    unweighted            =  np.sum(Y*log_P, axis = 1)
    cost                  =  -1.0*np.dot(weights,unweighted)
    resid                 =  (Y-np.exp(log_P))
    X_w                   =  X*np.outer(weights, np.ones(m))
    grad                  =  -1.0*np.dot(X_w.T,resid)
    grad                  =  grad[:,1:]
    # use no.array(np.reshape()), otherwise l_bfgs_b have strange 
    # initialisation error for FORTRAN code
    return (cost, np.array(np.reshape(grad,(m*(k-1),))))
    
    
#TODO: analytical calculation of hessian for faster convergence 
def cost_grad_hess(Theta,Y,X,k,weights, bias_term = True):
    pass
    

#----------------------------------------  Softmax Regression  --------------------------------------------#
    
class SoftmaxRegression(object):
    '''
    Softmax classifier using l-bfgs-b optimization procedure.
    (Bias term is not added in the process of computation, so  it needs to be in
    data matrix). This implementation of softmax regression does not suffer from
    overparametrization (so it has unique soltuion), however in case of complete 
    separability will have the same problem as logistic regression (norm of coefficient 
    going to infinity, we)

    Parameters:
    -----------
    
    max_iter: int 
                Maximum nmber of iterations (default = 1000)
                
    tolerance: float 
                Precision threshold for convergence (default = 1e-10)
                
    stop_learning: float (default: 1e-5)
                If change in weighted log-likelihood is below stop_learning
                threshold then new parameters are discarded and model in hme will
                use old ones
                
    '''
    
    def __init__(self,tolerance = 1e-5, max_iter = 20, stop_learning = 1e-5):
        self.tolerance              = tolerance
        self.max_iter               = max_iter
        self.stop_learning          = stop_learning
        self.delta_param_norm       = 0
        self.delta_log_like         = 0
        self.theta                  = None
        

        
    def init_params(self,m,k):
        '''
        
        Parameters:
        -----------
        m: int
           Dimensionality of data        
        
        k: int
           Number of classes in classification problem
        '''
        self.m, self.k = m,k
        # for soft splits in beginning of training in HME make parameters smaller
        self.theta      = np.random.random([m,k])*0.1
        # restrict paramters so that softmax regression will not be overparamerised
        self.theta[:,0] = np.zeros(m)
        

    def fit(self,Y,X,weights):
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

        '''
        # initialise parameters
        n,m             = np.shape(X)
        k               = self.k
        
        # initiate parameters for fitting (avoids overparametarization)
        theta_initial   = np.zeros([m,k-1])
        if self.theta is None:
            self.init_params(m,k)
        
        # Use previously fitted values for refitting, if weights in HME changed a 
        # little this will provide much faster convergence since initialised parameters 
        # will be near optimal point.
        theta_initial  += self.theta[:,1:]
        
        # save recovery paramters in case log-likelihood drops due to underflow
        theta_recovery  = self.theta
        log_like_before = self.log_likelihood(X,Y,weights)
        
        # optimisation with lbfgsb
        fitter          = lambda theta: cost_grad(theta,Y,X,k,weights)
        theta,J,D       = fmin_l_bfgs_b(fitter,
                                        theta_initial,
                                        fprime = None,
                                        pgtol = self.tolerance,
                                        approx_grad = False,
                                        maxiter = self.max_iter)
        
        # theta with dimensionality m x k-1 
        theta           = np.reshape(theta,(m,k-1))
        
        # transform to standard softmax representattion with m x k dimensionality
        self.theta      = np.concatenate([np.zeros([m,1]), theta], axis = 1)
        
        # check behaviour of log-likelihood 
        log_like_after = self.log_likelihood(X,Y,weights)
        delta_log_like = (log_like_after - log_like_before) / n
        
        # Code below is for two following cases:
        #
        # CASE 1: 
        #         In process of fitting deep HME due to errors in floating point
        #         operations and underflows, when weights change is small 
        #         ( errors seem to start when total change in weights is 1e-30 and smaller)
        #         log-likelihood of model after refitting can be smaller than before.
        #         If that happens then model uses old parameters instead of new
        #
        # CASE 2:  
        #         Softmax regression suffers from the same
        #         drawback as logistic regression, in case of perfect
        #         or near perfect separability norm of parameters keep increasing ( basically
        #         multiplying optimal w by constant). In that case change in parameters does 
        #         not decrease, while change in log-likelihood is tiny.
        #
        if delta_log_like < self.stop_learning:
            self.theta     = theta_recovery
            delta_log_like = 0
            
        # save changes in likelihood and parameters
        delta = self.theta - theta_recovery
        self.delta_log_like   = delta_log_like
        self.delta_param_norm = np.sum(np.dot(delta.T,delta))
        
        
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
        log_P = log_softmax(self.theta,X_test)
        P     = np.exp(log_P)
        return P
        
        
    def predict_log_probs(self,X_test):
        '''
        Calculates matrix of log probabilities
        
        Parameters:
        -----------
        
        X_test: numpy array of size 'uknown x m' 
             Explanatory variables of test set
        
        Returns:
        -------
                 
        log_P: numpy array of size 'uknown x k'
             Matrix of log probabilities
             
        '''
        log_P = log_softmax(self.theta,X_test)
        return log_P
        
        
    def log_likelihood(self,X,Y,weights):
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
             
        Returns:
        --------
        
        weighted_log_likelihood: float
             log likelihood
        
        '''
        weighted_log_like = -1*cost_grad(self.theta[:,1:],Y,X,self.k,weights)[0]
        return weighted_log_like
        
        
    def posterior_log_probs(self,X,Y):
        '''
        Calculates probability of observing Y given X and parameters (for HME usage)
        '''
        log_p = np.sum(Y*log_softmax(self.theta,X), axis = 1)
        return log_p

        
if __name__ == "__main__":
    X      = np.ones([30000,3]) 
    X[:,0] = np.random.normal(0,1,30000)
    X[:,1] = np.random.normal(0,1,30000)
    X[10000:20000,0:2] = X[10000:20000,0:2]+10
    X[20000:30000,0:2] = X[20000:30000,0:2]+15
    Y = np.zeros([30000,3])
    Y[10000:20000,0] = 1
    Y[0:10000,1] = 1
    Y[20000:30000,2] = 1
    sr = SoftmaxRegression()
    sr.init_params(3,3)
    weights = np.ones(30000)/3
    sr.fit(Y,X,weights)
    Y_hat = sr.predict_probs(X)
    
