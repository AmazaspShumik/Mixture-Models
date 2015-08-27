# -*- coding: utf-8 -*-
"""
Weighted Linear Regression , Expert in HME model

   m - dimensionality of input (i.e. length of row in matrix X)
   n - number of observations
"""

import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_triangular

#------------------------------------ Least Squares Solvers-------------------------------#

def cholesky_solver_least_squares(part_one, part_two):
    '''
    Solves least squares problem using cholesky decomposition
    
    Parameters:
    -----------
    
    part_one: numpy array of size 'm x m', 
              Equals X.T * X
    part_two: numpy array of size 'm x 1'
              Equals X.T * Y
              
    Returns:
    --------
    Theta: numpy array of size 'm x 1'
              Vector of coefficients
    
    '''
    # R*R.T*Theta = part_two
    R = np.linalg.cholesky(part_one)
    # R*Z = part_two
    Z     = solve_triangular(R,part_two, check_finite = False, lower = True)
    # R.T*Theta = Z
    Theta = solve_triangular(R.T,Z, check_finite = False, lower = False)
    return Theta
    
    
def qr_solver(Q,R,Y):
    '''
    Solves least squares using qr decomposition.
    
    Parameters:
    -----------
    
    Q: numpy array of size 'n x m'
        Matrix Q in QR decomposition (Matrix of orthonormal vectors)
        
    R: numpy array of size 'm x m'
        Matrix R in QR decomposition (Matrix of projection coefficients on 
        orthonormal vectors)
        
    Y: numpy array of size ' n x 1'
        Vector of dependent variables
        
    Returns:
    -------
    Theta: numpy array of size 'm x 1'
         Vector of parameters
    '''
    qy      = np.dot(Q.T,Y)
    Theta   = solve_triangular(R,qy, check_finite = False, lower = False)
    return  Theta
   
   
def lstsq_wrapper(y,X):
    '''
    Uses C++ Linear Algebra Package to calculate coefficients and residuals
    of regression. Is much faster than other methods, since it calls C++ functions.
    
    Parameters:
    -----------
    
    Y: numpy array of size 'n x 1'
        Vector of dependent variables
        
    X: numpy array of size 'n x m'
        Explanatory variables
    '''
    theta,r,rank,s = np.linalg.lstsq(X,y)
    return theta
    
    
#----------------------------------
    
def norm_pdf_log_pdf(theta,y,x,sigma_2):
    '''
    Calculates probability of observing Y given Theta and sigma and explanatory
    varibales
    
    Parameters:
    ----------
    
    Theta: numpy array of size 'm x k', 
           Matrix of parameters
    Y: numpy array of size 'n x 1'
           Vector of dependent variables
    X: numpy array of size 'n x m'
           Matrix of inputs 
    sigma_2: numpy array of size 'm x 1'
           Vector of variances
    
    Returns:
    -------
    prob: numpy array of size 'n x 1'
          Probability of observing y given theta and X
    
    '''
    u              = y - np.dot(x,theta)
    log_normaliser = -1* np.log(np.sqrt(2*np.pi*sigma_2))
    log_main       = -u*u/(2*sigma_2)
    log_pdf        = log_normaliser + log_main
    prob           = np.exp(log_pdf)
    return [log_pdf,prob]
        
        
    
#------------------------------------- Weighted Linear Regression-------------------------#

class WeightedLinearRegression(object):
    '''
    Weighted Linear Regression
            
    Parameters:
    -----------
    
    solver: string (default = "qr")
         Numerical method to find weighted linear regression solution
         
    underflow_tol: float (default = 1e-20)
         Threshold to prevent underflow in likelihood computation
         
    '''
    
    def __init__(self, solver = "qr", stop_learning = 1e-3):
        self.solver              = solver
        self.theta               = None             
        self.var                 = 0               
        self.stop_learning       = stop_learning
        self.delta_param_norm    = 0
        self.deta_log_like       = 0


    def init_params(self,m):
        '''
        Initialises weights and preallocates memory
        
        Parameters:
        ----------
        m: int
           Number of parameters, should equal to dimensionality of data
           
        '''
        self.theta = np.random.normal(0,1,m)
        self.var   = 1 


    def fit(self,Y,X,weights = None):
        ''' 
        Fits weighted regression, updates coefficients and variance
        
        Parameters:
        -----------
        
        X: numpy array of size 'n x m'
             Explanatory variables
        
        Y: numpy array of size 'n x 1'
             Target variable can take only values 0 or 1
         
        weights: numpy array of size 'n x 1'
             Weights for observations
        
        '''
        n,m         =  np.shape(X)
        if weights is not None:
             w            =  np.sqrt(weights)
        else:
             w            =  np.ones(n)
             weights      =  w
        X_w          =  (X.T*w).T
        Y_w          =  Y*w
         
        if self.theta is None:
            self.init_params(m)
             
        # save paramters in case log-likelihood drops ( PRECISION ISSUE IN 
        # DEEP HIERARCHICAL MIXTURE OF EXPERTS)
        theta_recovery  =  self.theta
        var_recovery    =  self.var
        log_like_before =  self.log_likelihood(X,Y,weights)
        
        # use cholesky decomposition for least squares 
        if self.solver  == "cholesky":
           part_one     =  np.dot(X_w.T,X_w)
           part_two     =  np.dot(X_w.T,Y_w)
           self.theta   =  cholesky_solver_least_squares(part_one, part_two)
           
        # use qr decomposition for least squares
        elif self.solver == "qr":
            Q,R        = np.linalg.qr(X_w)
            self.theta = qr_solver(Q,R,Y_w)
            
        # lapack least squares solver
        elif self.solver == "lapack_solver":
            self.theta = lstsq_wrapper(Y_w,X_w)
            
        # calculate variances 
        vec_1          =  (Y_w - np.dot(X_w,self.theta))
        self.var       =  np.dot(vec_1,vec_1)/np.sum(weights)
        
        # if likelihood dropped ( PRECISION ISSUE) use recovery parameters
        # used in DEEP HIERARCHICAL MIXTURE OF EXPERTS
        log_like_after = self.log_likelihood(X,Y,weights)
        delta_log_like = ( log_like_after - log_like_before)/n
        if delta_log_like < self.stop_learning:
            self.theta = theta_recovery
            self.var   = var_recovery
            delta_log_like = 0
            
        # save change in parameters and likelihood
        delta                 = self.theta - theta_recovery
        self.delta_param_norm = np.sum(np.dot(delta.T,delta))
        self.delta_log_like   = delta_log_like
        
        
        
    def predict(self,X):
        '''
        Calculates point estimator based on learned parameters
        
        Parameters:
        -----------
        X: numpy array of size 'n x m'
             Explanatory variables
             
        Returns:
        --------
        X: numpy array of size 'unknown x 1'
           Explanatory variables from test set
        
        '''
        return np.dot(X,self.theta)
        
        
    def posterior_log_probs(self,X,Y):
        ''' 
        Wrapper for norm_pdf (primarily used in HME)
        '''
        log_pdf,pdf = norm_pdf_log_pdf(self.theta,Y,X,self.var)
        return log_pdf
        
        
    def log_likelihood(self,X,Y,weights = None):
        '''
        Returns log likelihood for linear regression with noise distributed 
        as Gaussian
        
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
             Log likelihood

        '''
        if weights is None:
            weights = np.ones(X.shape[0])
        log_pdf, pdf        = norm_pdf_log_pdf(self.theta,Y,X,self.var)
        log_likelihood      = np.sum(weights*log_pdf)
        return log_likelihood
        
        
    def posterior_cdf(self,X,y_lo = None,y_hi = None):
        ''' 
        Calculate probability of observing target variable in range [y_lo, y_hi]
        given explanatory variable and parameters
        
        Parameters:
        -----------
        X: numpy array of size 'unknown x n'
           Explanatory variables
           
        y_lo: numpy array of size 'unknown x 1'
           Lower bound 
           
        y_hi: numpy array of size 'unknown x 1'
           Upper bound
           
        Returns:
        --------
        delta_prob: numpy array of size 'unknown x 1'
            Probability of observing Y in range [y_lo, y_hi]
        '''
        # check that upper bound is higher than lower bound
        assert np.sum(y_hi<y_lo) == 0, "upper bound can not be smaller than lower bound"
        # calculate difference in cdfs
        means       = self.predict(X)
        std         = np.sqrt(self.var)
        upper_bound = 0
        lower_bound = 0
        if y_hi is not None:
           upper_bound = norm.cdf(y_hi,loc = means, scale = std)
        if y_lo is not None:
           lower_bound = norm.cdf(y_lo, loc = means, scale = std)
        delta_prob  = abs(upper_bound - lower_bound)
        return delta_prob



        