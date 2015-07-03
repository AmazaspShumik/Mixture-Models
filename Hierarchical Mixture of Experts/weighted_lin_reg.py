# -*- coding: utf-8 -*-
"""
Weighted Linear Regression , Expert in HME model

   m - dimensionality of input (i.e. length of row in matrix X)
   n - number of observations
"""

import numpy as np

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
    Z     = np.linalg.solve(R,part_two)
    # R.T*Theta = Z
    Theta = np.linalg.solve(R.T,Z)
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
    Theta   = np.linalg.solve(R,qy)
    return  Theta
    
#----------------------------------
    
def norm_pdf(theta,y,x,sigma_2):
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
    
    Output:
    -------
    prob: numpy array of size 'n x 1'
          Probability of observing y given theta and X
    
    '''
    normaliser = 1.0/np.sqrt(2*np.pi*sigma_2)
    u          = y - np.dot(x,theta)
    prob       = normaliser* np.exp( -0.5 * u*u / sigma_2 )
    return prob
    
#def hme_pdf_wrapper(theta,y,)
    
    
#------------------------------------- Weighted Linear Regression-------------------------#

class WeightedLinearRegression(object):
    '''
    Weighted Linear Regression
            
    Parameters:
    -----------
    solver: string
         Numerical method to find weighted linear regression solution
    '''
    
    def __init__(self, solver = "qr"):
        self.solver       = solver
        self.theta        = 0             # coefficients excluding bias term
        self.mse          = 0             # mean squared error
        self.var          = 0             # fitted variance


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


    def fit(self,X,Y,weights):
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
        W           =  np.diagflat(weights)
        # use cholesky decomposition for least squares 
        if self.solver == "cholesky":
           part_one    =  np.dot(np.dot(X.T,W),X)
           part_two    =  np.dot(np.dot(X.T,W),Y)
           self.theta  =  cholesky_solver_least_squares(part_one, part_two)
        # use qr decomposition for least squares
        elif self.solver == "qr":
            X_tilda    = np.dot(X.T,np.sqrt(W)).T
            Y_tilda    = Y*np.sqrt(weights)
            Q,R        = np.linalg.qr(X_tilda)
            self.theta = qr_solver(Q,R,Y_tilda)
        # calculate variances 
        vec_1       =  (Y - np.dot(X,self.theta))
        self.var    =  np.dot(vec_1,np.dot(vec_1,W))/np.sum(W)
        
        
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
        


