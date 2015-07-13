# -*- coding: utf-8 -*-


import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import label_binariser as lb
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


    
def cost_grad(Theta,Y,X,k,weights, underflow_tol = 1e-20,  bias_term = True):
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
             
    underflow_tol: float
             Threshold for preventing underflow
             
    lambda_reg: float
             Tikhunov regularization parameter
    
    Returns:
    --------
    
    tuple(int, np.array): tuple, of size 2
            First element is value of cost function, second element is numpy array 
            of size 'm x 1', representing gradient
    '''
    n,m                   =  np.shape(X)
    Theta                 =  np.reshape(Theta,(m,k))                                         
    log_P                 =  log_softmax(Theta,X)
    unweighted            =  np.sum(Y*log_P, axis = 1)
    cost        =  -1.0*np.dot(weights,unweighted)
    resid       =  (Y-np.exp(log_P))
    grad        =  -1.0*np.dot(np.dot(X.T,np.diagflat(weights)),resid)
    return (cost, np.reshape(grad,(m*k,)))
    

#----------------------------------------  Softmax Regression  --------------------------------------------#
    
class SoftmaxRegression(object):
    '''
    Softmax classifier using l-bfgs-b optimization procedure.
    (Bias term is not added in the process of computation, so  it needs to be in
    data matrix)

    Parameters:
    -----------
    
    max_iter: int 
                Maximum nmber of iterations (default = 1000)
                
    tolerance: float 
                Precision threshold for convergence (default = 1e-10)
                
    underflow_tol: float (default: 1e-20)
                Threshold for preventing underflow
                
    '''
    
    def __init__(self,tolerance = 1e-5, max_iter = 80, underflow_tol = 1e-20):
        self.tolerance              = tolerance
        self.max_iter               = max_iter
        self.underflow_tol          = underflow_tol
        self.delta_param_norm       = 0
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
        # for soft splits in beginning of training in HME make parameters smaller
        self.theta = np.random.random([m,k])*(1e-2)
        
        
        
    def _pre_processing_targets(self,Y,k):
        ''' 
        Preprocesses data, transforms from vector Y to ground truth matrix Y
        '''
        self.binarisator = lb.LabelBinariser(Y,k)
        return self.binarisator.convert_vec_to_binary_matrix()

        
    def fit(self,Y_raw,X,k,weights, preprocess_input = False):
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
            
        preprocess_input: bool
            If true Y_raw is assumed to be vector, and is transformed into
            ground truth matrix of zeros and ones of size 'n x k', where k is
            number of classes
        
        '''
        if preprocess_input is True:
            Y            =  self._pre_processing_targets(Y_raw,k)
        else:
            Y            =  Y_raw
        fitter          = lambda theta: cost_grad(theta,Y,X,k,weights,
                                                  self.underflow_tol)
        # initialise parameters
        n,m             = np.shape(X)
        self.k          = k
        
        # save recovery paramters in case log-likelihood drops due to underflow
        theta_recovery  = self.theta
        log_like_before = self.log_likelihood(X,Y,weights,k)
        
        theta_initial   = np.random.random([m,k])*1e-3
        if self.theta is not None:
            theta_initial += self.theta
        
        # optimisation
        theta,J,D       = fmin_l_bfgs_b(fitter,
                                        theta_initial,
                                        fprime = None,
                                        pgtol = self.tolerance,
                                        approx_grad = False,
                                        maxiter = self.max_iter)
        self.theta      = np.reshape(theta,(m,k))
        
        # check behaviour of log-likelihood 
        log_like_after = self.log_likelihood(X,Y,weights,k)
        delta_log_like = log_like_after - log_like_before
        if delta_log_like < self.underflow_tol:
            self.theta = theta_recovery
            
        # l2 of delta params norm
        delta = self.theta - theta_recovery
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
        
    
    def predict(self,X_test):
        '''
        For each observation in X predicts class to which it belongs
        This method can be used if in 'fit' method preprocess_input was True
        
        Parameters:
        -----------
        
        X: numpy array of size 'unknown x m'
            Expalanatory variables
            
        Returns:
        --------
        
        prediction: numpy array of size 'unknown x m'
            Estimated target value for each observation
            
        '''
        p          = self.predict_probs(X_test)
        prediction = self.binarisator.convert_prob_matrix_to_vec(p)
        return prediction
        
        
    def log_likelihood(self,X,Y,weights,k, preprocess = False):
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
             
        preprocess: bool
             If True transforms vector to ground truth matrix
             
        Returns:
        --------
        
        weighted_log_likelihood: float
             log likelihood
        
        '''
        if preprocess is True:
            Y = self._pre_processing_targets(Y,k)
        weighted_log_likelihood = -1*cost_grad(self.theta,Y,X,k,weights,self.underflow_tol)[0]
        return weighted_log_likelihood
        
        
if __name__=="__main__":
    X = np.ones([100,3])
    X[:,1] = np.random.normal(0,1,100)
    X[:,2] = np.random.normal(0,1,100)
    X[25:50,1:3] = X[25:50,1:3] + 10
    X[50:75,1:3] = X[50:75,1:3] + 20
    X[75:100,1:3] = X[75:100,1:3] + 30
    Y = np.zeros([100,1])
    Y[25:50,0] = 1
    Y[50:75,0] = 2
    Y[75:100,0] = 3
    sr = SoftmaxRegression()
    sr.init_params(3,4)
    sr.fit(Y[:,0],X,4,np.ones(100),True)
    Y_pred = sr.predict(X)
    print "diff"
    print np.sum(Y_pred!=Y[:,0])
    from statsmodels.discrete import discrete_model
    #result = discrete_model.MNLogit(Y[:,0],X).fit()
    #print result.summary()
    
    X = np.random.random([8,2])
    Theta = np.random.random([2,3])
    
#    def check(X,Theta):
#        l1 = log_softmax(Theta,X)
#        l2 = softmax(Theta,X)
#        print np.exp(l1) - l2

