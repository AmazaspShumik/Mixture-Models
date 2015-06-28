# -*- coding: utf-8 -*-

"""
     K - number of first levels gating options
     P - number of second level gating options
     n - number of observations
     m - dimensionality of data
"""

import SoftmaxRegression as sr
import numpy as np
import abc



    
def bounded_variable(x,lo,hi):
    '''
    Returns 'x' if 'x' is between 'lo' and 'hi', 'hi' if x is larger than 'hi'
    and 'lo' if x is lower than 'lo'
    '''
    if   x > hi:
        return hi
    elif x < lo:
        return lo
    else:
        return x



class AbstractHME(object):
    __metaclass__ = abc.ABCMeta
    '''
    Three level hierarchical mixture of experts model.
    
    This HME model consist of:  
                               Level 1 - softmax gating function
                               Level 2 - softmax gating functions
                               Level 3 - linear regression
    
    Parameters:
    -----------
    
    Y:  numpy array of size 'n x 1' 
         Vector of dependent variables
         
    X:  numpy array of size 'n x m'
         Matrix of inputs
         
    n_gates_first: int
         Number of gates for first level gating network
         
    n_gates_second: int 
         Number of gates for second level gating network
         
    error_bound_resp: float
         Accuracy parameter to prevent numerical underflow, when calculating responsibilities
    
    converge: float
         Threshold for convergence (if proportional change in lower 
                               bound is smaller then threshold then algorithm is stopped)
    max_iter: int
         Maximum number of iteration of EM algorithm
         
    verbose: boolean
         If True prints iteration number and value of lower bound at each iteration
    
    '''
    
    
    def __init__(self,Y,X,n_gates_first, clusters,
                                         error_bound_resp = 1e-10,
                                         max_iter         = 100, 
                                         converge         = 1e-6, 
                                         verbose          = True):
        ''' Initialise '''
        self.Y                     = Y               
        self.X                     = X               
        self.n,self.m              = np.shape(X)
        self.n_gates_first         = n_gates_first   
        self.clusters              = clusters
        # parameters for first gating network
        self.alpha                 = np.random.random([self.m,self.n_gates_first])
        # parameters for second gating network
        self.beta                  = [np.random.random([self.m,self.clusters[i]]) for i in range(self.n_gates_first)]
        # responsibilities
        self.responsibilities      = []
        for i in range(self.n):
            self.responsibilities.append([np.random.random(self.clusters[j]) for j in range(self.n_gates_first)])
        # accuracy threshold to prevent underflow in responsiilities calculation
        self.error_bound_resp      = error_bound_resp
        # list of lower bounds of log-likelihood of hme model
        self.lower_bounds          = []
        self.convergence_threshold = converge
        self.max_iter              = max_iter
        self.verbose               = verbose
        
        
    def iterate(self):
        delta = 1
        for i in range(self.max_iter):
            self._e_step()
            if len(self.lower_bounds) >= 2:
                delta = float(self.lower_bounds[-1] - self.lower_bounds[-2])/abs(self.lower_bounds[-2])
            if delta > self.convergence_threshold:
                self._m_step()
                if self.verbose:
                    iteration_verbose = "iteration {0} completed, lower bound of log-likelihood is {1} "
                    print iteration_verbose.format(i,self.lower_bounds[-1])
            else:
                print "algorithm converged"
                break
            
    #--------------------------------------------  E-step --------------------------------------------------#

    def _e_step(self):
        '''
        
        E-step in EM algorithm for training Hierarchical Mixture of Experts.
        Finds posterior probability of latent variable and lower bound of 
        log-likelihood.
        
        Calculates responsibilities (i.e. posterior probabilities of latent variables)
        and lower bound of log-likelihoood of model
        '''
        # lower bound of log-likelihood function
        lower_bound       = 0.0
        
        # calculate posterior probability of first gating network , dim = N x K
        resp_gates_first  = sr.softmax(self.alpha,self.X)                                                  
        
        # calculate posterior probability of second gating network given latent variable for first gate 
        # dim = [[N x P] x K]
        resp_gates_second = [sr.softmax(self.beta[i], self.X) for i in range(self.n_gates_first)]          
        
        # calculate posterior probability of experts, given latent variables for first and second gates, 
        # dim = [[N x P] x K]
        resp_experts  = self._expert_probabilities()
        
        # calculate responsibilities and lower bound of likelihood function        
        for n in range(self.n):
            for i in range(self.n_gates_first):
                for j in range(self.clusters[i]):
                    
                    # prevent underflow & overflow for expert network
                    expert      = bounded_variable(resp_experts[i][n,j],
                                                   self.error_bound_resp,
                                                   1-self.error_bound_resp)
                    
                    # prevent underflow & overflow for first level gating network
                    gate_first  = bounded_variable(resp_gates_first[n,i],
                                                   self.error_bound_resp,
                                                   1-self.error_bound_resp)
                    
                    # prevent underflow & overflow for second level gating network
                    gate_second = bounded_variable(resp_gates_second[i][n,j],
                                                   self.error_bound_resp,
                                                   1-self.error_bound_resp)
                    
                    # update lower bound
                    lower_bound += (np.log(expert)+np.log(gate_first)+np.log(gate_second))*self.responsibilities[n][i][j]
                    
                    # calculates p(Z_1 , Z_2 | X, Y)*P(Y|X)                    
                    self.responsibilities[n][i][j] = expert*gate_first*gate_second   # ??? if use lower_bound loose accuracy???
            #  normaliser = p(Y | X) , sum over both latent variables
            normaliser = sum([np.sum(e) for e in self.responsibilities[n]])
            # responsibility p(Z_1 , Z_2 | X, Y)
            self.responsibilities[n] /= normaliser
        self.lower_bounds.append(lower_bound)
        
        
    @abc.abstractmethod
    def _expert_probabilities(self):
        '''
        Calculates probabilities from experts.
        
        Abstract method that should have concrete implementation in subclasses
        '''
        raise NotImplementedError
        
    #--------------------------------------------  M-step --------------------------------------------------#
        

    def _m_step(self):
        '''
        M-step in EM algorithm for training Hierarchical Mixture of Experts.
        Finds parameters for gating networks and experts that maximise lower bound of 
        log-likelihood.
        '''
        self._m_step_expert_network()
        self._m_step_gating_networks()
            

    @abc.abstractmethod
    def _m_step_expert_network(self):
        '''
        M-step for expert network.
        
        Abstract method that should have concrete implementation in subclass
        '''
        raise NotImplementedError
                
    
    def _m_step_gating_networks(self):
        '''
        Chooses parameters that maximise lower bound by running softmax regression
        for first level gating network and weighted softmax regression for second 
        level gating network
        
        '''
        # first level gating network optimization
        H_first          = np.array([ [np.sum(self.responsibilities[i][j]) 
                                                        for j in range(self.n_gates_first)] 
                                                        for i in range(self.n)])
        gate_first_level = sr.SoftmaxRegression()
        gate_first_level.fit_matrix_output(H_first,self.X, np.ones(self.n))
        
        # second level gating network optimization        
        for i in range(self.n_gates_first):
            H_second = []
            weights = np.zeros(self.n)
            for n in range(self.n):
                H_second.append(self.responsibilities[n][i]/H_first[n,i])
                weights[n]        = H_first[n,i]
            gate_second_level = sr.SoftmaxRegression()
            gate_second_level.fit_matrix_output(np.array(H_second),self.X, weights)
            self.beta[i]      = gate_second_level.theta

    #--------------------------------- Predictions ----------------------------------------------#

    def predict_mean(self,X):
        ''' 
        Finds mean prediction of experts
        
        Parameters:
        -----------
        
        X: numpy array of size 'unknown x m'
            Test data, for which estimate of target values need to be found
            
        Returns:
        --------
        
        prediction: numpy array of size 'unknown x 1'
            Estimated target values
        '''
        pred_gate_one = sr.softmax(self.alpha,X)
        prediction    = np.zeros(self.n)
        for i in range(self.n_gates_first):
            pred_expert   = self._predict_expert(X,i)
            pred_gate_two = sr.softmax(self.beta[i],X)
            prediction   += pred_gate_one[:,i]*np.sum( pred_expert * pred_gate_two ,axis = 1)
        return prediction
        
        
    @abc.abstractmethod
    def _predict_expert(self,X):
        '''
        Point prediction by expert model.
        
        Abstract method that shoul have concrete implementation in subclass 
        '''
        raise NotImplementedError
        
        
    def posterior_dist(self,X,resolution = 0.01):
        pass
    
    
