# -*- coding: utf-8 -*-
"""


     K - number of first levels gating options
     P - number of second level gating options
     n - number of observations
     m - dimensionality of data

"""

import SoftmaxRegression as sr
import WeightedLinearRegression as wlr
import numpy as np



class HME_Gaussian(object):
    '''
    Three level hierarchical mixture of experts model.
    
    This HME model consist of:  
                               Level 1 - softmax gating function
                               Level 2 - softmax gating functions
                               Level 3 - linear regression
    
    Input:
    -------
    Y                        - numpy array of size 'n x 1', vector of dependent variables
    X                        - numpy array of size 'n x m', matrix of inputs
    n_gates_first            - number of gates for first level gating network
    n_gates_second           - number of gates for second level gating network
    error_bound_resp         - accuracy parameter to prevent numerical underflow, when
                               calculating responsibilities
    converge                 - threshold for convergence (if proportional change in lower 
                               bound is smaller then threshold then algorithm is stopped)
    max_iter                 - int, maximum number of iteration of EM algorithm
    verbose                  - if True prints iteration number and value of lower bound at each iteration
    
    '''
    
    
    def __init__(self,Y,X,n_gates_first, n_gates_second,
                                         error_bound_resp = 1e-10,
                                         max_iter         = 100, 
                                         converge         = 1e-6, 
                                         verbose          = True):
        ''' Initialise '''
        self.Y                     = Y               
        self.X                     = X               
        self.n,self.m              = np.shape(X)
        self.n_gates_first         = n_gates_first   
        self.n_gates_second        = n_gates_second
        # parameters for first gating network
        self.alpha                 = np.random.random([self.m,self.n_gates_first])
        # parameters for second gating network
        self.beta                  = np.random.random([self.n_gates_first,self.m,self.n_gates_second])
        # coefficients of linear regression
        self.gamma                 = np.random.random([self.n_gates_first,self.m,self.n_gates_second])
        # variance for error term of regression
        self.sigma_2               = np.ones([self.n_gates_second,self.n_gates_first])
        # responsibilities
        self.responsibilities      = np.random.random([self.n,self.n_gates_first,self.n_gates_second])
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
            self.e_step()
            if len(self.lower_bounds) >= 2:
                delta = float(self.lower_bounds[-1] - self.lower_bounds[-2])/abs(self.lower_bounds[-2])
            if delta > self.convergence_threshold:
                self.m_step()
                if self.verbose:
                    print "iteration {0} completed, lower bound of log-likelihood is {1} ...".format(i,self.lower_bounds[-1])
            else:
                print "algorithm converged"
                break
            
            
    #----------------------------------------  E-step ----------------------------------------------#


    def e_step(self):
        '''
        E-step in EM algorithm for training Hierarchical Mixture of Experts.
        Finds posterior probability of latent variable and lower bound of 
        log-likelihood.
        '''
        lower_bound,self.responsibilities = self._responsibilities_likelihood_compute()
        self.lower_bounds.append(lower_bound)


    def _responsibilities_likelihood_compute(self):
        '''
        Calculates responsibilities (i.e. posterior probabilities of latent variables)
        and lower bound of log-likelihoood of model
        '''
        # lower bound of log-likelihood function
        lower_bound       = 0.0
        responsibilities  = [np.zeros([self.n_gates_first,self.n_gates_second]) for i in range(self.n)]
        
        # calculate posterior probability of first gating network , dim = N x K
        resp_gates_first  = sr.softmax(self.alpha,self.X)                                                  
        
        # calculate posterior probability of second gating network given latent variable for first gate 
        # dim = [[N x P] x K]
        resp_gates_second = [sr.softmax(self.beta[i], self.X) for i in range(self.n_gates_first)]          
        
        # calculate posterior probability of experts, given latent variables for first and second gates, 
        # dim = [[N x P] x K]
        resp_experts      = [np.zeros([self.n,self.n_gates_second]) for i in range(self.n_gates_first)]        
        for i in range(self.n_gates_first):
            for j in range(self.n_gates_second):
                resp_experts[i][:,j] = wlr.norm_pdf(self.gamma[i][:,j],self.Y,self.X,self.sigma_2[j,i])
        
        # calculate responsibilities and lower bound of likelihood function        
        for n in range(self.n):
            for i in range(self.n_gates_first):
                for j in range(self.n_gates_second):
                    
                    # prevent underflow & overflow for expert network
                    expert      = self.bounded_variable(resp_experts[i][n,j],
                                                        self.error_bound_resp,
                                                        1-self.error_bound_resp)
                    
                    # prevent underflow & overflow for first level gating network
                    gate_first  = self.bounded_variable(resp_gates_first[n,i],
                                                        self.error_bound_resp,
                                                        1-self.error_bound_resp)
                    
                    # prevent underflow & overflow for second level gating network
                    gate_second = self.bounded_variable(resp_gates_second[i][n,j],
                                                        self.error_bound_resp,
                                                        1-self.error_bound_resp)
                    
                    # update lower bound
                    lower_bound += (np.log(expert)+np.log(gate_first)+np.log(gate_second))*self.responsibilities[n][i,j]                    
                    
                    # calculates p(Z_1 , Z_2 | X, Y)*P(Y|X)                    
                    responsibilities[n][i,j] = expert*gate_first*gate_second
            #  normaliser = p(Y | X) , sum over both latent variables
            normaliser = np.sum(responsibilities[n])
            # responsibility p(Z_1 , Z_2 | X, Y)
            responsibilities[n] /= normaliser
        return [lower_bound,responsibilities]
            
            
    #--------------------------------------- M-step ---------------------------------------------#


    def m_step(self):
        '''
        M-step in EM algorithm for training Hierarchical Mixture of Experts.
        Finds parameters for gating networks and experts that maximise lower bound of 
        log-likelihood.
        '''
        self._m_step_expert_network()
        self._m_step_gating_networks()
            

    def _m_step_expert_network(self):
        '''
        Weighted Linear regression for optimising parameters of experts.
        '''
        for i in range(self.n_gates_first):
            for j in range(self.n_gates_second):
                weights = []
                for n in range(self.n):
                    weights.append(self.responsibilities[n][i,j])
                W      = np.array(weights) 
                expert = wlr.WeightedLinearRegression(self.X,self.Y,W)
                expert.fit()
                self.sigma_2[j,i]   = expert.var
                self.gamma[i][:,j]  = expert.theta
                
    
    def _m_step_gating_networks(self):
        '''
        Chooses parameters that maximise lower bound by running softmax regression
        for first level gating network and weighted softmax regression for second 
        level gating network
        
        '''
        # first level gating network optimization
        H_first          = np.array([np.sum(self.responsibilities[i],axis = 1) for i in range(self.n)])
        gate_first_level = sr.SoftmaxRegression()
        gate_first_level.fit_matrix_output(H_first,self.X, np.ones(self.n))
        
        # second level gating network optimization
        H_second = np.zeros([self.n, self.n_gates_second])
        for i in range(self.n_gates_first):
            weights = np.zeros(self.n)
            for n in range(self.n):
                H_second[n,:]     = self.responsibilities[n][i,:]/H_first[n,i]
                weights[n]        = H_first[n,i]
            gate_second_level = sr.SoftmaxRegression()
            gate_second_level.fit_matrix_output(H_second,self.X, weights)
            self.beta[i]      = gate_second_level.theta


    def predict_mean(self,X):
        ''' Finds mean prediction of experts '''
        pred_gate_one = sr.softmax(self.alpha,X)
        prediction    = np.zeros(self.n)
        for i in range(self.n_gates_first):
            pred_expert   = np.dot(X,self.gamma[i])
            pred_gate_two = sr.softmax(self.beta[i],X)
            prediction   += pred_gate_one[:,i]*np.sum( pred_expert * pred_gate_two , axis = 1)
        return prediction
    
    
    #------------------------ Helper Methods ---------------------------------#
    
    @staticmethod
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
            

if __name__=="__main__":
    X = np.ones([6000,2])
    X[:,0] = np.linspace(0,6,6000)
    Y = np.zeros(6000)
    Y = 10*np.sin(X[:,0])+np.random.normal(0,1,6000)
    hme = HME_Gaussian(Y,X,12,8)
    hme.iterate()
    Y_hat = hme.predict_mean(X)

    
    