# -*- coding: utf-8 -*-
"""


     K - number of first levels gating options
     P - number of second level gating options
     N - number of observations
     

"""

import SoftmaxRegression as sr
import WeightedLinearRegression as wlr
import numpy as np




class HME_Gaussian(object):
    '''
    Three level hierarchical mixture of experts model,
    
    This HME model consist of:  
                               Level 1 - softmax gating function
                               Level 2 - softmax gating functions
                               Level 3 - linear regression
    
    '''
    
    
    def __init__(self,Y,X,n_gates_first, n_gates_second):
        self.Y                = Y               # target variables
        self.X                = X               # explanatory variables
        self.n,self.m         = np.shape(X)
        self.n_gates_first    = n_gates_first
        self.n_gates_second   = n_gates_second
        # parameters for first gating network
        self.alpha            = np.zeros([self.m,self.n_gates_first])
        # parameters for second gating network
        self.beta             = [np.zeros([self.m,self.n_gates_second]) for i in range(self.n_gates_first)]
        # coefficients of linear regression
        self.gamma            = [np.zeros([self.m,self.n_gates_second]) for i in range(self.n_gates_first)]
        # variance for error term of regression
        self.sigma_2          = np.ones([self.n_gates_second,self.n_gates_first])
        # responsibilities
        self.responsibilities = [np.zeros([self.n_gates_first,self.n_gates_second]) for i in range(self.n)]
        
        
    def e_step(self):
        '''
        E-step in EM algorithm for training Hierarchical Mixture of Experts
        '''
        # calculate posterior probability of first gating network
        resp_gates_first    = sr.softmax(self.alpha,self.X)                                                  # dim = N x K
        # calculate posterior probability of second gating network
        resp_gates_second   = [sr.softmax(self.beta[i], self.X) for i in range(self.n_gates_first)]          # dim = [[N x P] x K]
        # calculate posterior probability of expert model
        resp_experts        = [wlr.norm_matrix_pdf(self.gamma[i],self.Y,self.X,self.sigma_2[:,i]) for i in range(self.n_gates_first)] # dim = [[N x P] x K]
        for i in range(self.n):
            # for expert & second level gating get i-th row of each matrix in corresponding responsibility list
            expert_n     = np.array([list(resp_expert[i,:]) for resp_expert in resp_experts])                # rows = K, columns = P
            gate_two_n   = np.array([list(resp_gate_second[i,:]) for resp_gate_second in resp_gates_second]) # rows = K, columns = P
            gate_first_n = np.outer(resp_gates_first[i,:], np.ones(self.n_gates_second))                     # row  = K, column  = 1
            self.responsibilities[i] = expert_n*gate_two_n*gate_first_n
            self.responsibilities[i] = self.responsibilities[i]/np.sum(self.responsibilities)
    

    def m_step(self):
        self._m_step_expert_network()
        self._         
            

    def _m_step_expert_network(self):
        '''
        Runs maximisation step for experts in HME
        
        '''
        for i in range(self.n_gates_first):
            for j in range(self.n_gates_second):
                weights = []
                for n in range(self.n):
                    weights.append(self.responsibilities[n][i,j])
                W      = np.array(weights) 
                expert = wlr.WeightedLinearRegression(self.X,self.Y,W)
                expert.fit()
                # update parameters
                self.sigma_2[i,j] = expert.var
                self.gamma[i]     = expert.
                
                
    
    def _m_step_gating_network_level_two(self):
        '''
        Runs maximization step for second level gating network in HME
        '''
        for i in range(self.n_gates_first):
            for j in range(self.n_gates_second):
                weights = []
                for n in range(self.n):
                    weights.append(self.responsibilities[n][i,j])
                W = np.array(weights)
                
    
    def _m_step_gating_network_level_one(self):
        '''
        Runs maximisation step for first level gating network in HME
        '''
        pass
        
    
    
if __name__=="__main__":
    X = np.ones([10,2])
    X[1:5,0] = X[1:5,0] + 5
    Y = np.random.random(10)
    Y[0:5] = Y[0:5] + 5
    hme = HME_Gaussian(Y,X,2,3)
    hme.e_step()
    
    