# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:22:33 2015

@author: amazaspshaumyan
"""

import SoftmaxRegression as sr
import WeightedLinearRegression as wlr
import numpy as np
import 




class HME_Gaussian(object):
    
    
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
        self.sigma_2          = np.zeros([self.n_gates_first,self.n_gates_second])
        # responsibilities
        self.responsibilities = [np.zeros([self.n_gates_first,self.n_gates_second]) for i in range(self.m)]
        
    def e_step(self):
        '''
        E-step in EM algorithm for training Hierarchical Mixture of Experts
        '''
        resp_gate_first    = sr.softmax(self.aplha,self.X)                
        resp_gate_second   = [sr.softmax(self.beta[i], self.X) for i in range(self.n_gates_first)]
        resp_expert        = 

    def lower_bound_likelihood(self):
        
    
    def e_step(self):
        pass
    
    def m_step(self):
        pass
    