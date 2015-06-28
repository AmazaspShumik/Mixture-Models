# -*- coding: utf-8 -*-

"""
     K - number of first levels gating options
     P - number of second level gating options
     n - number of observations
     m - dimensionality of data
"""

from abstract_hme import AbstractHME

class LogisticRegressionHME(AbstractHME):
    
    def __init__(self,*args):
        super(LogisticRegressionHME,self).__init__(*args)
        # coefficients for softmax regression experts
        self.gamma    = [np.random.random([self.m,self.clusters[i]]) for i in range(self.n_gates_first)]
        
        
        
        
    
    
    
