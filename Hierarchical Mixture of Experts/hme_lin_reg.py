# -*- coding: utf-8 -*-
"""
     K - number of first levels gating options
     P - number of second level gating options
     n - number of observations
     m - dimensionality of data
"""


from abstract_hme import AbstractHME
import numpy as np
import WeightedLinearRegression as wlr


class LinearRegressionHME(AbstractHME):
    '''
    Hierarchical Mixture of Linear Regressions
    
    '''
    
    def __init__(self,*args):
        super(LinearRegressionHME,self).__init__(*args)
        # coefficients for linear regression experts
        self.gamma    = [np.random.random([self.m,self.clusters[i]]) for i in range(self.n_gates_first)]
        # variance for error term of regression
        self.sigma_2  = [np.ones(self.clusters[i]) for i in range(self.n_gates_first)]
    
    
    #################################### Abstract Methods Implemetation ###################################        
        
        
        
    #----------------------------------------- E-step ----------------------------------------------------#
        
        
    def _expert_probabilities(self):
        '''
        Calculates probabilities from expert network
        
        Overrides abstract method
        '''
        data = [np.zeros([self.n,self.clusters[i]]) for i in range(self.n_gates_first)]        
        for i in range(self.n_gates_first):
            for j in range(self.clusters[i]):
                data[i][:,j] = wlr.norm_pdf(self.gamma[i][:,j],self.Y,self.X,self.sigma_2[i][j])
        return data
        
        
    #----------------------------------------- M-step -----------------------------------------------------#
        
        
    def _m_step_expert_network(self):
        '''
        Weighted Linear regression for optimising parameters of experts.
        '''
        for i in range(self.n_gates_first):
            for j in range(self.clusters[i]):
                weights = []
                for n in range(self.n):
                    weights.append(self.responsibilities[n][i][j])
                W      = np.array(weights) 
                expert = wlr.WeightedLinearRegression(self.X,self.Y,W)
                expert.fit()
                self.sigma_2[i][j]  = expert.var
                self.gamma[i][:,j]  = expert.theta
    
    
    #----------------------------------------- Prediction ---------------------------------------------------#
    
    def _predict_expert(self,X,i):
        '''
        Point prediction by expert model
        '''
        return np.dot(X,self.gamma[i])
        
if __name__ == "__main__":
    X      = np.zeros([1000,2])
    X[:,0] = np.linspace(0, 10, 1000)
    X[:,1] = np.ones(1000)
    Y = np.sin(X[:,0])*4 + np.random.normal(0,1,1000)
    hme = LinearRegressionHME(Y,X,10,[8,8,8,8,8,8,8,8,8,8])
    hme.iterate()
    
    
        
        
        
    