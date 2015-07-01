# -*- coding: utf-8 -*-

"""
     K - number of first levels gating options
     P - number of second level gating options
     n - number of observations
     m - dimensionality of data
"""

from abstract_hme import AbstractHME
import numpy as np
import logistic_reg as lr
import label_binariser as lb

class LogisticRegressionHME(AbstractHME):
    
    def __init__(self,*args):
        super(LogisticRegressionHME,self).__init__(*args)
        # coefficients for softmax regression experts
        self.gamma    = [np.random.random([self.m,self.clusters[i]]) for i in range(self.n_gates_first)]
        self
        
    ################################### Abstract Method Implementation ###########################
    
    #-------------------------------------- E-step -----------------------------------------------#
    
    def _expert_probabilities(self):
        data = [np.zeros([self.n,self.clusters[i]]) for i in range(self.n_gates_first)]
        for i in range(self.n_gates_first):
            for j in range(self.clusters[i]):
                data[i][:,j] = lr.logistic_pdf(self.gamma[i][:,j],self.Y,self.X)
        return data
        
        
    def _m_step_expert_network(self):
       '''
       Weighted Linear regression for optimising parameters of experts
       '''
       for i in range(self.n_gates_first):
           for j in range(self.clusters[i]):
               weights = []
               for n in range(self.n):
                   weights.append(self.responsibilities[n][i][j])
               W      = np.array(weights) 
               expert = lr.LogisticRegression()
               expert.fit(self.Y,self.X, W)
               self.gamma[i][:,j]  = expert.theta
               
    def _predict_expert(self,X,i):
        expert = lr.LogisticRegression()
        return expert.predict(X,self.gamma[i])
        
        
if __name__ == "__main__":
    X = np.ones([50,3])
    X[:,1] = np.random.normal(0,1,50)
    X[:,2] = np.random.normal(0,1,50)
    X[25:50,:] = X[25:50,:] + 10
    Y = np.array(["y" for i in range(50)])
    Y[25:50] = "n"
    hme = LogisticRegressionHME(Y,X,10,[8,8,8,8,8,8,8,8,8,8])
    hme.iterate()
            
        
        
    
    
    
