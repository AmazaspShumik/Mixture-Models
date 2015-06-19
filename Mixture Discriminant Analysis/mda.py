# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:59:57 2015

@author: amazaspshaumyan
"""




import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

class MDA(object):
    '''
    Mixture Discriminant Analysis
    '''
    
    def __init__(self,gt,X,clusters):
        self.Y                 =  gt
        self.X                 =  X
        self.n, self.m         =  np.shape(X)               # n - observations; m - dimension   
        self.k                 =  np.shape(Y)[1]            # k - number of classes
        self.clusters          =  clusters
        self.class_prior       =  np.zeros(self.m)          # class prior probabilities
        self.latent_var_prior  =  [np.ones(clusters[i], dtype = np.float)/clusters[i] for i in range(self.k)] 
        self.freq              =  np.sum(self.Y, axis = 0)  # number of elements in each class
        self.covar             =  np.eye(self.m)
        self.mu                =  [np.zeros([self.m,clusters[i]]) for i in range(self.k)] # means
        self.responsibilities  =  [np.random.random([self.n,clusters[i]]) for i in range(self.k)]
        
        
    def initialise_params(self):
        '''Runs k-means algorithm for parameter initialisation'''        
        pass
        
        
        
    def iterate(self):
        ''' '''
        self._class_prior_compute()
        # construct responsibility matrix
        
    def e_step(self):
        '''
        E-step of Expectattion Maximization Algorithm
        Calculates posterior distribution of latent variable for each class
        '''
        for i,resp_k in enumerate(self.responsibilities):
            for j in range(self.clusters[i]):
                resp_k[:,j]     = mvn.pdf(X,self.mu[i][:,j],self.covar)*self.latent_var_prior[i][j]
            normaliser = np.sum(resp_k, axis = 1)
            resp_k    /= np.outer(normaliser,np.ones(self.clusters[i]))
        
        
    def m_step(self):
        covar = np.zeros([self.m,self.m])
        for i in range(self.k):
            
            # calculate mixing probabilities
            class_indicator          = np.outer(self.Y[:,i],np.ones(self.clusters[i]))*self.responsibilities[i]
            weighted_freq            = np.sum(class_indicator , axis = 0)
            print "weighted frequency"
            print weighted_freq
            self.latent_var_prior[i] = weighted_freq/self.freq
            
            # calculate means
            weighted_means           = np.array([np.sum(np.dot(X.T,np.diagflat(class_indicator[:,j])), axis = 1) 
                                                 for j in range(self.clusters[i])]).T
            self.mu[i]               = weighted_means / np.outer(weighted_freq,np.ones(self.m)).T
            print "means"
            print self.mu[i]
            
            # calculate pooled covariance matrix
            for j in range(self.clusters[i]):
                centered = self.X - np.outer(self.mu[i][:,j],np.ones(self.n)).T
                covar   += np.dot(np.dot(centered.T, np.diagflat(class_indicator[:,j])),centered)
            self.covar = covar/self.n
        
    
    def _m_step_latent_variable_prior(self):
        pass
            
            
    
    def _m_step_mu(self):
        pass
    
    def log_likelihood_lower_bound(self):
        pass
        
    
    
    
    
    
    
    def _class_prior_compute(self):
        ''' Computes prior probability of observation being in particular class '''
        self.class_prior = self.freq/np.sum(self.freq)       
        
        
#--------------------- Helper Methods --------------------------------------------#
        
if __name__=="__main__":
    X = np.random.normal(0,1,[100,3])
    #X[0:25,:] -= 10*np.ones([25,3])
    #X[50:70,:]+= 15*np.ones([20,3])
    X[50:100,:]+= 20*np.ones([50,3])
    Y = np.zeros([100,2])
    Y[0:50,0] = 1
    Y[50:100,1] = 1
    plt.plot(X[:,0],X[:,1],"b+")
    mda = MDA(Y,X,[1,1])
    mda._class_prior_compute()
    for i in range(100):
        mda.e_step()
        mda.m_step()
    
    