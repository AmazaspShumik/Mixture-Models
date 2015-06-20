# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

class MDA(object):
    '''
    Mixture Discriminant Analysis
    '''
    
    def __init__(self,gt,X,clusters, max_iter_init = 100, init_restarts       = 2, 
                                                          init_conv_theshold  = 1e-2,
                                                          iter_conv_threshold = 1e-8,
                                                          max_iter            = 900,
                                                          verbose             = True,
                                                          accuracy            = 1e-5):
        self.Y                   =  gt
        self.X                   =  X
        self.n, self.m           =  np.shape(X)               # n - observations; m - dimension   
        self.k                   =  np.shape(Y)[1]            # k - number of classes
        self.clusters            =  clusters
        self.class_prior         =  np.zeros(self.m)          # class prior probabilities
        self.latent_var_prior    =  [np.random.random(clusters[i]) for i in range(self.k)] 
        self.freq                =  np.sum(self.Y, axis = 0)  # number of elements in each class
        self.covar               =  np.eye(self.m)
        self.mu                  =  [np.random.random([self.m,clusters[i]]) for i in range(self.k)] # means
        self.responsibilities    =  [np.random.random([self.n,clusters[i]]) for i in range(self.k)]
        self.lower_bounds        =  []
        self.kmeans_maxiter      =  max_iter_init
        self.kmeans_retsarts     =  init_restarts
        self.max_iter            =  max_iter
        self.kmeans_theshold     =  init_conv_theshold
        self.mda_threshold       =  iter_conv_threshold
        self.verbose             =  verbose
        self.accuracy            =  accuracy
        
        
    def initialise_params(self):
        '''Runs k-means algorithm for parameter initialisation'''        
        pass
        
        
        
    def iterate(self):
        ''' '''
        self._class_prior_compute()
        delta = 1
        for i in range(self.max_iter):
            self.e_step()
            if len(self.lower_bounds) >= 2:
                delta_change = float(self.lower_bounds[-1] - self.lower_bounds[-2])
                delta        = delta_change/abs(self.lower_bounds[-2])
            if delta > self.mda_threshold:
                self.m_step()
                if self.verbose:
                    iteration_verbose = "iteration {0} completed, lower bound of log-likelihood is {1} "
                    print iteration_verbose.format(i,self.lower_bounds[-1])
            else:
                print "algorithm converged"
                break
        
        
    def e_step(self):
        self._e_step_lower_bound_likelihood()
        
    def _e_step_lower_bound_likelihood(self):
        '''
        Calculates posterior distribution of latent variable for each class
        and lower bound for log-likelihood of data
        '''
        lower_bound = 0.0
        for i,resp_k in enumerate(self.responsibilities):
            for j in range(self.clusters[i]):
                prior            = mvn.pdf(X,self.mu[i][:,j],self.covar),self.accuracy,1-self.accuracy)                       # prevent underflow
                weighting        = self.Y[:,i] * self.responsibilities[i][:,j]
                w                =  weighting*np.log(prior) + weighting*np.log(self.latent_var_prior[i][j])
                lower_bound     += (np.sum(w))
                resp_k[:,j]      = prior*self.latent_var_prior[i][j]
            normaliser = np.sum(resp_k, axis = 1)
            resp_k    /= np.outer(normaliser,np.ones(self.clusters[i]))
            lower_bound += np.log(self.class_prior[i])*self.freq[i]
        self.lower_bounds.append(lower_bound)
        
        
    def m_step(self):
        '''
        M-step of Expectation Maximization Algorithm
        Calculates
        '''
        covar = np.zeros([self.m,self.m])
        for i in range(self.k):
            
            # calculate mixing probabilities
            class_indicator          = np.outer(self.Y[:,i],np.ones(self.clusters[i]))*self.responsibilities[i]
            weighted_freq            = np.sum(class_indicator , axis = 0)
            self.latent_var_prior[i] = weighted_freq/self.freq
            
            # calculate means
            weighted_means           = np.array([np.sum(np.dot(X.T,np.diagflat(class_indicator[:,j])), axis = 1) 
                                                 for j in range(self.clusters[i])]).T
            self.mu[i]               = weighted_means / np.outer(weighted_freq,np.ones(self.m)).T

            # calculate pooled covariance matrix
            for j in range(self.clusters[i]):
                centered = self.X - np.outer(self.mu[i][:,j],np.ones(self.n)).T
                covar   += np.dot(np.dot(centered.T, np.diagflat(class_indicator[:,j])),centered)
            self.covar = covar/self.n
        


    def _class_prior_compute(self):
        ''' Computes prior probability of observation being in particular class '''
        self.class_prior = self.freq/np.sum(self.freq)       
        
#--------------------- Helper Methods --------------------------------------------#
    
    
    @staticmethod
    def bounded_variable(x,lo,hi):
        '''
        Returns 'x' if 'x' is between 'lo' and 'hi', 'hi' if x is larger than 'hi'
        and 'lo' if x is lower than 'lo'
        '''
        if   x > hi:
            return hi*np.ones(len(x))
        elif x < lo:
            return lo*np.ones(len(x))
        else:
            return x 
        
        
if __name__=="__main__":
    X = np.random.normal(0,1,[100,3])
    X[0:25,:] -= 10*np.ones([25,3])
    X[50:75,:]+= 15*np.ones([25,3])
    X[75:100,:]+= 5*np.ones([25,3])
    Y = np.zeros([100,2])
    Y[0:50,0] = 1
    Y[50:100,1] = 1
    plt.plot(X[:,0],X[:,1],"b+")
    mda = MDA(Y,X,[2,2])
    mda._class_prior_compute()
    for i in range(100):
        mda.e_step()
        mda.m_step()
    
    