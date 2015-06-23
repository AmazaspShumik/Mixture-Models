# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class MDA(object):
    '''
    Mixture Discriminant Analysis
    '''
    
    def __init__(self,gt,X,clusters, max_iter_init = 100, init_restarts       = 2, 
                                                          init_conv_theshold  = 1e-2,
                                                          iter_conv_threshold = 1e-20,
                                                          max_iter            = 900,
                                                          verbose             = True,
                                                          accuracy            = 1e-5):
        self.Y                   =  gt
        self.X                   =  X
        self.n, self.m           =  np.shape(X)               # n - observations; m - dimension   
        self.k                   =  np.shape(Y)[1]            # k - number of classes
        self.clusters            =  clusters
        self.class_prior         =  np.zeros(self.m)          # class prior probabilities
        self.latent_var_prior    =  [np.ones(clusters[i])/clusters[i] for i in range(self.k)] 
        self.freq                =  np.sum(self.Y, axis = 0)  # number of elements in each class
        self.covar               =  np.eye(self.m)
        self.mu                  =  [np.zeros([self.m,clusters[i]]) for i in range(self.k)]  # means
        self.responsibilities    =  [0.001*np.zeros([self.n,clusters[i]]) for i in range(self.k)]
        self.lower_bounds        =  []
        self.kmeans_maxiter      =  max_iter_init
        self.kmeans_retsarts     =  init_restarts
        self.max_iter            =  max_iter
        self.kmeans_theshold     =  init_conv_theshold
        self.mda_threshold       =  iter_conv_threshold
        self.verbose             =  verbose
        self.accuracy            =  accuracy
        
        
    def train(self):
        self._initialise_params()
        self._iterate()
        
    def _posterior_prob(self):
        
        
        
    def _initialise_params(self):
        '''Runs k-means algorithm for parameter initialisation'''
        
        # initialise class priors
        self._class_prior_compute()
        
        # calculate responsibilities using k-means results      
        for i,cluster in enumerate(self.clusters):
            kmeans = KMeans(n_clusters = cluster, 
                            max_iter    = self.kmeans_maxiter,
                            init       = "k-means++",
                            tol        = self.kmeans_theshold)
            kmeans.fit(self.X[self.Y[:,i]==1,:])
            prediction = kmeans.predict(self.X[self.Y[:,i]==1,:])
            #covar = np.zeros([self.m,self.m])
            for j in range(cluster):
                self.responsibilities[i][self.Y[:,i]==1,j] = 1*(prediction==j)
                
        # initialise parameters of mda through M-step
        self.m_step()
        if self.verbose:
            print "Initialization step complete"
            

    def _iterate(self):
        ''' '''
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
                prior            = self.bounded_variable(mvn.pdf(self.X,self.mu[i][:,j],self.covar),self.accuracy,1-self.accuracy)
                weighting        = self.Y[:,i] * resp_k[:,j]
                w                =  weighting*np.log(prior) + weighting*np.log(self.latent_var_prior[i][j])
                lower_bound     += np.sum(w)
                resp_k[:,j]      = prior*self.latent_var_prior[i][j]
            normaliser = np.sum(resp_k, axis = 1)
            resp_k    /= np.outer(normaliser,np.ones(self.clusters[i]))
        self.lower_bounds.append(lower_bound)
        
        
    def m_step(self):
        '''
        M-step of Expectation Maximization Algorithm
        Calculates
        '''
        covar = np.zeros([self.m,self.m])
        for i in range(self.k):
            for j in range(self.clusters[i]):
                
                # calculate mixing probabilities
                class_indicator          = self.Y[:,i]*self.responsibilities[i][:,j]
                self.latent_var_prior[i][j] = np.sum(class_indicator)/self.freq[i]
    
                # calculate means
                weighted_means           = np.sum(np.dot(X.T, np.diagflat(class_indicator)), axis=1)
                self.mu[i][:,j]               = weighted_means / np.sum(class_indicator)
                
                # calculate pooled covariance matrix
                centered = self.X - np.outer(self.mu[i][:,j],np.ones(self.n)).T
                addition = np.dot(np.dot(centered.T, np.diagflat(class_indicator)),centered)
                covar   += addition
                
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
        x[x>hi]=hi
        x[x<lo]=lo
        return x
        
        
if __name__=="__main__":
    X = np.random.normal(0,0.2,[24,2])
    X[0:6,0] = np.random.normal(2,0.2,6)
    X[0:6,1] = np.random.normal(2,0.2,6)
    X[6:12,0] = np.random.normal(3,0.2,6)
    X[6:12,1] = np.random.normal(4,0.2,6)
    X[12:18,:]  = np.random.normal(0,0.2,[6,2])
    X[18:24,:]  = np.random.normal(-4,0.2,[6,2])
    #X[0:12,:] = np.random.normal(0,1,[12,2])
    #X[12:24,:] = np.random.normal(4,1,[12,2])
    Y = np.zeros([24,2])
    Y[0:12,0] = 1
    Y[12:24,1] = 1
    plt.plot(X[:,0],X[:,1],"b+")
    mda = MDA(Y,X,[2,2])
    mda.train()
    
    
    
    
    