# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import psi
from sklearn.cluster import KMeans
from scipy.linalg import pinvh
from scipy.misc import logsumexp
from scipy.stats import t
from math import *
import warnings
from numpy import pi

#TODO: lower bound & convergence check using lower bound

class StudentMultivariate(object):
    '''
    Multivariate Student Distribution
    '''
    def __init__(self,mean,precision,df,d):
        self.mu   = mean
        self.prec = precision
        self.df   = df
        self.d    = d
                
    def pdf(self,x):
        Num = gamma(1. * (self.d+self.df)/2)
        Denom = ( ( gamma(1.*self.df/2) * pow(self.df*pi,1.*self.d/2) * pow(np.linalg.det(self.prec),1./2) * 
                    pow(1 + (1./self.df)*np.dot(np.dot((x -
                    self.mu),np.linalg.inv(self.prec)), (x - self.mu)),1.* (self.d+self.df)/2)) 
                )
        d = 1. * Num / Denom 
        return d



#----------  Variational Gaussian Mixture Model with Automatic Relevance Determination ---------#

class VBGMMARD(object):
    '''
    Variational Bayeisian Gaussian Mixture Model with Automatic Relevance 
    Determination. Implemented model automatically selects number of relevant 
    components through mixture of Type II Maximum Likelihood and Mean Field
    Approximation. In constrast to standard 
    
    Parameters:
    -----------       
    max_components: int
       Maximum number of mixture components
       
    max_iter: int (DEFAULT = 10)
       Maximum number of iterations
       
    conv_thresh: float (DEFAULT = 1e-3)
       Convergence threshold 
       
    prune_thresh: float
       Threshold for pruning components
       
    n_kmean_inits: int
       Number of time k-means algorithm will be run with different centroid 
       seeds
       
    rand_state: int
       Random number that is used for initialising centroids
       
    mfa_max_iter: int
       Maximum number of iterations for Mean Field Approximation of lower bound for 
       evidence function 
       
    References:
    -----------
    Adrian Corduneanu and Chris Bishop, Variational Bayesian Model Selection 
    for Mixture Distributions (2001)
    
    '''
    
    def __init__(self, max_components,means = None, dof = None, covar = None,  
                       weights = None, beta = 1e-3, max_iter = 100,
                       conv_thresh = 1e-5,n_kmean_inits = 3, prune_thresh = 1e-2,
                       rand_state = 1, mfa_max_iter = 3):
        self.n_components               =  max_components
        self.dof0, self.scale_inv0      =  dof,covar
        self.weights0,self.means0       =  weights,means
        self.beta0                      =  beta
        self.max_iter,self.conv_thresh  =  max_iter, conv_thresh
        self.n_kmean_inits              =  n_kmean_inits
        self.prune_thresh               =  prune_thresh
        self.rand_state                 =  rand_state
        self.mfa_max_iter               =  mfa_max_iter
        self.converged                  =  False
        # parameters of predictive distribution
        self.St                         =  None
        # boolean that identifies whther model was fitted or not
        self.is_fitted                  =  True
        
      
    def _init_params(self,X):
        '''
        Initialise parameters
        '''
        self.n, self.d         = X.shape
        
        # Initialise parameters for all priors, these parameters are used in 
        # variational approximation at each iteration so they should be saved
        # and not changed
            
        # initialise prior on means & precision matrices
        if self.means0 is None:
            kms = KMeans(n_init = self.n_kmean_inits, n_clusters = self.n_components, 
                         random_state = self.rand_state)
            self.means0     = kms.fit(X).cluster_centers_
            
        # broad prior over precision matrix
        if self.scale_inv0 is None:
            # heuristics that seems to work pretty good
            diag_els        = (np.max(X,0) - np.min(X,0))**2
            self.scale_inv0 = np.diag( diag_els  )
            self.scale0     = np.diag( 1./ diag_els )
            
        # initialise weights
        if self.weights0 is None:
            self.weights0  = np.ones(self.n_components) / self.n_components
          
        # initial number of degrees of freedom
        if self.dof0 is None:
            self.dof0           = self.d
            
        # clusters that are not pruned 
        self.active             = np.array([True for _ in range(self.n_components)])
        
        # checks initialisation errors in case parameters are user defined
        assert self.dof0 >= self.d,( 'Degrees of freedom should be larger than '
                                         'dimensionality of data')
        assert self.means0.shape[0] == self.n_components,('Number of centrods defined should '
                                                          'be equal to number of components')
        assert self.means0.shape[1] == self.d,('Dimensioanlity of means and data '
                                                   'should be the same')
        assert self.weights0.shape[0] == self.n_components,('Number of weights should be equal '
                                                           'to number of components')
        
        # At first iteration these parameters are equal to priors, but they change 
        # at each iteration of mean field approximation
        self.scale   = np.array([np.copy(self.scale0) for _ in range(self.n_components)])
        self.means   = np.copy(self.means0)
        self.weights = np.copy(self.weights0)
        self.dof     = self.dof0*np.ones(self.n_components)
        self.beta    = self.beta0*np.ones(self.n_components)
        

    def _update_logresp_k(self, X, k):
        '''
        Updates responsibilities for single cluster, calculates expectation
        of logdet of precision matrix.
        
        Parameters:
        -----------
        X: numpy array of size [n_samples,n_features] 
           Data matrix
           
        k: int
           Cluster index

        Returns:
        --------
        log_pnk: numpy array of size [n_features,1]
                 Responsibilities without normalisation
        '''
        # calculate expectation of logdet of precision matrix
        scale_logdet   = np.linalg.slogdet(self.scale[k])[1]
        e_logdet_prec  = sum([psi(0.5*(self.dof[k]+1-i)) for i in range(1,self.d+1)])
        e_logdet_prec += scale_logdet + self.d*np.log(2)
           
        # calculate expectation of quadratic form (x-mean_k)'*precision_k*(x - mean_k)
        x_diff         = X - self.means[k,:]
        e_quad_form    = np.sum( np.dot(x_diff,self.scale[k,:,:])*x_diff, axis = 1 )
        e_quad_form   *= self.dof[k]
        
        # responsibilities without normalisation
        log_pnk        = np.log(self.weights[k]) + 0.5*e_logdet_prec - e_quad_form
        log_pnk       -= 0.5*self.d / self.beta[k]
        return log_pnk
                
                
    def _update_resps(self,X):
        '''
        Updates distribution of latent variable (responsibilities)
        
        Parameters:
        -----------
        X: numpy array of size [n_samples,n_features] 
           Data matrix

        Returns:
        --------
        p: numpy array of size [n_samples, n_components]
           Responsibilities
        '''
        # log of responsibilities before normalisaton
        log_p     = [self._update_logresp_k(X,k) for k in range(self.n_components)]
        log_sum_p = logsumexp(log_p,axis = 0, keepdims = True)
        log_p    -= log_sum_p
        p         = np.exp(log_p)
        return p.T
    
    
    def _update_means_precisions(self, Nk, Xk, Sk):
        '''
        Updates distribution of means and precisions
        
        Parameters:
        -----------
        Nk: numpy array of size [n_components,1]
            Sum of responsibilities by component
        
        Xk: list of numpy arrays of length n_components
            Weighted average of observarions, weights are responsibilities
        
        Sk: list of numpy arrays of length n_components
            Weighted variance of observations, weights are responsibilities 
        '''
        for k in range(self.n_components):
            # update mean and precision for each cluster
            self.beta[k]   = self.beta0 + Nk[k]
            self.means[k]  = (self.beta0*self.means0[k,:] + Nk[k]*Xk[k]) / self.beta[k]
            self.dof[k]    = self.dof0 + Nk[k] + 1
            Xkdiff         = Xk[k] - self.means0[k,:]
            self.scale[k,:,:]  = pinvh(self.scale_inv0 +  Nk[k]* ( Sk[k] + 
                             self.beta0/(self.beta0 + Nk[k])*np.outer(Xkdiff,Xkdiff)))

                             
    def _check_convergence(self,n_components_before,means_before):
        '''
        Checks convergence

        Parameters:
        -----------
        n_components_before: int 
            Number of components on previous iteration
            
        means_before: numpy array of size [n_components, n_features]
            Cluster means on previous iteration
            
        Returns:
        --------
        :bool 
            If True then converged, otherwise not
        '''
        conv = True
        for mean_before,mean_after in zip(means_before,self.means):
            mean_diff = mean_before - mean_after
            conv  = conv and np.sum(np.abs(mean_diff)) / self.d < self.conv_thresh
        return conv
    
                             
    def _postprocess(self,X):
        '''
        Performs postprocessing after convergence
        
        Theoretical Note:
        =================
        We needed extremely broad prior covariance for good convergence, and 
        irrelevant cluster elimination, however broad prior also adds huge 
        bias to estimated covariance, so after convergence we remove part of 
        covariance that is due to prior and update weights after that.
        
        Parameters:
        -----------
        X: numpy array of size [n_samples, n_features]
           Data matrix
        '''
        for i,W in enumerate(self.scale):
            self.scale[i,:,:] = pinvh(pinvh(W) - self.scale_inv0)
        resps            = self._update_resps(X)
        self.weights     = np.sum(resps,0) / self.n
        
        
    def fit(self, X):
        '''
        Fits Variational Bayesian GMM with ARD, automatically determines number 
        of mixtures
        
        Parameters:
        -----------
        X: numpy array [n_samples,n_features]
           Data matrix
        '''
        # initialise all parameters
        self._init_params(X)
        
        # when fitting new model old parmaters for predictive distribution are 
        # not valid any more
        if self.is_fitted is True : self.St = None
        
        active = np.array([True for _ in range(self.n_components)])        
        for j in range(self.max_iter):
            for i in range(self.mfa_max_iter):
                                
                # STEP 1:   Approximate distribution of latent vatiable, means and 
                #           precisions using Mean Field Approximation method
                
                # calculate responsibilities
                resps = self._update_resps(X)
                
                # precalculate some intermediate statistics
                Nk     = np.sum(resps,axis = 0)
                Xk     = [np.sum(resps[:,k:k+1]*X,0)/Nk[k]  for k in range(self.n_components)]
                diff_x = [X - Xk[k] for k in range(self.n_components)]
                Sk     = [np.dot(resps[:,k]*diff_x[k].T,diff_x[k])/Nk[k] for  \
                          k in range(self.n_components)]
                          
                # update distributions of means and precisions
                means_before = np.copy(self.means)
                self._update_means_precisions(Nk,Xk,Sk)
                
                # STEP 2: Maximize lower bound with respect to weights, prune
                #         clusters with small weights & check convergence 
                if i+1 == self.mfa_max_iter:
                    
                    # update weights to maximize lower bound  
                    self.weights      = Nk / self.n
                    
                    # prune all irelevant weights
                    active              = self.weights > self.prune_thresh
                    self.means0         = self.means0[active,:]
                    self.scale          = self.scale[active,:,:]
                    self.weights        = self.weights[active]
                    self.weights       /= np.sum(self.weights)
                    self.dof            = self.dof[active]
                    self.beta           = self.beta[active]
                    n_components_before = self.n_components
                    self.means          = self.means[active,:]
                    self.n_components   = np.sum(active)
                    
                    # check convergence
                    if n_components_before == self.n_components:
                        self.converged  = self._check_convergence(n_components_before,
                                                                  means_before)
                    
                    # if converged postprocess
                    if self.converged == True:
                        self._postprocess(X)
                        self.is_fitted = True
                        return
                        
        warnings.warn( ("Algorithm did not converge!!! Maximum number of iterations "
                        "achieved. Try to change either maximum number of iterations "
                        "or conv_thresh parameters"))
        self._postprocess(X)
        self.is_fitted  = True
        
        
    def predict_cluster_prob(self,x):
        '''
        Calculates of observation being in particular cluster
        
        Parameters:
        -----------
        x: numpy array of size [n_samples_test_set, n_features]
           Data matrix for test set
           
        Returns:
        --------
        : numpy array of size [n_samples_test_set, n_components]
           Responsibilities for test set
        '''
        return self._update_resps(x)
    
    
    def predict_cluster(self,x):
        '''
        Predicts which cluster generated test data
        
        Parameters:
        -----------
        x: numpy array of size [n_samples_test_set, n_features]
           Data matrix for test set
           
        Returns:
        --------
        : numpy array of size [n_samples_test_set, n_components]
           Responsibilities for test set
        '''
        return np.argmax( self._update_resps(x), 1)
        
        
    def _predict_params(self):
        '''
        Calculates parameters for predictive distribution
        '''
        self.St = []
        for k in range(self.n_components):
            df    = self.dof[k] + 1 - self.d
            prec  = self.scale[k,:,:] * self.beta[k] * df / (1 + self.beta[k])
            self.St.append(StudentMultivariate(self.means[k,:],prec,self.dof[k],self.d))
        
        
    def predictive_pdf(self,x):
        '''
        PDF Predictive distribution
        
        Parameters:
        -----------
        x: numpy array of size [n_samples_test_set,n_features]
           Data matrix for test set
           
        Returns:
        --------
        : numpy array
           Value of pdf of predictive distribution at x
        '''
        # check whether prediction parameters were calculated before
        if self.is_fitted is True and self.St is None:
            self._predict_params()       
        return [w*st.pdf(x) for w,st in zip(self.weights,self.St)]


    def get_params(self):
        '''
        Returns dictionary with all learned parameters
        '''
        covars = [1./df * pinvh(sc) for sc,df in zip( self.scale, self.dof)]
        params = {'means': self.means, 'covars': covars,'weights': self.weights}
        return params



class VBGMMARDGClassifier(object):
    '''
    Generative classifier
    
    '''
    
    def __init__(self, max_components, means = None, dof = None, covar = None,  
                       weights = None, beta = 1e-3, max_iter = 100,
                       conv_thresh = 1e-5,n_kmean_inits = 3, prune_thresh = 1e-2,
                       rand_state = 1, mfa_max_iter = 3):
                           
        self.n_components               =  max_components
        self.dof, self.covar            =  dof,covar
        self.weights,self.means         =  weights,means
        self.beta                       =  beta
        self.max_iter,self.conv_thresh  =  max_iter, conv_thresh
        self.n_kmean_inits              =  n_kmean_inits
        self.prune_thresh               =  prune_thresh
        self.rand_state                 =  rand_state
        self.mfa_max_iter               =  mfa_max_iter
        # boolean that identifies whether model was fitted or not
        self.is_fitted                  =  True
        
        
    def _init_params(self):
        '''
        Initialise parameters
        '''
        pass
        

    
    def _fit(self,X,Y):
        '''
        Finds distribution of explanatory variables for each class
        '''
        # calculate prior p( y = Ck)
        prior = np.sum(Y, axis = 0) / self.n
        
        # calculate p(x | y = Ck) * p( y = Ck )
        post  = [VBGMMARD(self.n_components[k],self.means[k,:], self.dof[k],
                 self.covar[k],self.weights[k], self.beta[k],)]
        
        
        
        
        
    
    
    def fit(self,X,Y):
        '''
        Fits classification model
        
        Parameters:
        -----------
        X: numpy array of size [n_samples, n_features]
           Matrix of explanatory variables
           
        Y: numpy array of size [n_samples, 1]
           Vector of dependent variables
        '''
        # binarise
        
        # initialise all parameters
        
        
        
        
        
    def predict(self,x):
        '''
        Predicts class to which observations belong
        
        Parameters:
        -----------
        x: numpy array of size [n_samples_test, n_features]
           Data matrix for test set
           
        Returns:
        --------
        classes: numpy array of size [n_sample_test, 1]
           Predicted class            
        '''
        pass
    
    
    def predict_prob(self,x):
        '''
        Predicts class to which observations belong
        
        Parameters:
        -----------
        x: numpy array of size [n_samples_test, n_features]
           Data matrix for test set
           
        Returns:
        --------
        probs: numpy array of size [n_sample_test, n_classes]
           Matrix of probabilities
        '''
        pass
    

        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    X = np.zeros([300,2])
    # [1,1]
    X[0:100,0] = np.random.normal(1,2,100)
    X[0:100,1] = np.random.normal(1,1,100) 
    # [12,5]
    X[100:200,0] = np.random.normal(25,3,100)
    X[100:200,1] = np.random.normal(19,3,100) 
    # [-4,-13]
    X[200:300,0] = np.random.normal(-23,1,100)
    X[200:300,1] = np.random.normal(-23,2,100)
    vbgmm = VBGMMARD(max_components = 5,init_type = 'auto')
    resps = vbgmm.fit(X)
    plt.plot(X[:,0],X[:,1],'ro')
    plt.plot(vbgmm.means[:,0],vbgmm.means[:,1],'go',markersize = 10)
    plt.show()
    
    print "\n real means"
    print np.mean(X[0:100,:],0)
    print np.mean(X[100:200,:],0)
    print np.mean(X[200:300,:],0)
    
    print '\n means by algorithm'
    print vbgmm.means
    
    print '\n covariance by algorithm'
    print vbgmm.get_params()["covars"]
    
    print "\n real covariance"
    print np.cov(X[0:100,:].T)
    print np.cov(X[100:200,:].T)
    print np.cov(X[200:300,:].T)
    
    # Old Faithful Data
    import os
    import pandas as pd
    
    os.chdir("/Users/amazaspshaumyan/Desktop/MixtureExperts/VBGMM with ARD/")
    Data = pd.read_csv("old_faithful.txt")
    vbgmm_of = VBGMMARD(max_components = 20, init_type = 'auto', conv_thresh = 1e-3)
    r = vbgmm_of.fit(np.array(Data[['eruptions','waiting']]))
    plt.plot(Data['eruptions'],Data['waiting'],'bo')
    plt.plot(vbgmm_of.means[:,0],vbgmm_of.means[:,1],'ro')
    plt.show()
    
    resps = vbgmm.predict_cluster_prob( X )
    clust = vbgmm.predict_cluster( X )
    
#    print "Selected number of clusters {0}".format(vbgmm_of.means.shape[0])
    

    
    
            
            
            
            

    
    
    