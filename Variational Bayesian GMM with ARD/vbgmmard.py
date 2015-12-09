# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import psi
from scipy.special import gammaln
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from scipy.linalg import pinvh
from scipy.linalg import eigvalsh
from scipy.misc import logsumexp

LOG_PI = np.log(np.pi)
LOG_2  = np.log(2)

#-------------------------- Wishart Distribution ------------------------------#


def wishart_log_normaliser(scale,dof,scale_logdet):
    '''
    Negative logarithm of normalisation constant for Wishart distribution
    
    Parameters:
    -----------
    scale: numpy array of size [n_features,n_features]
         Scale matrix
         
    dof: int
         Degrees of freedom
         
    scale_logdet: float
         Logarithm of determinant of scale matrix
    
    Returns:
    --------
    :float
         negative logarithm of normalization constant for Wishart distribution
    '''
    D = scale.shape[0]
    return (
             + 0.5 * dof * scale_logdet
             + 0.5 * dof * D * LOG_2
             + 0.25 * D * (D-1) * LOG_PI 
             + np.sum([ 0.5 * gammaln(dof + 1 - i) for i in range(D)])
           )
    

def wishart_entropy(dof,scale,prec_logdet, scale_logdet):
    '''
    Entropy of wishart distribution
    
    Parameters:
    -----------
    dof: int
         Degrees of freedom
         
    scale: numpy array of size [n_features, n_features]
         Scale matrix
    
    prec_logdet: float
         Expectation of logdet of covariance matrix
         
    scale_logdet: float
         Logarithm of determinant of scale matrix
    
    Returns:
    --------
    : float
         Entropy of Wishart distribution
    '''
    D = scale.shape[0]
    return (
              wishart_log_normaliser(scale,dof,scale_logdet)
            - 0.5 * (dof - D - 1) * prec_logdet
            + 0.5 * dof * D
           )

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
    
    def __init__(self, max_components, dof = None, covar = None, weights = None, beta = 1e-3, 
                       means = None,init_type = 'automatic',max_iter = 25,
                       conv_thresh = 1e-3,n_kmean_inits = 3, prune_thresh = 1e-2,
                       rand_state = 1, mfa_max_iter = 2):
        self.n_components               =  max_components
        self.dof0, self.scale_inv0      =  dof,covar
        self.weights0,self.means0       =  weights,means
        self.beta0                      =  beta
        self.init_type                  =  init_type
        self.max_iter,self.conv_thresh  =  max_iter, conv_thresh
        self.n_kmean_inits              =  n_kmean_inits
        self.prune_thresh               =  prune_thresh
        self.rand_state                 =  rand_state
        self.mfa_max_iter               =  mfa_max_iter
        self.converged                  =  False
        # expectation of log determinant of precision matrices
        self.prec_logdet                =  [0]*self.n_components
        # log of determinant for scale matrices 
        self.scale_logdet               =  [0]*self.n_components
        # list of lower bounds (is updated at each iteration of algorithm)
        self.lower_bounds               =  [np.NINF]
        
        # DELTE AFTER TESTING
        self.e_log_like_list            = [np.NINF]
        self.e_log_qlat_list            = [np.NINF]   
        self.e_log_mp_list              = [np.NINF]
        self.e_log_qmp_list             = [np.NINF]   
        self.e_log_lat_list             = [np.NINF]         
        
        
           
    def _init_params(self,X):
        '''
        Initialise parameters
        '''
        self.n, self.d         = X.shape
        
        # Initialise parameters for all priors, these parameters are used in 
        # variational approximation at each iteration so they should be saved
        # and not changed
        if self.init_type is 'auto':
            # initialise prior on means & precision matrices
            if self.means0 is None:
               kms = KMeans(n_init = self.n_kmean_inits, n_clusters = self.n_components, 
                         random_state = self.rand_state)
               self.means0     = kms.fit(X).cluster_centers_
            # broad prior over precision matrix
            if self.scale_inv0 is None:
                # heuristics that seems to work pretty good
                diag_els        =  (np.max(X,0) - np.min(X,0))**2
                self.scale_inv0 = np.diag( diag_els  )
                self.scale0     = np.diag( 1./ diag_els )
            # initialise weights
            if self.weights0 is None:
                self.weights0  = np.ones(self.n_components) / self.n_components
        elif self.init_type is 'random':
            # randomly initialise prior for means and precision matrices
            if self.means0 is None:
               self.means0     = shuffle(X, n_samples = self.n_components, 
                                          random_state = self.rand_state)
            if self.scale0 is None:
               np.random.seed(self.rand_state)
               self.scale0     = np.diag(1e-3*np.random.random(self.d))
            # randomly initialise weights
            if self.weights0 is None:
               self.weights0    = np.random.random(self.n_components)
               self.weights0   /= np.sum(self.weights0)
        if self.dof0 is None:
            self.dof0           = self.d
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
        

    def _update_logresp_k(self, X, k, bound_calculate = False):
        '''
        Updates responsibilities for single cluster, calculates expectation
        of logdet of precision matrix.
        
        Parameters:
        -----------
        X: numpy array of size [n_samples,n_features] 
           Data matrix
           
        k: int
           Cluster index
           
        bound_calculate: boolean
           If True calculates part of lower bound
           
        Returns:
        --------
        log_pnk: numpy array of size [n_features,1]
                 Responsibilities without normalisation
        '''
        # calculate expectation of logdet of precision matrix
        scale_logdet   = np.linalg.slogdet(self.scale[k])[1]
        e_logdet_prec  = sum([psi(0.5*(self.dof[k]+1-i)) for i in range(1,self.d+1)])
        e_logdet_prec += scale_logdet + self.d*np.log(2)
        
        # save value of logdet for using in lower bound calculation
        if bound_calculate is True:
           self.prec_logdet[k]  = e_logdet_prec
           self.scale_logdet[k] = scale_logdet
           
        # calculate expectation of quadratic form (x-mean_k)'*precision_k*(x - mean_k)
        x_diff         = X - self.means[k,:]
        e_quad_form    = np.sum( np.dot(x_diff,self.scale[k,:,:])*x_diff, axis = 1 )
        e_quad_form   *= self.dof[k]
        
        # responsibilities without normalisation
        log_pnk        = np.log(self.weights[k]) + 0.5*e_logdet_prec - e_quad_form
        log_pnk       -= 0.5*self.d / self.beta[k]
        return log_pnk
                
                
    def _update_resps(self,X, bound_calculate = False):
        '''
        Updates distribution of latent variable (responsibilities)
        
        Parameters:
        -----------
        X: numpy array of size [n_samples,n_features] 
           Data matrix
           
        bound_calculate: boolean
           If True calculates part of lower bound
           
        Returns:
        --------
        
        
        '''
        # log of responsibilities before normalisaton
        log_p     = [self._update_logresp_k(X,k) for k in range(self.n_components)]
        log_sum_p = logsumexp(log_p,axis = 0, keepdims = True)
        log_p    -= log_sum_p
        p         = np.exp(log_p)
        # precompute values for lower bound 
        if bound_calculate is True:
            self.e_log_qlat = np.sum(np.log(p)*p)
        return p.T
    
    
    def _update_means_precisions(self, Nk, Xk, Sk):
        '''
        Updates distribution of means and precisions 
        '''
        for k in range(self.n_components):
            
            # update mean and precision for each cluster
            self.beta[k]   = self.beta0 + Nk[k]
            self.means[k]  = (self.beta0*self.means0[k,:] + Nk[k]*Xk[k]) / self.beta[k]
            self.dof[k]    = self.dof0 + Nk[k] + 1
            Xkdiff         = Xk[k] - self.means0[k,:]
            self.scale[k]  = pinvh(self.scale_inv0 + Nk[k]* ( Sk[k] + 
                             self.beta0/(self.beta0 + Nk[k])*np.outer(Xkdiff,Xkdiff) )) 
                             
    
    def _lower_bound(self,Nk,Xk,Sk):
        '''
        Lower bound and convergence check
        
        Parameters:
        -----------
        Nk: numpy array of size [n_components,1]
            Sum of responsibilities by component
        
        Xk: list of numpy arrays of length n_components
            Weighted average of observarions, weights are responsibilities
        
        Sk: list of numpy arrays of length n_components
            Weighted variance of observations, weights are responsibilities 
        '''
        e_log_like = 0    
        e_log_mp   = 0    
        e_log_lat  = 0    
        e_log_qmp  = 0    
        for k in range(self.n_components):
            
            # E[log(P(X| mu, prec, latent_variable))]
            x_diff_k    = Xk[k] - self.means[k,:]
            e_log_like += ( 0.5 * Nk[k]*(self.prec_logdet[k] - float(self.d)/self.beta[k] - 
                            self.d*(LOG_2 + LOG_PI) -
                            self.dof[k]*(np.trace(np.dot(Sk[k],self.scale[k])) -
                            self.dof[k]*np.sum(np.dot(x_diff_k,self.scale[k])*x_diff_k)) ) 
                          )
            # E[log(P(mu,precision))], prior of parameters
            m_diff_k    = self.means[k,:] - self.means0[k,:]
            e_log_mp   += ( 0.5 * self.d *(np.log(self.beta0) - LOG_PI - LOG_2) +
                            0.5 * self.prec_logdet[k] - 
                            0.5 * self.d * self.beta0 / self.beta[k] - 
                            0.5 *self.beta0 *self.dof[k] *np.dot(np.dot(m_diff_k,self.scale[k]),m_diff_k)-
                            0.5 * self.dof[k]*np.trace(np.dot(self.scale_inv0,self.scale[k])) +
                            0.5 * (self.dof0 - self.d - 1) * self.prec_logdet[k]
                          )
            # E[log(P(Z))], latent variable
            e_log_lat  += Nk[k]*np.log(self.weights[k])
            
            # E[log(Q(mu,precision))], apprximation to posterior of params
            e_log_qmp  += ( 0.5 * self.prec_logdet[k] +
                            0.5 * self.d * (np.log(self.beta[k]) - LOG_2 - LOG_PI) - 
                            0.5 * self.d - 
                            wishart_entropy(self.dof[k], self.scale[k], 
                                          self.prec_logdet[k],self.scale_logdet[k])
                          )
        
        #scale0_logdet    = np.linalg.slogdet(self.scale0)[1]
        scale0_logdet   = np.sum( np.log( np.diag( self.scale0 ) ) )
        e_log_mp       -= self.n_components * wishart_log_normaliser(self.scale0,
                                                                     self.dof0,
                                                                     scale0_logdet)
                                                                     
        lower_bound     = e_log_like + e_log_mp + e_log_lat - e_log_qmp - self.e_log_qlat
        
        self.e_log_like_list.append(e_log_like)
        self.e_log_qlat_list.append(-self.e_log_qlat) 
        self.e_log_mp_list.append(e_log_mp)
        self.e_log_qmp_list.append(-e_log_qmp) 
        self.e_log_lat_list.append(e_log_lat)        
        
        # lower bound is non-decreasing, otherwise there is error in software        
        #assert lower_bound >= self.lower_bounds[-1],'Lower Bound should be non-decreasing!'
        
        # check convergence         
        if lower_bound  - self.lower_bounds[-1] <= self.conv_thresh:
            self.converged = True
        self.lower_bounds.append(lower_bound)
        

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
        
        active = np.array([True for _ in range(self.n_components)])        
        # calculate responsibilities
        for j in range(self.max_iter):
            for i in range(self.mfa_max_iter):
                
                # if True, then mean-field approximation is completed
                end_approx =  i+1 == self.mfa_max_iter
                
                # STEP 1:   Approximate distribution of latent vatiable, means and 
                #           precisions using Mean Field Approximation method
                resps = self._update_resps(X, end_approx)
                
                # precalculate some intermediate statistics
                Nk     = np.sum(resps,axis = 0)
                #print [np.sum(resps[:,k:k+1]*X,0) for k in range(self.n_components)]
                Xk     = [np.sum(resps[:,k:k+1]*X,0)/Nk[k]  for k in range(self.n_components)]
                diff_x = [X - Xk[k] for k in range(self.n_components)]
                Sk     = [np.dot(resps[:,k]*diff_x[k].T,diff_x[k])/Nk[k] for  \
                          k in range(self.n_components)]
                          
                # update means & precisions for each cluster
                self._update_means_precisions(Nk,Xk,Sk)
                
                # STEP 2: Maximize lower bound with respect to weights, prune
                #         clusters with small weights & calculate lower bound 
                if end_approx is True:
                    
                    # update weights to maximize lower bound  
                    self.weights      = Nk / self.n
                    
                    # calculate lower bound
                    self._lower_bound(Nk,Xk,Sk)
                    
                    # prune all iirelevant weights
                    active            = self.weights > self.prune_thresh
                    self.means        = self.means[active,:]
                    self.means0       = self.means0[active,:]
                    self.scale        = self.scale[active,:,:]
                    self.weights      = self.weights[active]
                    self.weights     /= np.sum(self.weights)
                    self.dof          = self.dof[active]
                    self.beta         = self.beta[active]
                    self.n_components = np.sum(active)

        return resps
        
    def predictive_dist(self,x):
        '''
        
        '''
        pass
    
    
     #----------------------  Getter & Setter methods ------------------------   
        
    def get_params(self):
        '''
        Returns disctionary with all learned parameters
        '''
        covars = [pinvh(sc*df) for sc,df in zip( self.scale, self.dof)]
        params = {'means': self.means, 'covars': covars,'weights': self.weights}
        return params



class VBGMMARDClassifier(object):
    
    def __init__(self):
        pass
    
    def fit(self):
        pass
    
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
    
    
class VBGMMARDNoveltyDetector(object):
    
    def __init__(self,components):
        self.components = components
    
    def fit(self,X,Y):
        pass
    

        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    X = np.zeros([300,2])
    # [1,1]
    X[0:100,0] = np.random.normal(1,3,100)
    X[0:100,1] = np.random.normal(1,3,100) 
    # [12,5]
    X[100:200,0] = np.random.normal(12,2,100)
    X[100:200,1] = np.random.normal(5,2,100) 
    # [-4,-13]
    X[200:300,0] = np.random.normal(-4,1,100)
    X[200:300,1] = np.random.normal(-13,2,100)
    plt.plot(X[:,0],X[:,1],'ro')
    plt.show()
    vbgmm = VBGMMARD(max_components = 10,init_type = 'auto')
    resps = vbgmm.fit(X)
    
    print "real means"
    print np.mean(X[0:100,:],0)
    print np.mean(X[100:200,:],0)
    print np.mean(X[200:300,:],0)
    
    print 'means by algorithm'
    print vbgmm.means
    
    print 'covariance by algorithm'
    print vbgmm.get_params()
    
    # Old Faithful Data
    import os
    import pandas as pd
    
    os.chdir("/Users/amazaspshaumyan/Desktop/MixtureExperts/Variational Bayesian GMM with ARD/")
    Data = pd.read_csv("old_faithful.txt")
    vbgmm_of = VBGMMARD(max_components = 3, init_type = 'auto')
    r = vbgmm_of.fit(np.array(Data))
    plt.plot(Data['eruptions'],Data['waiting'],'bo')
    plt.plot(vbgmm_of.means[:,0],vbgmm_of.means[:,1],'ro')
    plt.show()
    
    print "Selected number of clusters {0}".format(vbgmm_of.means.shape[0])
    
    plt.plot(vbgmm.e_log_like_list,'b-')
    plt.title("log-like")
    plt.show()
    
    plt.plot(vbgmm.e_log_mp_list,'b-')
    plt.title("log mean - precision")
    plt.show()
    
    plt.plot(vbgmm.e_log_lat_list,'b-')
    plt.title("log latent variable")
    plt.show()
    
    plt.plot(vbgmm.e_log_qmp_list,'b-')
    plt.title("log approx mean-precision")
    plt.show()
    
    plt.plot(vbgmm.e_log_qlat_list,'b-')
    plt.title("log approx latent")
    plt.show()
    
    
    
            
            
            
            

    
    
    