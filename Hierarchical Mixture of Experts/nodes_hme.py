# -*- coding: utf-8 -*-

import numpy as np
import abc
import weighted_lin_reg as wlr
import softmax_reg as sr
import weighted_gda as wgda
from scipy.misc import logsumexp
from helpers import *


########################################## Abstarct Base Node Class ######################################################




class Node(object):
    __metaclass__ = abc.ABCMeta
    '''
    Abstract base class for gating and expert nodes.
    
    
    Parameters:
    -----------
    n: int
        Number of observations in training set
        
    node_position: int
        Position of this node in heap array
        
    k: int 
        Branching parameter of the tree
        
    m: int
        Dimensionality of input data
        
    bias: bool
        True if X contains bias term
        
    underflow_tol: float
        Smallest value probability can take when calculating responsibilities,
        prevents underflow
        
    stop_learning: float
        Does not allow expert or gater model to learn if change in likelihood 
        is small
        
    max_iter: int
        Number of iterations before convergence (for softmax regression in gates)
        
    conv_threshold: float
        Convergence parameter (for softmax regression in gates)
        
    stop_learning_softmax: float
        If change in weighted log_likelihood of softmax is smaller than threshold
        
    '''
    
    def __init__(self,n,node_position,k,m, bias_term = True,    underflow_tol            = 1e-10,
                                                                classes                  = 2,
                                                                max_iter                 = 100, 
                                                                conv_threshold           = 1e-10,
                                                                stop_learning_softmax    = 1e-10,
                                                                stop_learning_regression = 1e-20,
                                                                stop_learning_wgda       = 1e-20): 
        self.weights            = np.zeros(n, dtype = np.float64)
        self.bound_weights      = np.zeros(n, dtype = np.float64)
        self.node_position      = node_position
        self.k                  = k
        self.underflow_tol      = underflow_tol
        self.m                  = m
        self.bias               = bias_term
        self.max_iter           = max_iter
        self.conv_threshold     = conv_threshold
        self.n                  = n
        # log-likelihood
        self.log_like_test      = 0
        self.stop_learning_sr   = stop_learning_softmax
        self.stop_learning_wlr  = stop_learning_regression
        self.stop_learning_wgda = stop_learning_wgda
        self.classes            = classes
        
        
    @abc.abstractmethod
    def _m_step_update(self):
        pass
        
        
    @abc.abstractmethod
    def up_tree_pass(self):
        pass
        
        
    @abc.abstractmethod
    def down_tree_pass(self):
        pass
        
        
    @abc.abstractmethod
    def _prior(self):
        pass
        
        
    @abc.abstractmethod
    def propagate_prediction(self):
        pass
        
        
    def get_childrens(self,nodes):
        '''
        Gets children of current node.
        
        Parameters:
        -----------
        
        nodes: list of size equal number of nodes in HME
             List with all nodes of HME
             
        Returns:
        --------
        
        children_nodes: list of size k (branching factor of tree)
             List of children of node
        '''
        children_nodes = []
        for i in range(1,self.k+1):
            child_position = self.node_position*self.k + i
            if child_position >= len(nodes):
               raise NodeNotFoundError(self.node_position,self.node_type,"does not have children")
            children_nodes.append(nodes[child_position])
        return children_nodes
        
        
    def get_parent_and_birth_order(self,nodes):
        '''
        Gets parent of current node and finds number of children to the left.
        
        Parameters:
        -----------
        
        nodes: list of size equal number of nodes in HME
             List with all nodes of HME
             
        Returns:
        --------
        
        [parent,birth_order]: list 
             First element of list os parent of node, second identifies child position
        '''
        parent_index      =  (self.node_position - 1) / self.k
        if parent_index < 0:
            raise NodeNotFoundError(self.node_position,self.node_type,"does not have parent")
        birth_order       =  (self.node_position - 1) % self.k
        parent            =  nodes[parent_index]
        return [parent, birth_order]
        
        
    def has_parent(self):
        '''
        Returns True if node has parent, False if otherwise
        '''
        if self.node_position == 0:
            return False
        return True
        
        
    def get_delta_param_norm(self):
        ''' L2 norm of change in parameters of gate model'''
        return self.model.delta_param_norm
        
        
    def get_delta_log_like(self):
        ''' Returns change in likelihood on m-step'''
        return self.model.delta_log_like
        
        
    
############################################### Gate Node ################################################################


#----------------------------------------- Abstarct Gater Class ---------------------------------------------------------#



class AbstractGaterNode(Node):
    '''
    Abstract gate node class
    '''
    
    def __init__(self,*args,**kwargs):
        super(AbstractGaterNode,self).__init__(*args,**kwargs)
        self.responsibilities = np.zeros([self.n,self.k])
        self.normaliser       = np.zeros(self.n)
        self.node_type        = "gate"

    
    def down_tree_pass(self,X,nodes):
        '''
        Calculates responsibilities and performs weighted maximum 
        likelihood estimation
        
        Parameters:
        -----------
        
        X: numpy array of size 'n x m'
            Explanatory variables
            
        nodes: list of size equal number of nodes in HME
             List with all nodes of HME
             
        '''
        # E-step of EM algorithm
        if self.has_parent() is True:
            parent,birth_order                   = self.get_parent_and_birth_order(nodes)
            self.weights                         = parent.responsibilities[:,birth_order] - parent.normaliser
            self.weights                        += parent.weights
        log_H = self.responsibilities - np.outer(self.normaliser, np.ones(self.k))
        H     = np.exp(log_H)
        
        # bound weights to prevent underflow in weighted regression
        self.bound_weights =  bounded_variable(np.exp(self.weights),self.underflow_tol)
        
        # M-step of EM algorithm
        self._m_step_update(H,X)

        
    def up_tree_pass(self,X,nodes):
        '''
        Calculates prior probability of latent variables and combines 
        prior probability of children to calculate posterior for the 
        latent variable corresponding to node
        
        Parameters:
        -----------
        
        X: numpy array of size 'n x m'
            Explanatory variables
            
        nodes: list of size equal number of nodes in HME
             List with all nodes of HME
             
        '''
        self._prior(X)
        children = self.get_childrens(nodes)
        
        # check that all children are of the same type
        if len(set([e.node_type for e in children])) != 1:
               raise ValueError("Children nodes should have the same node type")
               
        # prior probabilities calculation
        for i,child_node in enumerate(children):
            if child_node.node_type == "expert":
               self.responsibilities[:,i] += child_node.weights
            elif child_node.node_type == "gate":
               self.responsibilities[:,i] += logsumexp(child_node.responsibilities, axis = 1)
            else:
                raise TypeError("Unidentified node type")
                
        #prevent underflow
        self.normaliser         = logsumexp(self.responsibilities, axis = 1)
    
    
    def propagate_prediction(self,X,nodes,predict_type = "predict_response", y_lo=None, y_hi=None):
        '''
        Returns weighted mean of predictions in experts which are in subtree
        
        Parameters:
        -----------
        
        X: numpy array of size 'unkonwn x m'
            Explanatory variables for test set
            
        nodes: list of size equal number of nodes in HME
             List with all nodes of HME
             
        predict_type: str
             Can be "predict_response", "predict_prob", "predict_cdf"
             "predict_resposne"   - works for all type of experts 
             "predict_prob"       - works for classification experts ('wgda','softmax')
             "predict_cdf"        - works only for 'gaussian' expert
            
        Returns:
        --------
        
        mean_prediction: numpy array of size 'unknown x m'
             Weighted prediction
        '''
        self._prior(X)
        children        = self.get_childrens(nodes)
        n,m             = np.shape(X)
        mean_prediction = None
        for i,child in enumerate(children):
            w                   = np.exp(self.responsibilities[:,i])
            children_average    = child.propagate_prediction(X,nodes,predict_type,y_lo,y_hi)
            if len(children_average.shape) > 1:
                k                = children_average.shape[1]
                w                = np.outer(w,np.ones(k))
            if mean_prediction is None:
                mean_prediction  = (w * children_average)
            else:
                mean_prediction += (w * children_average)
        return mean_prediction
        

    def _m_step_update(self,H,X):
        ''' Updates parameters running weighted softmax regression '''
        self.model.fit(H,X,self.bound_weights)
        
    
    def _prior(self,X):
        '''Calculates  prior probabilities for latent variables'''
        probs = self.model.predict_log_probs(X)
        self.responsibilities = probs
        
        
#----------------------------------------- implementations of Gaters ---------------------------------------------#
    
        
class GaterNodeSoftmax(AbstractGaterNode):
    '''
    Gate node of Hierarchical Mixture of Experts with softmax transfer function.
    Calculates responsibilities and updates parmameters using weighted softmax regression.
    '''
    
    def __init__(self,*args,**kwargs):
        ''' Initialises gate node '''
        super(GaterNodeSoftmax,self).__init__(*args,**kwargs)
        self.model = sr.SoftmaxRegression(self.conv_threshold, self.max_iter,self.stop_learning_sr)
        self.model.init_params(self.m,self.k)
        
        
class GaterNodeWGDA(AbstractGaterNode):
    '''
    Gate node of Hierarchical Mixture of Experts with weighted gaussian discriminant
    analysis as gating model. Calculates responsibilities and updates parameters 
    of gating model.
    '''
    
    def __init__(self,*args,**kwargs):
        ''' Initialises gate node '''
        super(GaterNodeWGDA,self).__init__(*args,**kwargs)
        self.model = wgda.WeightedGaussianDiscriminantAnalysis(bias_term     = self.bias, 
                                                               stop_learning = self.stop_learning_wgda)
        if self.bias is True:
            self.model.init_params(self.m-1,self.k)
        else:
            self.model.init_params(self.m,self.k)


 
################################################## Expert Nodes ##########################################################
 
 
#----------------------------------------- Abstarct Expert Class ---------------------------------------------------------#
      
      
      
class ExpertNodeAbstract(Node):
    '''
    Abstract Base Class for experts (linear, logistic etc. regressions) 
    '''

    def down_tree_pass(self,X,Y,nodes):
        '''
        Calculates responsibilities and performs weighted maximum likelihood
        estimation.
        
        Parameters:
        -----------
        
        X: numpy array of size 'n x m'
            Explanatory variables
            
        Y: numpy array of size 'n x m'
            Target variables that should be approximated
            
        nodes: list of size equal number of nodes in HME
             List with all nodes of HME
             
        '''
        # E-step of EM algorithm
        parent, birth_order = self.get_parent_and_birth_order(nodes)
        
        self.weights           =  parent.responsibilities[:,birth_order] - parent.normaliser
        self.weights          += parent.weights 
        
        # prevent underflow in weighted regressions
        self.bound_weights = bounded_variable(np.exp(self.weights),self.underflow_tol)
        
        # M-step of EM algorithm
        self._m_step_update(X,Y)

       
    def up_tree_pass(self,X,Y):
        '''
        Calculates prior probability of latent variables corresponding to 
        expert at node and likelihood. 
        
        Parameters:
        -----------
        
        X: numpy array of size 'n x m'
            Explanatory variables
            
        Y: numpy array of size 'n x 1'
             Target variable that should be approximated
             
        '''
        self._prior(X,Y)
        
                
    def propagate_prediction(self,X,nodes, predict_type = "mean",y_lo=None,y_hi=None):
        '''
        Returns prediction of expert for test input X
        
        Parameters:
        -----------
        
        X: numpy array of size 'unkonwn x m'
            Explanatory variables for test set
            
        nodes: list of size equal number of nodes in HME
             List with all nodes of HME
             
        predict_type: str
             Can be "predict_response", "predict_prob", "predict_cdf"
             "predict_resposne"   - works for all type of experts 
             "predict_prob"       - works for classification experts ('wgda','softmax')
             "predict_cdf"        - works only for 'gaussian' expert
             
        Returns:
        --------
        : numpy array of size 'unknown x m'
             Weighted prediction
        
        '''
        if predict_type == "predict_probs":
            return self.model.predict_probs(X)
        elif predict_type == "predict_response":
            return self.model.predict(X)
        elif predict_type == "predict_cdf":
            return self.model.posterior_cdf(X,y_lo,y_hi)
        else:
            raise NotImplementedError("Not implemented prediction type")
        
        
    def propagate_log_probs(self,X,Y):
        ''' Returns probability of observing Y given X and parameters'''
        return self.model.posterior_log_probs(X,Y)
     
         
    def _prior(self,X,Y):
        ''' Calculates probability of observing Y given X and parameters of regression '''
        self.weights = self.model.posterior_log_probs(X,Y)
        

    def _m_step_update(self,X,Y):
        ''' Updates parameters of linear regression (coefficient and estimates of variance) '''
        # parameters are updated and saved in expert 
        self.model.fit(Y,X,self.bound_weights)

        
        
#-------------------------------------- Implementation of Expert Nodes --------------------------------------------------#

        
class ExpertNodeLinReg(ExpertNodeAbstract):
    '''
    Expert node in Hierarchical Mixture of Experts, with expert being 
    standard weighted linear regression.
    '''
    
    def __init__(self,*args,**kwargs):
        ''' Initialise linear regression expert node '''
        super(ExpertNodeLinReg,self).__init__(*args,**kwargs)
        self.model = wlr.WeightedLinearRegression(stop_learning = self.stop_learning_wlr)
        self.model.init_params(self.m)
        self.node_type = "expert"
        
    
class ExpertNodeSoftmaxReg(ExpertNodeAbstract):
    '''
    Expert Node with Softmax model as an expert
    '''

    def __init__(self,*args, **kwargs):
        super(ExpertNodeSoftmaxReg,self).__init__(*args,**kwargs)
        self.model = sr.SoftmaxRegression( tolerance       = self.conv_threshold, 
                                             max_iter      = self.max_iter,
                                             stop_learning = self.stop_learning_sr)
        self.model.init_params(self.m, self.classes)
        self.node_type = "expert"
               
        
        
class ExpertNodeWGDA(ExpertNodeAbstract):
    '''
    Expert Node with Gaussian Discriminant Analysis as an expert
    '''
    
    def __init__(self,*args,**kwargs):
        super(ExpertNodeWGDA,self).__init__(*args,**kwargs)
        self.model = wgda.WeightedGaussianDiscriminantAnalysis(stop_learning = self.stop_learning_wgda,
                                                               bias_term     = self.bias)
        if self.bias is True:
            self.model.init_params(self.m-1,self.classes)
        else:
            self.model.init_params(self.m,self.classes)
        self.node_type ="expert"
        
        

