# -*- coding: utf-8 -*-

import numpy as np
import abc
import WeightedLinearRegression as wlr
import SoftmaxRegression as sr
import logistic_reg as lr




#-------------------------------  Base Node Class --------------------------------

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
        
    underflow_tol: float
        Samllest value probability can take when calculating responsibilities,
        prevents underflow
        
    max_iter: int
        Number of iterations before convergence (for softmax regression in gates)
        
    conv_threshold: float
        Convergence parameter (for softmax regression in gates)
        
    '''
    
    def __init__(self,n,node_position,k,m,underflow_tol = 1e-2,max_iter = 100, conv_threshold = 1e-5):
        self.weights         = np.ones(n)
        self.node_position   = node_position
        self.k               = k
        self.underflow_tol   = underflow_tol
        self.m               = m
        self.max_iter        = max_iter
        self.conv_threshold  = conv_threshold
        
        
    @abc.abstractmethod
    def _m_step_update(self):
        raise NotImplementedError
        
        
    @abc.abstractmethod
    def up_tree_pass(self):
        raise NotImplementedError
        
        
    @abc.abstractmethod
    def down_tree_pass(self):
        raise NotImplementedError
        
        
    @abc.abstractmethod
    def _prior(self):
        raise NotImplementedError
        
    @abc.abstractmethod
    def propagate_mean_prediction(self):
        raise NotImplementedError
        
        
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
        parent       =  nodes[parent_index]
        return [parent, birth_order]
        
        
    def has_parent(self):
        '''
        Returns True if node has children, False if otherwise
        '''
        if self.node_position == 0:
            return False
        return True
        
        
#-------------------------------- Inner Node --------------------------------------#
        
        
class GaterNode(Node):
    '''
    Gate node of Hierarchical Mixture of Experts.
    Calculates responsibilities and updates parmameters using weighted softmax regression.
    '''
    
    
    def __init__(self,*args,**kwargs):
        ''' Initialises gate node '''
        super(GaterNode,self).__init__(*args,**kwargs)
        self.gater = sr.SoftmaxRegression(self.conv_threshold, self.max_iter)
        self.gater.init_weights(self.m,self.k)
        self.responsibilities = np.zeros([self.n,self.k])
        self.node_type = "gate"
        
        
    def _m_step_update(self,H,X):
        ''' Updates parameters of softmax regression'''
        # parameters are updated and saved in gater
        self.gater.fit_matrix_output(H,X,self.weights) 
        
    
    def _prior(self,X):
        '''Calculates  prior probabilities for latent variables'''
        self.responsibilities = sr.softmax(self.gater.theta,X)
        
        
    def up_tree_pass(self,X,nodes):
        self.prior(X)
        #print self.responsibilities
        children = self.get_childrens(nodes)
        # all children should be of the same type
        if len(set([e.node_type for e in children])) != 1:
               raise ValueError("Children nodes should have the same node type")         
        for i,child_node in enumerate(children):
            if child_node.node_type == "expert":
               self.responsibilities[:,i] *= child_node.weights
            elif child_node.node_type == "gate":
               #print "Gater"
               self.responsibilities[:,i] *= np.sum(child_node.responsibilities, axis = 1)
               #print self.responsibilities
            else:
                raise TypeError("Unidentified node type")
        #print self.responsibilities
        self.normaliser = np.sum(self.responsibilities, axis = 1)
        
        
    def down_tree_pass(self,X,nodes):
        if self.has_parent() is True:
            parent,birth_order = self.get_parent_and_birth_order(nodes)
            self.weights       = parent.weights*parent.responsibilities[:,birth_order]/parent.normaliser
        H = self.responsibilities / np.outer(self.normaliser, np.ones(self.k))
        self.m_step_update(H,X)
        
        
    def propagate_mean_prediction(self,X,nodes):
        self.prior(X)
        children        = self.get_childrens(nodes)
        n,m             = np.shape(X)
        mean_prediction = np.zeros(n)
        for i,child in enumerate(children):
            mean_prediction+= (self.responsibilities[:,i] * child.propagate_mean_prediction(X,nodes))
        return mean_prediction
        
        
      
############################################## Expert Nodes #####################################################
      
      
class ExpertNodeAbstract(Node):
    
    def _m_step_update(self,X,Y):
        ''' Updates parameters of linear regression (coefficient and estimates of variance) '''
        # parameters are updated and saved in expert
        self.expert.fit(X,Y,self.weights)
        
    def down_tree_pass(self,X,Y, nodes):
        '''
        
        
        '''
        parent, birth_order = self.get_parent_and_birth_order(nodes)
        self.weights        = parent.weights * parent.responsibilities[:,birth_order]/parent.normaliser
        self.m_step_update(X,Y)
        
    def propagate_mean_prediction(self,X,nodes):
        '''
        
        '''
        return self.expert.predict(X)
        
        
#-------------------------------------- Linear Regression Expert Node --------------------------------------------

        
class ExpertNodeLinReg(ExpertNodeAbstract):
    
    def __init__(self,*args,**kwargs):
        ''' Initialise linear regression expert node '''
        super(ExpertNodeLinReg,self).__init__(*args,**kwargs)
        self.expert = wlr.WeightedLinearRegression()
        self.expert.init_weights(self.m)
        self.node_type = "expert"
        
    def _prior(self,X,Y):
        ''' Calculates probability of observing Y given X and parameters of regression '''
        self.weights = wlr.norm_pdf(self.expert.theta,Y,X,self.expert.var)
        self.weights = bounded_variable(self.weights,self.underflow_tol, 1-self.underflow_tol)
        
    def up_tree_pass(self,X,Y):
        self.prior(X,Y)
        

#-------------------------------------- Logistic Regression Expert Node --------------------------------------------
        
    
class ExpertNodeLogisticReg(ExpertNodeAbstract):
    
    def __init__(self,n,node_position,k,m):
        pass
        
        
        
#----------------------------------------- Helper Methods & Classes ----------------------------------------------#
        
def bounded_variable(x,lo,hi):
    '''
    Returns 'x' if 'x' is between 'lo' and 'hi', 'hi' if x is larger than 'hi'
    and 'lo' if x is lower than 'lo'
    '''
    x[ x > hi] = hi
    x[ x < lo] = lo
    return x
    
class NodeNotFoundError(Exception):
    
    def __init__(self,node_position, node_type, message):
        self.np = node_position
        self.nt = node_type
        self.m  = message
        
    def __str__(self):
        return " ".join(["Node with index ",str(self.np)," of type ",str(self.nt),self.m])

        
        