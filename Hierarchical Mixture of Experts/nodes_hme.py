# -*- coding: utf-8 -*-

import numpy as np
import abc
import WeightedLinearRegression as wlr
import SoftmaxRegression as sr
import 




#-------------------------------  Base Node Class --------------------------------

class Node(object):
    __metaclass__ = abc.ABCMeta
    '''
    Parameters:
    -----------
    weights: numpy array of size n
        Vector of weights for weighted inner and expert nodes
        
    node_position: int
        Position of this node in heap array
        
    k: int 
        Branching parameter of the tree
    '''
    
    def __init__(self,n,node_position,k, underflow_tol = 1e-2):
        self.weights       = np.ones(n)
        self.node_position = node_position
        self.k             = k
        self.underflow_tol = underflow_tol
        
        
    @abc.abstractmethod
    def m_step_update(self):
        raise NotImplementedError
        
        
    @abc.abstractmethod
    def up_tree_pass(self):
        raise NotImplementedError
        
        
    @abc.abstractmethod
    def down_tree_pass(self):
        raise NotImplementedError
        
        
    @abc.abstractmethod
    def prior(self):
        raise NotImplementedError
        
    @abc.abstractmethod
    def propagate_mean_prediction(self):
        raise NotImplementedError
        
        
    def get_childrens(self,nodes):
        children_nodes = []
        for i in range(1,self.k+1):
            children_nodes.append(nodes[self.node_position*self.k + i])
        return children_nodes
        
        
    def get_parent_and_birth_order(self,nodes):
        parent_index      =  (self.node_position - 1) / self.k
        birth_order       =  (self.node_position - 1) % self.k
        parent       =  nodes[parent_index]
        return [parent, birth_order]
        
        
    def has_parent(self):
        if self.node_position == 0:
            return False
        return True
        
        
#-------------------------------- Inner Node --------------------------------------#
        
        
class GaterNode(Node):
    
    
    def __init__(self,n,node_position,k,m,tolerance = 1e-5, max_iter = 100):
        '''
        Parameters:
        -----------
        
        n: int
           Number of observations in data set
           
        m: int 
           Dimensionality of data
           
        k: int 
           Branching parameter
          
        '''
        super(GaterNode,self).__init__(n, node_position,k)
        self.gater = sr.SoftmaxRegression(tolerance, max_iter)
        self.gater.init_weights(m,k)
        self.responsibilities = np.zeros([n,k])
        self.node_type = "gate"
        
        
    def m_step_update(self,H,X):
        ''' Updates parameters'''
        # parameters are updated and saved in gater
        self.gater.fit_matrix_output(H,X,self.weights) 
        
    
    def prior(self,X):
        '''Calculates  prior probs'''
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
    
    def m_step_update(self,X,Y):
        ''' Updates parameters '''
        # parameters are updated and saved in expert
        self.expert.fit(X,Y,self.weights)
        
    def down_tree_pass(self,X,Y, nodes):
        parent, birth_order = self.get_parent_and_birth_order(nodes)
        self.weights        = parent.weights * parent.responsibilities[:,birth_order]/parent.normaliser
        self.m_step_update(X,Y)
        
    def propagate_mean_prediction(self,X,nodes):
        return self.expert.predict(X)
        
        
#-------------------------------------- Linear Regression Expert Node --------------------------------------------

        
class ExpertNodeLinReg(ExpertNodeAbstract):
    
    def __init__(self,n,node_position,k,m):
        super(ExpertNodeLinReg,self).__init__(n,node_position,k)
        self.expert = wlr.WeightedLinearRegression()
        self.expert.init_weights(m)
        self.node_type = "expert"
        
    def prior(self,X,Y):
        ''' Calculates probability of observing'''
        self.weights = wlr.norm_pdf(self.expert.theta,Y,X,self.expert.var)
        self.weights = bounded_variable(self.weights,self.underflow_tol, 1-self.underflow_tol)
        
    def up_tree_pass(self,X,Y):
        self.prior(X,Y)
        

#-------------------------------------- Logistic Regression Expert Node --------------------------------------------
        
    
class ExpertNodeLogisticReg(object):
    
    def __init__
        
        
        
#----------------------------------------- Helper Methods ----------------------------------------------#
        
def bounded_variable(x,lo,hi):
    '''
    Returns 'x' if 'x' is between 'lo' and 'hi', 'hi' if x is larger than 'hi'
    and 'lo' if x is lower than 'lo'
    '''
    x[ x > hi] = hi
    x[ x < lo] = lo
    return x
        
if __name__=="__main__":
        pass
    
        
    
        
        