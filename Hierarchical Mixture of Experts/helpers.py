# -*- coding: utf-8 -*-

import numpy as np
import random



def train_test_split(x,y,test_p = 0.25):
    '''
    Divides data set into train and test data sets
    
    Parameters:
    ----------
    
    x: numpy array of size 'n x k' (k can be 1 and above)
       Exogeneous variables
       
    y: numpy array of size 'n  x m' (m can be 1 and above)
       Endogeneous variables
       
    test_p: float
       Proportion of data that should go to testing set
       
    Returns:
    --------
    
    [x_train,x_test]: list of size 2
       First element of list is training set, second is testing set
        
    '''
    n            = x.shape[0]  
    sample_index = random.sample(xrange(n), int(n*test_p))
    train_index  = [e for e in xrange(n) if e not in set(sample_index)]
    
    if len(x.shape) > 1:
        x_test       = x[sample_index,:]
        x_train      = x[train_index,:]
        
    if len(y.shape) > 1:
        y_test       = y[sample_index,:]
        y_train      = y[train_index,:]
        
    if len(x.shape) == 1:
        x_test       = x[sample_index]
        x_train      = x[train_index]
        
    if len(y.shape) == 1:
        y_test       = y[sample_index]
        y_train      = y[train_index]
        
    return [x_train,x_test,y_train,y_test]
    
    


def bounded_variable(x,lo,hi=None):
    '''
    Bounds variable from below and above, prevents underflow and overflow
    
    Parameters:
    -----------
    
    x: numpy array of size 'n x k' (k can be 1)
       input vector
       
    hi: float
       Upper bound
       
    lo: float
       Lower bound
       
    Returns:
    --------
    : numpy array of size 'n x k'
       
    '''
    def _bounded_vector(z,lo,hi):
        if hi is not None:
            z[ z > hi] = hi
        z[ z < lo] = lo
        return z
    if len(np.shape(x)) > 1:
        for i in range(np.shape(x)[1]):
            pass
            x[:,i] = _bounded_vector(x[:,i],lo,hi)
        return x
    return _bounded_vector(x,lo,hi)
    
    
    
class NodeNotFoundError(LookupError):
    '''
    Error raised in case node is not found
    '''
    
    def __init__(self,n_pos,n_type, message):
        m            = "Node with index {0} of type {1} {2}"
        self.message = m.format(n_pos,n_type,message)
        
    def __str__(self):
        return self.message
        
class NodeModelNotImplemented(NotImplementedError):
    '''
    Error raised in case model is not implemented for node
    '''
    
    def __init__(self,model_name,n_type):
        m            = "Model {0} is not implemented for node type {1}"
        self.message = m.format(model_name, n_type)
        
    def __str__(self):
        return self.message
