# -*- coding: utf-8 -*-

import numpy as np


def bounded_variable(x,lo,hi):
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
        z[ z > hi] = hi
        z[ z < lo] = lo
        return z
    if len(np.shape(x)) > 1:
        for i in range(np.shape(x)[1]):
            x[:,i] = _bounded_vector(x[:,i],lo,hi)
        return x
    return _bounded_vector(x,lo,hi)
    
    
    
class NodeNotFoundError(Exception):
    '''
    Error raised in case node is not found
    '''
    
    def __init__(self,node_position, node_type, message):
        self.np = node_position
        self.nt = node_type
        self.m  = message
        
    def __str__(self):
        return " ".join(["Node with index ",str(self.np)," of type ",str(self.nt),self.m])
