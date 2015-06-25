# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-


import numpy as np
from scipy.sparse import csr_matrix

class ClassificationTargetError(Exception):
    '''
    Exception raised in case of mismatch between number of expected and 
    observed classes in target vector (for classification problem)
    '''
    
    def __init__(self, expected, observed):
        self.e = expected
        self.o = observed
        
    def __str__(self):
        s = 'Mismatch in number of classes, expected  - {0} , observed - {1}'.format(self.e,self.o) 
        return s
        


class LabelBinariser(object):
    
    '''
    Binarize labels in a one-vs-all fashion.
    
    Allows easily transform vector of targets for classification to ground truth 
    matrix and easily make inverse transformation.
    
    n = n_samples , k = n_classes
    
    Parameters:
    ------------
    
    Y: numpy array of size 'n_samples x 1'
       Target variables, vector of classes in classification problem
    
    k: int
       Number of classes 
    
    '''
    
    def __init__(self,Y,k):
        
        self.Y          = Y
        self.n          = np.shape(Y)[0]
        self.k          = k
        
        #  mapping between set of integers to set of classes
        classes         = set(Y)
        if len(classes) != k:
            raise ClassificationTargetError(k,len(classes))
        self.direct_mapping  = {}
        self.inverse_mapping = {}
        for i,el in enumerate(classes):
            print i,el
            self.direct_mapping[el] = i
            self.inverse_mapping[i] = el
            
            
    def convert_vec_to_binary_matrix(self, compress = False):
        '''
        Converts vector to ground truth matrix
        
        Parameters:
        ------------
                
        compress: bool
               If True will use csr_matrix to output compressed matrix
                  
        Returns:
        --------
        
        Y: numpy array of size 'n x k'
               Ground truth matrix , column number represents class index,
               each row has all zeros and only one 1.  
                
        '''
        Y = np.zeros([self.n,self.k])
        for el,idx in self.direct_mapping.items():
            Y[self.Y==el,idx] = 1
        if compress is True:
            return csr_matrix(Y)
        return Y
            
            
    def convert_binary_matrix_to_vec(self,B, compressed = False):
        '''
        Converts ground truth matrix to vector of classificaion targets
        
        Parameters:
        -----------
        compressed: bool
             If True input is csr_matrix, otherwise B is numpy array
            
        Returns:
        ---------
        
        Y: numpy array of size 'n x 1'
            Vector of targets, classes
        '''
        if compressed is True:
            B = B.dot(np.eye(np.shape(B)[1]))
        Y = np.zeros(self.n, dtype = self.Y.dtype)
        for i in range(np.shape(B)[1]):
            Y[B[:,i]==1] = self.inverse_mapping[i]
        return Y
        
        
    def convert_prob_matrix_to_vec(self,Y):
        '''
        Converts matrix of probabilities to vector of classification targets
        
        Parameters:
        -----------
        Y:  numpy array of size [n_samples,n_classes]
            Matrix of class probabilities, element at cell [i,j] shows probability
            that observation i belongs to class j
            
        Returns:
        --------
        
        Y: numpy array of size 'n x k'
            Ground truth matrix , column number represents class index,
            each row has all zeros and only one 1.
            
        '''
        Y_max = np.argmax(Y, axis = 1)
        Y     = np.array([self.inverse_mapping[e] for e in Y_max])
        return Y
        
        