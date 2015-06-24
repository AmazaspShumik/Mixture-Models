# -*- coding: utf-8 -*-


import numpy as np
from scipy.sparse import CSR

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
    
    
    '''
    
    def __init__(self,Y,k):
        
        self.Y          = Y
        self.n          = np.shape(Y)[0]
        self.k          = k
        
        #  mapping between set of integers to set of classes
        classes         = set(Y)
        if len(classes) != k:
            raise ClassificationTargetError(k,len(classes))
        direct_mapping  = {}
        inverse_mapping = {}
        for i,el in enumerate(classes):
            direct_mapping[el] = i
            inverse_mapping[i] = el
            
    def convert_vec_to_binary_matrix(self):
        '''
        Converts vector to ground truth matrix
        '''
        Y = np.zeros([self.n,self.k])
        for el,idx in self.direct_mapping:
            Y[self.Y==el,idx] = 1
        return Y
            
            
    def convert_binary_matrix_to_vec(self,B):
        Y = np.zeros(self.n, dtype = self.Y.dtype)
        for i in range(np.shape(B)[1]):
            Y[B[:,i]==1] = self.inverse_mapping[i]
        return Y
        
if __name__=="__main__":
    Y = np.array(["1","1","0","0","0"])
    lb = LabelBinariser(Y)
    Y_hat = lb.convert_vec_to_binary_matrix()
    print Y_hat
        
        
        
        
            

        
        