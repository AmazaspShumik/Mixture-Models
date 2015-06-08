# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 19:07:34 2015

@author: amazaspshaumyan
"""

class GatingNetwork(object):
    
    def __init__(self,weights,X,Y,k):
        self._weights = weights
        self._X       = X
        self._Y       = target_preprocessing(Y,k)
        self.Theta    = np.zeros([1,1])
        
    def optimize(self):
        
    def get_theta(self):
        return self.Theta