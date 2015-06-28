# -*- coding: utf-8 -*-

import numpy as np


def logistic_sigmoid(M):
    '''
    Sigmoid function
    '''
    1.0/( 1 + np.exp(-1*M ))


class LogisticRegression(object):
    
    
    def __init__(self):
        pass
    
    def 