# -*- coding: utf-8 -*-

import nodes_hme as nh
import numpy as np
import matplotlib.pyplot as plt
import label_binariser as lb

class HME(object):
    '''
    Implementation of Hierarchical Mixture of Experts, supports only balanced tree 
    of arbitrary depth and arbitrary branching factor.
    
    Parameters:
    -----------
    
    Y: numpy array of size 'n x 1'
       Vector of dependent variables
       
    X: numpy array of size 'n x m'
       Matrix of inputs, training set
       
    expert_type: str
       Type of the expert to be used, either "logit" or "gaussian"
       
    k: int
       Branching parameter
       
    levels: int
       Number of levels in tree
       
    max_iter: int
       Maximum number of iterations of EM algorithm
    
    conv_thresh: float
       Convergence threshold, if change in lower bound of likelihood is smaller
       than threshold then EM algorithm terminates
       
    verbose: str
       If True prints likelihood and iteration information
       
    '''
    
    def __init__(self,Y,X,expert_type,branching = 3, levels = 4, max_iter    = 60,
                                                                 conv_thresh = 0.005,
                                                                 verbose     = False):
        self.nodes        = []
        n,m               = np.shape(X)
        self.Y            = Y
        self.X            = X
        assert expert_type in ["gaussian","logit"], "Parameter expert type is wrong"
        self.expert_type  = expert_type
        if expert_type == "logit":
            self.converter = lb.LabelBinariser(Y,2)
            self.Y         = self.converter.logistic_reg_direct_mapping()
        self.max_iter     = max_iter
        self._create_hme_topology(levels,n,m,branching)
        # lower bound of log likelihood, saves values of lower bounds for each
        # iteration
        self.log_like_low = []
        self.conv_thresh  = conv_thresh
        self.verbose      = verbose
        
        
    def _create_hme_topology(self,levels,n,m,k):
        ''' 
        Creates HME tree with given depth and branching parameter
        
        Parameters:
        -----------
        
        levels: int
           Number of levels in tree  
           
        n: int 
           Number of observations
           
        m: int 
           Dimensionality of data
        
        k: int
           Branching parameter

        '''
        node_counter = 0
        for level in range(levels):
            for node_pos in range(k**level):
                if level < levels-1 :
                    self.nodes.append(nh.GaterNode(n,node_counter,k,m))
                elif level == levels-1:
                    if self.expert_type   == "gaussian":
                        self.nodes.append(nh.ExpertNodeLinReg(n,node_counter,k,m))
                    elif self.expert_type == "logit":
                        self.nodes.append(nh.ExpertNodeLogisticReg(n,node_counter,k,m))
                node_counter+=1
                
                
    def _up_tree_pass(self):
        ''' 
        Performs up tree pass, calculates prior probabilities of latent variables
        and lower bound of log-likelihood
        '''
        likelihood = 0
        for i in range(len(self.nodes)):
            position = len(self.nodes) - i - 1
            node = self.nodes[position]
            if node.node_type == "expert":
                node.up_tree_pass(self.X,self.Y)
            elif node.node_type == "gate":
                node.up_tree_pass(self.X, self.nodes)
            likelihood += node.log_likelihood_lb
        self.log_like_low.append(likelihood)
                
                
    def _down_tree_pass(self):
        ''' 
        Performs down tree pass, calculates posterior probabilities of 
        latent variables and maximises lower bound of likelihood by updating parameters
        '''
        for node in self.nodes:
            if node.node_type == "expert":
                node.down_tree_pass(self.X,self.Y,self.nodes)
            elif node.node_type == "gate":
                node.down_tree_pass(self.X, self.nodes)
            
            
    def fit(self):
        '''
        Performs iterations of EM algorithm until convergence (or limit of iterations)
        '''
        for i in range(self.max_iter):
            self._up_tree_pass()
            if self.verbose is True:
                out = "iteration {0} completed, lower bound of log-likelihood is {1} "
                print out.format(i,self.log_like_low[-1])
            self._down_tree_pass()
            if len(self.log_like_low) >= 2:
                last = self.log_like_low[-1]
                prev = self.log_like_low[-2]
                if (last - prev)/abs(prev) < self.conv_thresh:
                    if self.verbose is True:
                       print "Algorithm converged"
                    break
            
            
    def predict_mean(self,X):
        prediction = self.nodes[0].propagate_mean_prediction(X,self.nodes)
        if self.expert_type == "logit":
            return self.converter.logistic_reg_inverse_mapping(prediction)
        return prediction
    
    
if __name__=="__main__":
#    X      = np.zeros([100,2])
#    X[:,0] = np.linspace(0, 10, 100)
#    X[:,1] = np.ones(100)
#    Y = X[:,0]*4 + np.random.normal(0,1,100)
#    hme = HME(Y, X)
#    hme.operate()
#    hme.iterate()
#    test coef
#    theta_exp = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,Y))

#    Regression example

#    X      = np.zeros([1000,2])
#    X[:,0] = np.linspace(0, 10, 1000)
#    X[:,1] = np.ones(1000)
#    Y = np.sin(X[:,0])*4 + np.random.normal(0,1,1000)
#    hme = HME(Y, X,"gaussian", verbose = True)
#    #hme.operate()
#    hme.fit()
#    Y_hat = hme.predict_mean(X)
#    plt.plot(Y,"b+")
#    plt.plot(Y_hat,"r-")
#    plt.show()

#    Classification example

    X = np.ones([300,3])
    X[:,1] = np.random.random(300)
    X[:,2] = np.random.random(300)
    Y = np.array(["y" for i in range(300)])
    Y[(X[:,1]-0.5)**2+(X[:,2]-0.5)**2 < 0.1] = "n"
    hme = HME(Y, X,"logit", verbose = True)
    hme.fit()
    Y_hat = hme.predict_mean(X)
    plt.plot(X[Y_hat=="n",1],X[Y_hat=="n",2],"r+")
    plt.plot(X[Y_hat=="y",1],X[Y_hat=="y",2],"b+")
    #hme.down_tree()
        
