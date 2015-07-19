# -*- coding: utf-8 -*-

import nodes_hme as nh
import numpy as np
import label_binariser as lb
from helpers import *



class HME(object):
    '''
    Implementation of Hierarchical Mixture of Experts, supports only balanced tree 
    of arbitrary depth and arbitrary branching factor.
    
    Parameters:
    -----------
    
    Y_train: numpy array of size 'n x 1'
       Dependent variables, training set
       
    X_train: numpy array of size 'n x m'
       Matrix of inputs, training set
       
    Y_test: numpy array of size 'n_test x 1'
       Dependent variables, test set
    
    X_test: numpy array of size 'n_test x m'
       Matrix of explanatory variables, test set
       
    expert_type: str
       Type of the expert to be used, either "logit" or "gaussian"
       
    gate_type: str
       Type of gating network used "softmax" or "wgda" (weighted gaussian discriminant
       analysis) 
       
    bias: bool
       If True adds bias term (columns of ones) to X matrix (It is expected that X does
       not contain bias term)
       
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
       
       
    References:
    -----------
    Jordan and Jacobs, Hierarchical Mixture of Experts and the EM algorithm, 1994
    Jordan and Xu, Convergence Results for the EM approach to mixture of experts architechtures, 1995
    '''
    
    def __init__(self,Y_train,X_train,Y_test,X_test, expert_type, gate_type = "softmax", 
                                                                  bias        = True,
                                                                  branching   = 2,
                                                                  levels      = 8, 
                                                                  max_iter    = 100,
                                                                  conv_thresh = 1e-20,
                                                                  verbose     = False):
        self.nodes        = []
        self.bias         = bias
        self.conv_thresh  = conv_thresh
        self.verbose      = verbose
        self.max_iter     = max_iter

        # add bias term if required
        if self.bias is True:
            n         = np.shape(X_train)[0]
            X_train   = np.concatenate([X_train,np.ones([n,1])], axis = 1)        

        # check that there is no errors in gate and expert type
        expert_types      = ["gaussian","softmax","wgda"]
        gater_types       = ["softmax","wgda"]
#        assert expert_type in expert_types, 'wrong model type for expert node'
#        assert gate_type   in gater_types,  'wrong model type for gater node'
        if expert_type not in expert_types:
            raise NodeModelNotImplementedError(expert_type, "expert")
        if gate_type not in gater_types:
            raise NodeModelNotImplementedError(gate_type, "gater")
        self.expert_type  = expert_type
        self.gate_type    = gate_type
        
        # transform target variables in case of logistic regression task
        if expert_type == "softmax" or expert_type == "wgda":
            self.classes   = len(set(Y_train))
            self.converter = lb.LabelBinariser(Y_train,self.classes)
            Y_train        = self.converter.convert_vec_to_binary_matrix(Y_raw = Y_train)
            Y_test         = self.converter.convert_vec_to_binary_matrix(Y_raw = Y_test)
            
        self.X             = X_train
        self.Y             = Y_train
        self.x             = X_test
        self.y             = Y_test
        self.n,self.m      = np.shape(self.X)
        
        # overall number of parameters in HME
        self.total_params  = 0
                
        # list of l2 norm of parameter change (in optimum change should be near 0)
        self.delta_param_norm    = []
        self.delta_log_like_lb   = []
        self.test_log_like       = []
        
        # create HME tree
        self._create_hme_topology(levels,branching)

        

    def _create_hme_topology(self,levels,k):
        ''' 
        Creates HME tree with given depth and branching parameter
        
        Parameters:
        -----------
        
        levels: int
           Number of levels in tree  
        
        k: int
           Branching parameter

        '''
        node_counter = 0
        for level in range(levels):
            for node_pos in range(k**level):
                
                # adding gating nodes (all but last levels of hme tree)
                if level < levels-1 :
                    
                    # softmax gating model 
                    if self.gate_type == "softmax":
                        self.nodes.append(nh.GaterNodeSoftmax(self.n,node_counter,k,
                                                                                  self.m,
                                                                                  classes = k))
                        self.total_params += (k-1)*self.m
                        
                    # weighted gaussian discriminant gating model
                    elif self.gate_type == "wgda":
                        self.nodes.append(nh.GaterNodeWGDA(self.n,node_counter,k,
                                                                               self.m,
                                                                               bias_term = self.bias,
                                                                               classes   = k ))
                        if self.bias is True:
                            self.total_params += k*(self.m-1)
                        else:
                            self.total_params += k*self.m
                                       
                #  adding expert nodes (last level of hme tree)
                elif level == levels-1:
                    
                    # linear regression expert
                    if self.expert_type   == "gaussian":
                        self.nodes.append(nh.ExpertNodeLinReg(self.n,node_counter,k,self.m))
                        self.total_params += self.m
                        
                    # multilogit regression ( softmax regression for classification)
                    elif self.expert_type == "softmax":
                        self.nodes.append(nh.ExpertNodeSoftmaxReg(self.n,node_counter,k,
                                                                                      self.m,
                                                                                      classes = self.classes))
                        self.total_params += (self.classes-1)*self.m
                        
                    # weighted gaussian discriminant analysis as model in expert node
                    elif self.expert_type == "wgda":
                        self.nodes.append(nh.ExpertNodeWGDA(self.n,node_counter,k,
                                                                                self.m,
                                                                                bias_term = self.bias,
                                                                                classes   = self.classes))
                        if self.bias is True:
                            self.total_params += self.classes*(self.m-1)
                        else:
                            self.total_params += self.classes*self.m
                node_counter+=1


    def _up_tree_pass(self):
        ''' 
        Performs up tree pass, calculates prior probabilities of latent variables
        '''
        for node in reversed(self.nodes):
            if node.node_type == "expert":
                node.up_tree_pass(self.X,self.Y)
            elif node.node_type == "gate":
                node.up_tree_pass(self.X, self.nodes)
            
                                
    def _down_tree_pass(self):
        ''' 
        Performs down tree pass, calculates posterior probabilities of 
        latent variables and maximises lower bound of likelihood by updating parameters
        '''
        delta_param_norm = 0
        delta_log_like   = 0
        for node in self.nodes:
            if node.node_type == "expert":
                node.down_tree_pass(self.X,self.Y,self.nodes)
            elif node.node_type == "gate":
                node.down_tree_pass(self.X,self.nodes)
            delta_param_norm += node.get_delta_param_norm()
            delta_log_like   += node.get_delta_log_like()
             
        # normalise change in parameters and lower bound of likelihood
        normalised_delta_params       = delta_param_norm  / self.total_params
        normalised_delta_like         = delta_log_like / self.n
        
        # save changes in likelihood  and parameters for last iteration 
        self.delta_param_norm.append(normalised_delta_params)
        self.delta_log_like_lb.append(normalised_delta_like)

            
            
    def fit(self):
        '''
        Performs iterations of EM algorithm until convergence (or limit of iterations)
        '''
        converged    = False
        for i in range(self.max_iter):
            self._up_tree_pass()
            self._down_tree_pass()
            if self.verbose is True:
                out = "iteration {0} completed , total change in parameters is {1}"
                print out.format(i,self.delta_param_norm[-1])
            
            # terminate algorithm if parameters changed by less than threshold
            param_change = self.delta_param_norm[-1]
            if abs(param_change) <= self.conv_thresh:
                    if self.verbose is True:
                       print "Algorithm converged"
                    converged = True
                    break
        if self.verbose is True and converged is False:
              print "Maximum number of iterations is reached"
            
            
    def predict(self,X, bias_term = False, predict_type = "predict_response", y_lo = None, y_hi = None):
        '''
        Returns weighted average of expert predictions
        
        Parameters:
        -----------
        
        X: numpy array of size 'unknown x m'
           Explanatory variables for test set
           
        bias_terms: bool
           If True, then columns of ones is appended to matrix X as last column
           
        predict_type: str
           Can be mean "predict_response", "predict_prob", "predict_cdf"
           "predict_response"   - works for all type of experts 
           "predict_prob"       - works for classification experts ('wgda','softmax')
           "prdict_cdf"         - works only for 'gaussian' expert
           
        
           
        Returns:
        --------
        prediction: numpy array of size 'unknown x 1'
        
        '''
        # include bias term if needed
        if self.bias is True:
            n = np.shape(X)[0]
            X = np.concatenate([X,np.ones([n,1])], axis = 1)
            
        # tree traversal for finidng predicted value
            
        # for classification cases use probability predictions (they will be 
        # transformed into response variable later)
        if self.expert_type in ["softmax" ,"wgda"] and predict_type == "predict_response":
            prediction = self.nodes[0].propagate_prediction(X,self.nodes,"predict_probs",y_lo,y_hi)
            
        # predict_probs is defined only for classification problems
        elif self.expert_type == "gaussian" and predict_type == "predict_probs":
            raise NotImplementedError(" 'predict_probs' is implemented only for classification experts")
        
        # predict_cdf is defined only for 'gaussian' expert
        elif self.expert_type != "gaussian" and predict_type == "predict_cdf":
            raise NotImplementedError(" 'predict_cdf' is implemented only for 'gaussian' expert")
            
        else:
            prediction = self.nodes[0].propagate_prediction(X,self.nodes,predict_type,y_lo,y_hi)
            
        # post processing (transform average probabilities to response variable)
        if self.expert_type in ["softmax" ,"wgda"] and predict_type == "means":
            return self.converter.convert_prob_matrix_to_vec(prediction)
        return prediction
              