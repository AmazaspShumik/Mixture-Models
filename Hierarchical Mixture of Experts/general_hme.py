# -*- coding: utf-8 -*-

import nodes_hme as nh
import numpy as np
import matplotlib.pyplot as plt

class HME(object):
    
    def __init__(self,Y,X,k = 3):
        self.nodes = []
        n,m        = np.shape(X)
        self.Y     = Y
        self.X     = X
        self.nodes.append(nh.GaterNode(n,0,k,m))
        self.nodes.append(nh.GaterNode(n,1,k,m))
        self.nodes.append(nh.GaterNode(n,2,k,m))
        self.nodes.append(nh.GaterNode(n,3,k,m))
        self.nodes.append(nh.GaterNode(n,4,k,m))
        self.nodes.append(nh.GaterNode(n,5,k,m))
        self.nodes.append(nh.GaterNode(n,6,k,m))
        self.nodes.append(nh.GaterNode(n,7,k,m))
        self.nodes.append(nh.GaterNode(n,8,k,m))
        self.nodes.append(nh.GaterNode(n,9,k,m))
        self.nodes.append(nh.GaterNode(n,10,k,m))
        self.nodes.append(nh.GaterNode(n,11,k,m))
        self.nodes.append(nh.GaterNode(n,12,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,13,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,14,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,15,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,16,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,17,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,18,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,19,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,20,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,21,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,13,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,14,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,15,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,16,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,17,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,18,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,19,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,20,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,21,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,22,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,23,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,24,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,25,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,26,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,27,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,28,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,29,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,30,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,31,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,32,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,33,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,34,k,m))
        self.nodes.append(nh.ExpertNodeLinReg(n,35,k,m))
        
    def operate(self):
        ''' Test children parent functions'''
        for node in self.nodes[0:4]:
            print "position "+str(node.node_position)+" children positions "+','.join([str(e.node_position) for e in node.get_childrens(self.nodes)]) 
        
        for node in self.nodes[1:]:
            print "position "+str(node.node_position)+" parent position and order of birth " + str(node.get_parent_and_birth_order(self.nodes)[0].node_position) + " "+str(node.get_parent_and_birth_order(self.nodes)[1])
        
        
    def up_tree(self):
        ''' Tests up tree algorithm'''
        for i in range(len(self.nodes)):
            position = len(self.nodes) - i - 1
            #print position
            node = self.nodes[position]
            if node.node_type == "expert":
                node.up_tree_pass(self.X,self.Y)
            elif node.node_type == "gate":
                node.up_tree_pass(self.X, self.nodes)
                
                
    def down_tree(self):
        for node in self.nodes:
            if node.node_type == "expert":
                node.down_tree_pass(self.X,self.Y,self.nodes)
            elif node.node_type == "gate":
                node.down_tree_pass(self.X, self.nodes)
            
            
    def iterate(self):
        for i in range(80):
            self.up_tree()
            self.down_tree()
            
    def predict_mean(self,X):
        return self.nodes[0].propagate_mean_prediction(X,self.nodes)

    
if __name__=="__main__":
#    X      = np.zeros([100,2])
#    X[:,0] = np.linspace(0, 10, 100)
#    X[:,1] = np.ones(100)
#    Y = X[:,0]*4 + np.random.normal(0,1,100)
#    hme = HME(Y, X)
#    hme.operate()
#    hme.iterate()
#    # test coef
#    theta_exp = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,Y))
    X      = np.zeros([1000,2])
    X[:,0] = np.linspace(0, 10, 1000)
    X[:,1] = np.ones(1000)
    Y = np.sin(X[:,0])*4 + np.random.normal(0,1,1000)
    hme = HME(Y, X)
    hme.iterate()
    Y_hat = hme.predict_mean(X)
    plt.plot(Y,"b+")
    plt.plot(Y_hat,"r-")
    plt.show()
    #hme.up_tree()
    #hme.down_tree()
        
