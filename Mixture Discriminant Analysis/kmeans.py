# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 21:59:40 2015

@author: amazaspshaumyan
"""

import numpy as np
import random

class Kmeans(object):
    '''
    K-means algorihm for clustering.

    Parameters:
    -----------
    
    clusters           - (int)   number of expected clusters
    dim                - (int)   dimensionality of input
    epsilon            - (float) convergence threshold for k-means
    iteration_limit    - (int)   maximum number of iteration, where each 
                                 iteration consists of e_step and m_step
    data               - (list)  list of lists, where each inner list is 
                                 single data point
    
    '''
    
    def __init__(self, clusters, dim, epsilon, iteration_limit, data):
        self.k = clusters
        self.m = dim
        self.r = [0]*len(data) # vector of cluster assignments
        self.convergence_epsilon = epsilon
        self.iteration_limit = iteration_limit
        
        
    def loss(self):
        ''' 
        Calculates loss function of K-means
        J =  sum_n[ sum_k [r_n_k*||x_n-mu_k||^2]]]
        '''
        r = self.r
        mu = self.clusters
        J = sum([np.dot((np.array(x)-mu[r[i]]).T,np.array(x)-mu[r[i]]) for i,x in enumerate(self.data)])
        return J
    
    def initialise(self):
        ''' randomly choses points from list'''
        self.clusters = random.sample(self.data,self.k)     
        
    def e_step(self):
        ''' E-step in K means algorithm, finds assignment of points to centroids'''
        for n,data_point in enumerate(self.data):
            min_cl = 0
            min_sq_dist = -1
            for i,cluster in enumerate(self.clusters):
                dist_sq = sum([ (data_point[i]-cluster[i])**2 for i in range(self.m)])
                if min_sq_dist==-1:
                    min_sq_dist = dist_sq
                else:
                    if dist_sq < min_sq_dist:
                        min_sq_dist = dist_sq
                        min_cl = i
            self.r[n] = min_cl

            
    def m_step(self):
        ''' M-step in K-means algorithm, finds centroids that minimise loss function'''
        self.clusters = [[0]*self.m for i in range(self.k)] # update clusters
        cluster_counts = [0]*self.k
        for i,x in enumerate(self.data):
            cluster_counts[self.r[i]]+=1
            self.clusters[self.r[i]] = [self.clusters[self.r[i]][j]+x[j] for j in range(self.m)]
        mean_vector = lambda x,n: [float(el)/n for el in x]
        self.clusters = [mean_vector(self.clusters[i],cluster_counts[i]) for i in range(self.k)] 
            
    
    def run_k_means(self):
        ''' 
        Runs single pass of k-means algorithm
        '''
        self.initialise() # initialise clusters
        next_loss = self.loss() # calculate loss function for initial clusters
        prev_loss = next_loss +2*self.convergence_epsilon
        iteration = 0
        losses = []
        while prev_loss - next_loss > self.convergence_epsilon and iteration < self.iteration_limit:
            self.e_step()
            self.m_step()
            prev_loss = next_loss
            losses.append(prev_loss)
            next_loss = self.loss()
            iteration+=1
        
            
    def run(self, reruns = 10):
        ''' 
        Runs k-means several times and choosed and chooses parameters (mean vectors,
        point cluster allocation) from the k-means run with smallest value of 
        loss function.
        
        (Since loss function is not convex,it is not guaranteed that parameters 
        obtained from single k-means algorithm pass will give global minimum
        of k-means loss function)
        '''
        clusters = [[0]*self.m for i in range(self.k)]
        loss_before = -1
        r = self.r
        for i in range(reruns):
            self.run_k_means()
            loss_new = self.loss()
            if loss_before==-1:
                loss_before = loss_new
                clusters = [el[:] for el in self.clusters]
                r = self.r[:]
            else:
                if loss_new < loss_before:
                    loss_before = loss_new
                    clusters = [el[:] for el in self.clusters]
                    r = self.r[:]
                    
        self.final_r = r
        self.final_clusters = clusters
        
        
    def mda_params(self):
        ''' 
        Calculates initial parameters for GMM based on cluster allocation of
        points in best K-means
        '''
        total=0
        mixing = [0]*self.k
        covars = [np.zeros([self.m,self.m], dtype = np.float64) for i in range(self.k)]
        mu = [np.zeros(self.m, dtype = np.float64) for i in range(self.k)]
        for i,dp in enumerate(self.data):
            k = self.final_r[i] # cluster
            x = np.array(dp, dtype = np.float64)
            mixing[k]+=1
            total+=1
            mu[k]+=x
            covars[k]+=np.outer(x,x)
        mu = [mu[j]/p for j,p in enumerate(mixing)]
        covars = [1.0/mixing[j]*(covars[j] - mixing[j]*np.outer(mu[j],mu[j])) for j in range(self.k)]
        mixing = [float(p)/total for p in mixing]
        
        matrix_to_list = lambda x: [list(e) for e in x]
        mixing = mixing
        mu = matrix_to_list(mu)
        covariance = [matrix_to_list(e) for e in covars]
        return {"mixing":mixing,"mu":mu,"covariance":covariance}