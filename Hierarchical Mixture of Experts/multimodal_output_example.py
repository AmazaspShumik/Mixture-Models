# -*- coding: utf-8 -*-

import numpy as np
import general_hme as hm 
import matplotlib.pyplot as plt

def prob_grid(hme,X,y_lo,y_hi,n_steps, posterior_type = "pdf"):
    '''
    Calculates probability of observing values of y in grid given parameters of
    hme model and matrix of dependent variables
    '''
    n,m    = np.shape(X)
    assert m==1, "can plot only 2-d plots"
    x_grid = np.outer(X[:,0], np.ones(n_steps-1))
    y_grid = np.zeros([n,n_steps-1])
    P_grid = np.zeros([n,n_steps-1])
    step   = (y_hi-y_lo)/n_steps
    y_hi_i = y_lo + step
    for i in range(1,n_steps-1):
        if posterior_type == "pdf":
            P_grid[:,i]  = hme.predict(X,predict_type = "predict_cdf",y_lo = y_lo ,y_hi = y_hi_i)
        elif posterior_type == "cdf":
            P_grid[:,i]  = hme.predict(X,predict_type = "predict_cdf",y_lo = None ,y_hi = y_hi_i)
        y_grid[:,i]  = y_lo + (y_hi_i - y_lo)/2
        y_hi_i      += step
        y_lo        += step
    return [x_grid,y_grid,P_grid]
    
    
if __name__=="__main__":
    
    ################################### EXAMPLE 1 ################################################
    
    # Multimodal example 1
    
    # create data set scaled sinusoid line and its reflection relative to x-axis
    X = np.zeros([300,1])
    Y = np.zeros([300,1])
    thr0 = 150; thr1 = 300
    X[0:thr0,0]    = np.linspace(0,6,thr0)
    X[thr0:thr1,0] = np.linspace(0,6,thr0)
    Y[0:thr0,0] = 6*np.sin(X[0:thr0,0]) + np.random.normal(0,1,thr0)
    Y[thr0:thr1,0] = -6*np.sin(X[thr0:thr1,0]) + np.random.normal(0,1,thr0)
    
    # train hme with weighted gaussian discriminant as gate model 
    # and linear regression as expert model. HME tree depth is 8 and branching 
    # factor is 2
    hme = hm.HME(Y[:,0], X,Y[:,0],X,"gaussian",bias = True, gate_type = "wgda",
                                                                     levels    = 4,
                                                                     branching = 2)   
    hme.fit()
    
    # calculate probability of observing points in squares formed by grid 
    y_lo = -10*np.ones(thr0)
    y_hi = 10*np.ones(thr0)
    x1,x2,p = prob_grid(hme,X[0:thr0,:],y_lo,y_hi,n_steps = 200,posterior_type = "pdf")
    n,m = np.shape(x1)
    
    # plot raw probabilities, pdf
    plt.pcolor(x1,x2,p,cmap = "coolwarm")
    plt.title("Probabilities, pdf")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    
    # In some cases it can be useful to estimate probability of dependent value being 
    # lower than particular threshold. For instance in financial industry it can be usefull 
    # to estimate probability of some catastrophic drop in value of portfolio.
    y_lo_two = -8*np.ones(thr0)
    y_hi_two = 10*np.ones(thr0)
    X1,X2,P = prob_grid(hme,X[0:thr0,:],y_lo_two,y_hi_two,n_steps = 200, posterior_type = "cdf")
    
    # plot raw probabilities, cdf
    plt.pcolor(X1,X2,P,cmap = "coolwarm")
    plt.title("Probabilities, cdf")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    # plot of original data along with pdf
    plt.pcolor(x1,x2,p,cmap = "coolwarm")
    plt.plot(X[:,0],Y,"c+")
    plt.title("Probabilities & Original Data, pdf")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    
    
    ################################# EXAMPLE 2 ###################################################
    
    
    # Multimodal Example 2
    X          = np.zeros([300,1])
    Y          = np.zeros([300,1])
    X[0:200,0]   = np.linspace(-3,3,200)
    X[200:300,0] = np.linspace(-3,3,100)
    Y[0:200,0]   = -1*X[0:200,0]**2 + 5 + np.random.normal(0,1,200)
    Y[200:300,0] = X[200:300,0]**2 - 5 + np.random.normal(0,1,100)
    
    hme = hm.HME(Y[:,0], X,Y[:,0],X,"gaussian",bias = True, gate_type = "wgda",
                                                            levels    = 2,
                                                            branching = 10)
    hme.fit()
                                                            
    # calculate probability of observing points in squares formed by grid 
    y_lo = -7*np.ones(200)
    y_hi = 7*np.ones(200)
    x1,x2,p = prob_grid(hme,X[0:200,:],y_lo,y_hi,n_steps = 200,posterior_type = "pdf")
    n,m = np.shape(x1)
    
    # plot raw probabilities, pdf
    plt.pcolor(x1,x2,p,cmap = "coolwarm")
    plt.title("Probabilities, pdf")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    
    # CDF
    y_lo_two = -8*np.ones(200)
    y_hi_two = 10*np.ones(200)
    X1,X2,P = prob_grid(hme,X[0:200,:],y_lo_two,y_hi_two,n_steps = 200, posterior_type = "cdf")
    
    # plot raw probabilities, cdf
    plt.pcolor(X1,X2,P,cmap = "coolwarm")
    plt.title("Probabilities, cdf")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    # plot of original data along with pdf
    plt.pcolor(x1,x2,p,cmap = "coolwarm")
    plt.plot(X[:,0],Y,"c+")
    plt.title("Probabilities & Original Data, pdf")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    
    #############################  EXAMPLE 3 #####################################################

    X      = np.ones([1000,1])
    X[0:500,0]     = np.linspace(0, 10, 500)
    X[500:1000,0]  = np.linspace(0, 10, 500)
    Y = np.zeros(1000)
    Y[0:500]     = X[0:500,0]*4 + np.random.normal(0,1,500) + 100
    Y[500:1000]  = X[500:1000,0]*(-4) + np.random.normal(0,1,500) + 115
    hme = hm.HME(Y, X,Y[0:10],X[0:10,:],"gaussian",bias = True, gate_type = "wgda",verbose = False,
                                                                                      levels    = 6,
                                                                                      branching = 2)
    hme.fit()
    y_lo = 40*np.ones(500)
    y_hi = 160*np.ones(500)
    x1,x2,p = prob_grid(hme,X[0:500,:],y_lo,y_hi,n_steps = 400,posterior_type = "pdf")
    

    plt.pcolor(x1,x2,p,cmap = "coolwarm")
    #plt.plot(X[:,0],Y,"c+")
    plt.title("Probabilities & Original Data, pdf")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    
    
    
    
    
    
    

    
    X      = np.ones([1500,1])
    X[:,0] = np.linspace(0, 10, 1500)
    Y = np.zeros(1500)
    Y[0:500]  = X[0:500,0]*4 + np.random.normal(0,1,500) + 100*np.ones(500)
    Y[500:1000] = X[500:1000,0]*(-4) + np.random.normal(0,1,500) + 200*np.ones(500)
    Y[1000:1500] = X[1000:1500,0]*8 + np.random.normal(0,1,500) + 160*np.ones(500)
    hme = hm.HME(Y, X,Y[0:10],X[0:10,:],"gaussian",bias = True, gate_type = "softmax",verbose = True, levels = 3, branching = 4)
    hme.fit()
    Y_hat = hme.predict(X)
    plt.plot(Y,"b+")
    plt.plot(Y_hat,"r-")
    plt.show()

        
    X      = np.zeros([3000,1])
    Y      = np.zeros([3000,1])
    X[:,0] = np.linspace(0, 10, 3000)
    Y[:,0] = np.sin(X[:,0])*4 + np.random.normal(0,1,3000)
    hme = hm.HME(Y[:,0], X,Y[0:10],X[0:10,:],"gaussian",bias = True, gate_type = "softmax",verbose = False,
                 levels = 2, branching  = 4)
    hme.fit()
    y_lo = -6*np.ones(3000)
    y_hi = 6*np.ones(3000)
    x1,x2,p = prob_grid(hme,X,y_lo,y_hi,n_steps = 400,posterior_type = "pdf")
    
    Y_hat = hme.predict(X)
    plt.plot(Y,"b+")
    plt.plot(Y_hat,"r-")
    plt.show()
    
    plt.pcolor(x1,x2,p,cmap = "coolwarm")
    plt.plot(X[:,0],Y,"c+")
    plt.title("Probabilities & Original Data, pdf")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

#    Another Regression example

#    X = np.zeros([3000,1])
#    X[:,0] = np.linspace(0,12.3,3000)
#    Y = (X[:,0]-1)*X[:,0]*(X[:,0]-2)*(X[:,0]-8)*(X[:,0]-9.8)*(X[:,0]-12) + np.random.normal(0,150,3000)
#    hme = HME(Y, X,Y[0:10],X[0:10,:],"gaussian",bias = True, gate_type = "softmax",verbose = True)   
#    hme.fit()
#    Y_hat = hme.predict(X, predict_type = "")
#    plt.plot(Y,"b+")
#    plt.plot(Y_hat,"r-")
#    plt.show()


#    Classification example

#    X = np.ones([4000,2])
#    X[:,0] = np.random.random(4000)
#    X[:,1] = np.random.random(4000)
#    Y = np.array(["y" for i in range(4000)])
#    Y[(X[:,0]-0.5)**2+(X[:,1]-0.5)**2 < 0.1] = "n"
#    hme = HME(Y, X,Y[0:10],X[0:10,:],"wgda",bias = True, gate_type = "wgda",verbose = True)
#    hme.fit()
#    Y_hat = hme.predict_mean(X)
#    plt.plot(X[Y_hat=="n",0],X[Y_hat=="n",1],"r+")
#    plt.plot(X[Y_hat=="y",0],X[Y_hat=="y",1],"b+")
    
    
    
#    X = np.ones([4000,2])
#    X[:,0] = np.random.random(4000)
#    X[:,1] = np.random.random(4000)
#    Y = np.array(["y" for i in range(4000)])
#    Y[(X[:,0]-0.2)**2+(X[:,1]-0.2)**2 < 0.04] = "n"
#    Y[(X[:,0]-0.8)**2+(X[:,1]-0.8)**2 < 0.04] = "n"
#    Y[(X[:,0]-0.5)**2+(X[:,1]-0.5)**2 < 0.04] = "n"
#    hme = HME(Y, X,Y[0:10],X[0:10,:],"softmax",bias = True, gate_type = "softmax",verbose = True)
#    hme.fit()
#    Y_hat = hme.predict_mean(X)
#    plt.plot(X[Y_hat=="n",0],X[Y_hat=="n",1],"r+")
#    plt.plot(X[Y_hat=="y",0],X[Y_hat=="y",1],"b+")
#    plt.plot(X[Y_hat != Y,0],X[Y_hat!=Y,1],"g+")
    
#   Another Classification example
    
#    X      = np.ones([30000,3]) 
#    X[:,0] = np.random.normal(0,1,30000)
#    X[:,1] = np.random.normal(0,1,30000)
#    X[10000:20000,0:2] = X[10000:20000,0:2]+10
#    X[20000:30000,0:2] = X[20000:30000,0:2]+15
#    Y = np.ones(30000)
#    Y[10000:20000] = 0
#    Y[20000:30000] = 2
#    
#    start_time = time.time()
#    hme = HME(Y, X[:,:-1],Y[0:10],X[0:10,:],"wgda",bias = True, gate_type = "softmax",verbose = True)
#    hme.fit()
#    print "OOps"
#    print np.shape(X)
#    Y_hat = hme.predict_mean(X[:,:-1])
    
    
    
    

        
       
        
