function [C,P,L] = MultinomialMixtureModel(X,categories,k)
% Multinomial Mixture Model
%
% Parameters:
% -----------
%    X: matrix of size [n_samples, n_features]
%       Design matrix
%    
%    categories: number or vector of size [1,n_features]
%       If vector, then each element of it identifies possible
%       number of categories. If only single number is passed
%       then it is assumed that all elements of each data point 
%       has the same number of categories.
%
%    k: integer
%       Number of centroids
%
% Returns:
% --------
% 
%    C: vector of size [n_samples,1]   
%       Cluster assignments
%    
%    P: Matrix of size [n_samples,k]
%       Matrix of probabilities (probabilities of observation
%       belongs to particular centroid)





end

