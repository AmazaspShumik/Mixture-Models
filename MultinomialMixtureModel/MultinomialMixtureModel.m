function [C,P,L] = MultinomialMixtureModel(X,cats,k, max_iter, conv_thresh)
% Multinomial Mixture Model fitted with Expectation Maximization
% This implementation allows each element in single data point 
% have different number of categories.
%
% Parameters:
% -----------
%    X: matrix of size [n_samples, n_features]
%       Design matrix
%    
%    cats: number or vector of size [1,n_features]
%       If vector, then each element of it identifies possible
%       number of categories. If only single number is passed
%       then it is assumed that all elements of each data point 
%       has the same number of categories.
%
%    k: integer
%       Number of centroids
%    
%    max_iter: integer
%       Maximum number of iterations of EM algorithm
%
%    conv_thresh: float
%       Convergence threshold
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

% sample size & dimensionality
[n,m] = size(X);

% cell array of centroids
centroids  = cell(k,1);

% initialise each centroid
[cn,cm] = size(cats);

% check that categories vector is correct 
message = 'Category vector is not properly defined';
assert(cn==1 && cm==1 || cn==m && cm==1 || cm==m && cn==1, message);

% initialise each centroid randomly
for i = 1:k
    new_center = cell(m,1);
    for j = 1:m
        val           = rand(cats(j));
        new_center{j} = val/sum(val);
    end
    centroids{i} = new_center;
end

% initialise prior belief
prior    = rand(k);
p        = prior/sum(prior);
% matrix of responsibilities
resps    = zeros(n,k);

% Iterations of EM algorithm
for iter = 1:max_iter
    
    % E-step, finds responsibilities
    
    
    
    
    % M-step 
    






end

