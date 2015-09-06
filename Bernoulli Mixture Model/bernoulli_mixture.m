function [C,P,L] = BernoulliMixture( X , k , maxIter, convThresh)
%% 
% Implements Bernoulli Mixture Model using EM for maximum likelihood
% optimization.
% Best effort was made to make computation numerically stable, prevent
% underflow and other issues.
%
% Parameters:
% -----------
%             X: matrix of size [ n_samples, n_features ]
%                 Matrix of observations (should consist of only zeros and ones)
%
%             k: integer
%                 Number of clusters (Number of values latent variable can 
%                 take)
%
% Returns:
% --------
%            P: matrix of size [n_samples, k]
%                Matrix of responsibilities, each column j shows probability
%                that observations belong to j-th cluster
%            C: vector of size [n_samples,1]
%                Cluster assignment for each cluster 
%            L: vector of size 
%       

%% Parameter initialisation 

% use sparse matrix , save memory
if ~issparse(X)
    X = sparse(X);
end

[n,m]       = size(X);
IX          = 1 - X;
logPrior    = log( ones(1,k)/k );     % vector with log of prior probabilities
logPs       = rand(k,m);              % vector with probabilities of success 
logPf       = 1 - logPs;              % vector with probability of failure
logPs       = log(logPs);             % log transformation of probability of failure
logPf       = log(logPf);             % log transformation of probability of succes
resps       = ones(n,k);              % vector of responsibilities
logLike     = NaN*ones(1,maxIter);

for i = 1:maxIter
    %% E-step
    
    for j = 1:k
        resps(:,j) = sum(bsxfun(@times,X,logPs(j,:)),2) + sum(bsxfun(@times,IX,logPf(j,:)),2) + logPrior(j);
    end
    
    %% calculate log-likelihood
    logJoint = resps;
    
    % since we do not do probability multiplication after this point 
    % we can go back from log scale to normal, however if value of log p
    % is small exp( log (p) ) = 0, to avoid that we calculate exp( log(p) + max) 

    resps       = exp(bsxfun(@plus,resps,max(resps,[],2)));
    resps       = bsxfun(@rdivide,resps,sum(resps,2));
    loglike     = sum(sum(resps.*logJoint - resps.*log(resps))) / n;
    logLike(i)  = loglike;
    if i > 1
        if loglike - logLike(i-1) < convThresh
            P           = resps;
            [m,mi]      = max(P,[],2);
            C           = mi;
            L           = logLike;
            return;
        end
    end
        
    
    %% M-step
    
    n_k        = sum(resps,1);
    logPrior   = n_k /n ; 
    for j = 1:k
        % this is inefficient for time complexity , but prevents
        % possibility of underflow
        logPs(j,:) = sum( bsxfun(@times, X, resps(:,j)), 1) / n_k(j);
        logPf(j,:) = sum( bsxfun(@times, IX, resps(:,j)), 1) / n_k(j);
    end
    logPs     = log(logPs);
    logPf     = log(logPf);
    logPrior  = log(logPrior);

end

warning('EM procedure did not converge!!!')
P           = resps;
[m,mi]      = max(P,[],2);
C           = mi;
logLike(i)  = loglike;
L           = logLike;

    
end






