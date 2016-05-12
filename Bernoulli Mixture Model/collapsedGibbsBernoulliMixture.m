function [ ] = collapsedGibbsBernoulliMixture(X,nSamples,nBurnin,nLag)                                                               
% Bayesian Bernoulli Mixture Model fitted with collapsed
% Gibbs Sampler
% 
% Parameters
% ----------
% X: matrix of size (n_samples,n_features)
%    Data Matrix
%
% nComponents: integer
%    Number of components in mixture model
%
% nSamples: integer, optional (DEFAULT = 10000)
%    Number of samples
% 
% nBurnin: float, optional (DEFAULT = 0.25)
%    Proportion of samples that are discarded
% 
% nLag: int, optional (DEFAULT = 10)
%    Lag between samples ()
% 
% priorParams: struct, optional
%    Parameters of prior distribution
%
%
% Returns
% -------

% handle optional parameters
if ~exist('nSamples','var')
    nSamples = 10000;
end

if ~exist('nBurnin','var')
    nBurnin  = 2500;
end

if ~exist('nThin','var')
    nThin    = 10;
end
    
% number of datapoints & dimensionality
[nDataSamples,nFeatures] = size(X);

% parameters of prior distribution
if ~exist('priorParams','var')
    latentDist = 1 + rand(nComponents);
    muAlpha  = 1 + rand(nComponents,nFeatures);
    muBeta   = 1 + rand(nComponents,nFeatures);
else
    latentDist = priorParams.latentDist;
    muAlpha  = priorParams.muAlpha;
    muBeta   = priorParams.muBeta;
end

for i = 1:nSamples
    
    
    
    
end




end

