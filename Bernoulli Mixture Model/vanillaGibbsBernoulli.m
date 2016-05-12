function [ means, logProb ] = vanillaGibbsBernoulli(X,nComponents,nSamples,nBurnin,nThin)                                                               
% Bayesian Bernoulli Mixture Model fitted with vanilla
% Gibbs Sampler
% 
% Parameters
% ----------
% X: matrix of size (nDataSamples,nFeatures)
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
% 

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

prSample = dirchrnd(latendDist,1);


for i = 1:(nSamples*nThin+nBurnin)
    
    % sample p( z_i | X, Z_{-i}, mu, pr )
    latentSample = sparse(mnrnd(1,prSample,nDataSamples));
    Nk           = sum(latentSample,1);
    Xweighted    = X'*latentSample;
    IXweighted   = Nk - Xweighted;
    
    % sample p( pr | X, Z, mu_{1:k} )
    prSample = dirchrnd( latentDist + Nk,1 );
    
    % sample p( mu_k | X, Z, mu_{-k}, pr )
    muSample = betarnd(muAlpha + Xweighted,muBeta + IXweighted);
    
    if i > nBurnin && mod(i-nBurnin,nThin)==0
        % accept sample after burnin & thinning
        
    end
    
    
end
        
        
    
    
    
    


end

