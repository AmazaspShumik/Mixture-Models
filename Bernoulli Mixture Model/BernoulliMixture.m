classdef BernoulliMixture < handle
    % Bernoulli Mixture Model 
    
    properties(SetAccess = private, GetAccess = public)
        maxIter;      % maximum number of iterations of EM
        convThresh;   % convergence threshold for EM
        nComps;       % number of components
        Ps;           % probabilities of success
        logLike;      % log likelihood
    end
    
    methods(Access = public)

        function obj = BernoulliMixture(nComps,maxIter,convThresh)
        % Constructor for Bernoulli Mixture Model
        %
        % Parameters:
        % -----------
        %   nComps: integer
        %      Number of clusters
        %   
        %   maxIter: integer (OPTIONAL, DEFAULT = 100)
        %      Maximum number of iterations for EM algorithm
        %
        %   convThresh: double (OPTIONAL, DEFAULT = 1e-3)
        %      Threshold for convergence   
            
            % if user did not provide values use defaults
            switch nargin
                case 1
                    maxIter        = 100;
                    convThresh     = 1e-3;
                case 2
                    convThresh     = 1e-3;
            end
            
            % assign instance variables
            obj.nComps     = nComps;       
            obj.maxIter    = maxIter;
            obj.convThresh = convThresh;
            obj.logLike    = -inf*ones(maxIter,1);
        end
        
        
        function fit(obj,X,Ps)
        % Fits Bernoulli Mixture Model using EM algorithm 
        %
        % Parameters:
        % -----------
        %     X: matrix of size [ n_samples, n_features ]
        %        Matrix of observations (should consist of only zeros and ones)
        %
        %     Ps: matrix of size [ n_clusters, n_features] (OPTIONAL)
        %        Matrix of succes probabilities
        
            % n_samples, n_features
            [n,d]    = size(X);
            
            % use sparse matrix , save memory
            if ~issparse(X)
               X  = sparse(X);
            end
            IX               = 1 - X;
            
            % if matrix of succes probabilities is not provided by user,
            % initialise it (and its log)
            if nargin == 2
                Ps   = rand(obj.nComps,d);
            end
            
            Pf               = 1 - Ps;
            logPs            = log(Ps);
            logPf            = log(Pf);
            logPrior         = ones(obj.nComps,1) / obj.nComps; 
            resps            = ones(n,obj.nComps);
            
            for iter = 1:obj.maxIter
            %% E-step: calculate responsibilities & log-likelihood
            
            % calculate log responsibilities
            for j = 1:obj.nComps
                resps(:,j)    = sum(bsxfun(@times,X,logPs(j,:)),2) + ...
                                sum(bsxfun(@times,IX,logPf(j,:)),2) + ...
                                logPrior(j);
            end
            
            % reuse unnormalised log responsibilities for log-likelihood
            % calculation & calculate normalised responsibilities
            logJoint          = resps;
            lse               = BernoulliMixture.logsumexp(resps);
            logResps          = bsxfun(@minus,resps,lse);
            resps             = exp(logResps);
            obj.logLike(iter) = sum(sum(resps .* ( logJoint - logResps)));
            
            % check convergence
            if iter >= 100
                if obj.logLike(iter) - obj.logLike(iter - 1) < obj.convThresh
                    msg = 'Algorithm converged'
                end
            end
            
            
            %% M-step: calculate matrix of success probabilities & mixing 
            %  coefficients
            Nk                = sum(resps);
            prior             = Nk / n;
            logPrior          = log(prior);
            for k = 1 : obj.nComps
                Ps(k,:) = sum( bsxfun(@times,X,resps(:,k)) ) / Nk(k);
                Pf(k,:) = sum( bsxfun(@times,IX,resps(:,k)) ) / Nk(k);
            end
            ps                = Ps
            pf                = 1 - Ps
            logPs             = log(Ps)
            logPf             = log(Pf)
            
            
            
            end
            resps
            
        end
    end

    
    methods(Access = public , Static = true)
        
        function lse = logsumexp(X)
        % Calculates log(sum(exp(X),2)) while avoiding numerical underflow.
        % 
        % Parameters:
        % -----------
        % X: maxtrix of size [n_samples, n_clusters]
        %     Data Matrix
        %
        % Returns:
        % --------
        % lse: vector of size [n_samples, 1]
        %     Value of log(sum(exp(X),2))
        
            m   = max(X,[],2);
            lse = log(sum(exp(bsxfun(@minus,X,m)),2));
            lse = lse + m; 
        end
        
        
    end
    
end

