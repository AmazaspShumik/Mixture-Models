X = zeros(1000,3);
X(1:330,1) = 1;
X(330:700,2) = 1;
X(700:1000,3)  =1;
k=3;
maxIter = 20;
convThresh = 1e-2;
[P,C,I] = bernoulli_mixture(X,3,1000,1e-5);