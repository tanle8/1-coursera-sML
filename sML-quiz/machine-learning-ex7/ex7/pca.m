function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
% m - the number of examples, n - the number of features
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

% Compute the covariance matrix.
cov = (1/m)*(X'*X);

% Compute the principal components using SVD function with covariance
% matrix from previous step.
% U contains the principal components,
% S contains a diagonal matrix.
[U, S, V] = svd(cov);




% =========================================================================

end
