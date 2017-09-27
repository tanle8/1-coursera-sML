function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


m = size(X,1)



for i = 1:m
    
    % Create an 1 by K zero matrix to store distance value of is and centroids
    distances = zeros(1,K);
    
    % With each example i in example set, calculate the distance between i and each
    % centroid of all K centroids we have.
    for j = 1:K
        distances(1,j) = sqrt(sum((X(i,:)-centroids(j,:)).^2));
    end
    
    % Contains the index of the centroid closest to example i.
    % Find the indices of the minimum values of `distance` and
    % returns them in output vector `C`. 
    [d, C] = min(distances);
   
    % Vector C contains the index of the centroids.
    idx(i,1) = C;
end





% =============================================================

end

