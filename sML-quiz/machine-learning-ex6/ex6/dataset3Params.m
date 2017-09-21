function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Matrix for store results of all test pairs.
results = zeros(64,3);
cases = 0;

% Set of values to try for optimal pair of values for C and Sigma.
para = [0.01 0.03 0.1 0.3 1 3 10 30];

for i = 1:size(para, 2)
    for j = 1:size(para, 2)
        cases = cases + 1;
        C_test = para(i);
        sigma_test = para(j);
        %
        model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
        %
        predictions = svmPredict(model, Xval);
        
        % Compute the prediction error.
        pred_err = mean(double(predictions ~= yval));
        
        % Accumulate each case's result to 'results' matrix.
        results(cases, :) = [C_test, sigma_test, pred_err];
    end
end

% Sorts the rows of matrix 'result' in ascending order based on the
% elements in the third column (prediction error). That means the pair of
% of values for C and Sigma which has lowest prediction error will be placed on
% top (first row).
results_sorted = sortrows(results, 3);

% Return the values of C and sigma.
C = results_sorted(1,1);
sigma = results_sorted(1,2);


% =========================================================================

end
