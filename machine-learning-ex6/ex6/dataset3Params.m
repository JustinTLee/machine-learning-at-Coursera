function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_vector = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_vector = [0.01 0.03 0.1 0.3 1 3 10 30];
num_choices = length(C_vector);
pred_error = zeros(num_choices, num_choices);

C_index_min = 0;
sigma_index_min = 0;
current_lowest_error = 0;

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

for m = 1:num_choices
    
    C_current = C_vector(m);
    
    for n = 1:num_choices

        sigma_current = sigma_vector(n);
        model= svmTrain(X, y, C_current, @(x1, x2) gaussianKernel(x1, x2, sigma_current));
        pred = svmPredict(model, Xval);
        pred_error(m, n) = mean(double(pred ~= yval));

        if ((m == 1) && (n == 1)) || (current_lowest_error > pred_error(m, n))
            current_lowest_error = pred_error(m, n);
            C_index_min = m;
            sigma_index_min = n;
        else
        end
        
    end
    
end

C = C_vector(C_index_min);
sigma = sigma_vector(sigma_index_min);

% =========================================================================

end
