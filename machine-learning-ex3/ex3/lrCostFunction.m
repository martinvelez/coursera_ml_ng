function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%

%for i = 1:m
%  z = X(i,:) * theta;
%  J_i = -y(i) * log(sigmoid(z)) - (1 - y(i)) * log(1 - sigmoid(z));
%  J = J + J_i; 
%end
%J = J / m;
%reg = (lambda / (2 * m)) * sum(theta(2:1:end).^2);
%reg2 = reg
%J = J + reg

%J1 = 0
h = sigmoid(X * theta);
J = sum(-y' * log(h) - (1 .- y)' * log(1 .- h));
J = J / m;
reg = (lambda / (2 * m)) * sum(theta(2:1:end).^2);
J = J + reg;

% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%



% =============================================================

for j = 1:size(theta)
  for i = 1:m
    z = X(i,:) * theta;
    grad(j) = grad(j) + (sigmoid(z) - y(i)) * X(i,j);
  end
  grad(j) = grad(j) / m;
  if(j >= 2)
    grad(j) = grad(j) + (lambda / m) * theta(j);      
  end
end

grad = grad(:);


end
