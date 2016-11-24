function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%y
%size(y)
% Add ones to the X data matrix
a2 = sigmoid([ones(m, 1) X] * Theta1');
size(a2);
a3 = sigmoid([ones(m,1) a2] * Theta2');
size(a3);

%
for i = 1:m
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
  % create a R^k vector (logical array)
  new_y = (1:num_labels == y(i));
  for k = 1:num_labels
    % if y = 1
    c1 = -new_y(k) * log(a3(i,k));
    % if y = 0
    c2 = -(1 - new_y(k)) * log(1 - a3(i,k));
    J = J + c1 + c2;
  end
end
J = J / m;

reg = 0;
for j = 1:size(Theta1,1)
  for k = 2:size(Theta1,2);
    reg = reg + (Theta1(j,k)^2);
  end
end
for j = 1:size(Theta2,1)
  for k = 2:size(Theta2,2);
    reg = reg + (Theta2(j,k)^2);
  end
end
reg = (lambda / (2 * m)) * reg;
J = J + reg;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

disp ("The size of Theta1:"), disp(size(Theta1))
disp ("The size of Theta2:"), disp(size(Theta2))

for t = 1:m
  a1 = X(t,:)(:);
  size(a1);  
  a1 = [1; a1];
  z2 = Theta1 * a1;
  %disp ("The size of z2:"), disp(size(z2))
  a2 = [1; sigmoid(z2)];
  %disp ("The size of a2:"), disp(size(a2))
  z3 = Theta2 * a2;
  %disp ("The size of z3:"), disp(size(z3))
  a3 = sigmoid(z3);
  %disp ("The size of a3:"), disp(size(a3))
  new_y = (1:num_labels == y(t))(:);
  size(new_y);
  d3 = a3 - new_y;
  %equals the product of δ3 and Θ2 (ignoring the Θ2 bias units)
  half_d2 = Theta2(:,2:1:end)' * d3;
  %disp ("The size of half_d2:"), disp(size(half_d2))
  sg = sigmoidGradient(z2);
  %disp ("The size of sg:"), disp(size(sg))
  d2 = half_d2 .* sg;
  %disp ("The size of d2:"), disp(size(d2))
  Theta2_grad = Theta2_grad + (d3 * a2');
  Theta1_grad = Theta1_grad + (d2 * a1');
end
Theta2_grad = (1/m) .* Theta2_grad;
Theta1_grad = (1/m) .* Theta1_grad;

Theta2_grad(:,2:1:end) = Theta2_grad(:,2:1:end) + (lambda / m) .* Theta2(:,2:1:end)
Theta1_grad(:,2:1:end) = Theta1_grad(:,2:1:end) + (lambda / m) .* Theta1(:,2:1:end)
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
