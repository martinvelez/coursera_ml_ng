function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

if(typeinfo(z) == 'matrix')
  dims = size(z);
  for r = 1:rows(z)
    for c = 1:columns(z)
      g(r,c) = 1 / (1 + e^(-z(r,c)));
    end
  end
elseif(typeinfo(z) == 'scalar')
  g = 1 / (1 + e^(-z));
end


% =============================================================

end
