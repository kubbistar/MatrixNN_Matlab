function C = prod_tensor_vector(T, V)
% this function calculate product of a 3-order tensor T and a matrix V such 
% the i-th slice matrice C(:,:,i) of tensor C is the x3 product of the tensor T with the i-th column of V(:,i)   

% Author:  Professor Junbin Gao
% Copyright all reserved, last modified 5 July 2015

  [n1, n2, n3] = size(T);
  C = reshape(T, n1*n2, n3); % convert the tensor into a matrix such that the columns correspond to the slice T(:,:,i)
  C = C * V;   % Now tbe i-th column is the linear combination of columns of T with weights from G(:,i)
  C = reshape(C, n1, n2, size(V,2)); 
end