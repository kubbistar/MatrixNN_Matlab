function C = innerprod_tensor_matrix(T, V)
% Both T and V are 3order tensors. The result is a matrix such that we find out the inner product between the third slice of them. 
% the (ij) element of C is the inner product of slice T(:,:,i) and V(:,:,j)

% Author:  Professor Junbin Gao
% Copyright all reserved, last modified 5 July 2015

  [n1, n2, n3] = size(T);
  C = reshape(T, n1*n2, n3); % convert the tensor into a matrix such that the columns correspond to the slice T(:,:,i)
  C = C' * reshape(V, n1*n2, size(V,3)); 
  %C = C';
end