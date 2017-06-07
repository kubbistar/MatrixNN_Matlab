function [cost,grad] = MatrixNeuralNetworkRegressionCost(theta, layerSize, lambda, sparsityParam, beta, data, Y, linear)

% visibleSize: the sizes of input units (probably 25x25) 
% hiddenSize: the sizes of hidden units (probably 15*15) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 25x25x10000 matrix containing the training data.  So, data(:,:,i) is the i-th training example. 
% Y: expected output, say 25x25x10000 matrix which is exactly the input for
% autoencoder. 

% linear indicates the last layer is linear.
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (U1, V1, B1, U2, V2, B2,...) matrix/vector format, so that this 
% follows the notation convention of the lecture notes.

if strcmp(linear, 'linear')
    linear = 1;
else
    linear = 0;
end
L = length(layerSize)-1;
U = cell(1,L);
V = cell(1,L);
B = cell(1,L);
X = cell(1,L+1);
N = cell(1, L);

count_start = 1;
for i=2:L+1
    count_end = count_start + layerSize{i}.I*layerSize{i-1}.I - 1;
    U{i-1} = reshape(theta(count_start:count_end), layerSize{i}.I, layerSize{i-1}.I);
%     grad.U{i-1} = zeros(size(U{i-1}));
    
    count_start = count_end + 1;
    count_end = count_start+layerSize{i}.J*layerSize{i-1}.J - 1;
    V{i-1} = reshape(theta(count_start:count_end), layerSize{i}.J, layerSize{i-1}.J);
%     grad.V{i-1} = zeros(size(V{i-1}));
    
    count_start = count_end + 1;
    count_end = count_start+layerSize{i}.I*layerSize{i}.J - 1;
    B{i-1} = reshape(theta(count_start:count_end), layerSize{i}.I,layerSize{i}.J);
    
    count_start = count_end + 1;
%     grad.B{i-1} = zeros(size(B{i-1}));
end;
% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;


%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% forwarding ...
% The layer now starting with 1 as the first layer with neurons
X{1} = data; % The input is X_1. 
for i=1:L
   N{i} =bsxfun(@plus, prod_mat_tensor_mat(U{i}, X{i}, V{i}),  B{i});  % Summarising
   if (i+1>L)&&(linear)
       X{i+1} = N{i};
   else
       X{i+1} = 1./(1+exp(-N{i})); % Activating in sigmoid function
   end
   cost = cost + 0.5*lambda*(norm(U{i},'fro')^2 + norm(V{i},'fro')^2);
   if i+1 <= L
       rho = mean(X{i+1},3); %average activation value for sparsity
       part_delta{i+1} = beta*(-sparsityParam./rho + (1-sparsityParam)./(1-rho));
       cost = cost + beta * sum(sum(sparsityParam*log (sparsityParam ./ rho) + (1-sparsityParam)*log((1-sparsityParam)./(1-rho)))); 
   end;
end;
%X{L+1} is now the output of the network. 
cost = cost + 0.5 * sum(sum(sum((X{L+1} - Y).^2))) / size(data,3);

% Back-propogation
grad = [];
if linear
    delta{L+1} = - (Y - X{L+1} );
else
    delta{L+1} = - (Y - X{L+1} ).* X{L+1} .* (1 - X{L+1});
end

for i=L:-1:1
    if i>1
        delta{i} = (bsxfun(@plus, prod_mat_tensor_mat(U{i}', delta{i+1}, V{i}'), part_delta{i})).* X{i} .* (1 - X{i});
    end;
    gradU = sum_prod_tensors(prod_tensor_mat(delta{i+1}, V{i}'), permute(X{i}, [2,1,3]) )/size(data,3) + lambda * U{i};
    gradV = sum_prod_tensors(prod_tensor_mat(permute(delta{i+1},[2,1,3]), U{i}'), X{i} )/size(data,3) + lambda * V{i};
    gradB = mean(delta{i+1},3);
    grad = [gradU(:);gradV(:);gradB(:);grad];
end;

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
