function [cost,grad] = Tensor4DNeuralNetworkRegressionCost(theta, layerSize, lambda, sparsityParam, beta, data, Y, linear)

% visibleSize: the sizes of input units (probably 28x28) 
% hiddenSize: the sizes of hidden units (probably 15*15) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 28x28x10000 matrix containing the training data.  So, data(:,:,i) is the i-th training example. 
% Y: expected output, say 28x28x10000 matrix which is exactly the input for
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
W = cell(1,L);
P = cell(1,L);
B = cell(1,L);
X = cell(1,L+1);
N = cell(1, L);
part_delta = cell(1, L+1);

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
    count_end = count_start+layerSize{i}.K*layerSize{i-1}.K - 1;
    W{i-1} = reshape(theta(count_start:count_end), layerSize{i}.K, layerSize{i-1}.K);
%     grad.V{i-1} = zeros(size(W{i-1}));
    count_start = count_end + 1;
    count_end = count_start+layerSize{i}.M*layerSize{i-1}.M - 1;
    P{i-1} = reshape(theta(count_start:count_end), layerSize{i}.M, layerSize{i-1}.M);
%     grad.V{i-1} = zeros(size(P{i-1}));
    
    count_start = count_end + 1;
    count_end = count_start+layerSize{i}.I*layerSize{i}.J*layerSize{i}.K - 1;
    B{i-1} = reshape(theta(count_start:count_end), layerSize{i}.I,layerSize{i}.J,layerSize{i}.K,layerSize{i}.M);
    
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
dimsX = size(data);
numData = dimsX(end);
X{1} = data; % The input is X_1. 
scale = size(Y,1)*size(Y,2)*size(Y,3)*size(Y,4)*size(Y,5);
for i=1:L
    UU = {U{i}, V{i}, W{i},P{i}};%%
    N{i} = double(ttm(tensor(X{i}), UU, (1:length(UU))) + ttm(tensor(B{i}, [size(B{i}) 1]), ones(1,numData)', length(dimsX)));  
    %N{i} =bsxfun(@plus, prod_mat_tensor_mat(U{i}, X{i}, V{i}),  B{i});  % Summarising
    if (i+1>L)&&(linear)
       X{i+1} = N{i};
    else
       X{i+1} = 1./(1+exp(-N{i})); % Activating
    end
    cost = cost + 0.5*lambda*(norm(U{i},'fro')^2 + norm(V{i},'fro')^2 + norm(W{i},'fro')^2+ norm(P{i},'fro')^2) ;
    if i+1 <= L
       rho = mean(X{i+1},5); %average activation value for sparsity
       part_delta{i+1} = beta*(-sparsityParam./rho + (1-sparsityParam)./(1-rho));
       cost = cost + beta * sum(sum(sum(sparsityParam*log (sparsityParam ./ rho) + (1-sparsityParam)*log((1-sparsityParam)./(1-rho))))); 
   end;
end;
%X{L+1} is now the output of the network. 
cost = cost + 0.5 * sum(sum(sum(sum((X{L+1} - Y).^2)))) / size(data,5);

% Back-propogation
grad = [];
if linear
    delta{L+1} = - (Y - X{L+1} );
else
    delta{L+1} = - (Y - X{L+1} ).* X{L+1} .* (1 - X{L+1});
end

for i=L:-1:1
    UU = {U{i}, V{i}, W{i},P{i}}; 
    if i>1
        delta{i} = (bsxfun(@plus, double(ttm(tensor(delta{i+1}), UU, (1:length(UU)), 't')), part_delta{i})) .* X{i} .* (1 - X{i});
        %delta{i} = (bsxfun(@plus, prod_mat_tensor_mat(U{i}', delta{i+1}, V{i}'), part_delta{i})).* X{i} .* (1 - X{i});
    end;
    Ud = UU;
    Ud{1} = eye(size(UU{1},2));
    tmpXd = ttm(tensor(X{i}), Ud, [1:4]);
    gradU = double(ttt(tensor(delta{i+1}), tmpXd, [2 3 4 5], [2 3 4 5]))/size(data,5)  + lambda * U{i};
    Ud = UU;
    Ud{2} = eye(size(UU{2},2));
    tmpXd = ttm(tensor(X{i}), Ud, [1:4]);
    gradV = double(ttt(tensor(delta{i+1}), tmpXd, [1 3 4 5], [1 3 4 5]))/size(data,5) + lambda * V{i};
    Ud = UU;
    Ud{3} = eye(size(UU{3},2));
    tmpXd = ttm(tensor(X{i}), Ud, [1:4]);
    gradW = double(ttt(tensor(delta{i+1}), tmpXd, [1 2 4 5], [1 2 4 5]))/size(data,5) + lambda * W{i};
    Ud = UU;
    Ud{4} = eye(size(UU{4},2));
    tmpXd = ttm(tensor(X{i}), Ud, [1:4]);
    gradP = double(ttt(tensor(delta{i+1}), tmpXd, [1 2 3 5], [1 2 3 5]))/size(data,5) + lambda * P{i};
    
    %gradU = sum_prod_tensors(prod_tensor_mat(delta{i+1}, V{i}'), permute(X{i}, [2,1,3]) )/size(data,4) + lambda * U{i};
    %gradV = sum_prod_tensors(prod_tensor_mat(permute(delta{i+1},[2,1,3]), U{i}'), X{i} )/size(data,4) + lambda * V{i};
    gradB = mean(delta{i+1},5);
    %gradB = sum(delta{i+1},4)/scale;
    grad = [gradU(:);gradV(:);gradW(:);gradP(:);gradB(:);grad];
end;

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

