function theta = InitialiseTensor4DNeuralNetwork(layerSize) 

%% Initialize parameters randomly based on layer sizes.
% r  = sqrt(6) / sqrt(hiddenSize.I2*hiddenSize.J2+visibleSize.I1*visibleSize.J1+1);   
% we'll choose weights uniformly from the interval [-r, r]
% U1 = rand(hiddenSize.I2, visibleSize.I1) * 2 * r - r;
% V1 = rand(hiddenSize.J2, visibleSize.J1) * 2 * r - r;
% U2 = rand(visibleSize.I1, hiddenSize.I2) * 2 * r - r;
% V2 = rand(visibleSize.J1, hiddenSize.J2) * 2 * r - r;
% 
% B1 = zeros(hiddenSize.I2, hiddenSize.J2);
% B2 = zeros(visibleSize.I1, visibleSize.J1);
% 
% % Convert weights and bias gradients to the vector form.
% % This step will "unroll" (flatten and concatenate together) all 
% % your parameters into a vector, which can then be used with minFunc. 
% theta = [U1(:) ; V1(:); U2(:); V2(:); B1(:) ; B2(:)];

%% Assume multiple layer structure, L layers of neurons + input layer
% layerSize contains L+1 cells
% layerSize{1} is the size of the input layer
% layerSize{end} contains the size of the output layer
% Edited on 29/10/2015, Yi Guo
N = length(layerSize); 
L = N-1;
total = 1;
for i =1:L
    total = total + layerSize{i}.I*layerSize{i}.J*layerSize{i}.K*layer;
end
r = sqrt(L*3)/sqrt(total);
r = 0.3;
% theta has [U1 V1 B1 U2 V2 B2 ...] format. 
% X_l is of size I_l by J_l
% U_l is of size I_{l+1} by I_l
% V_l is of size J_l by J_{l+1}
% B_l is of size I_{l+1} by J_{l+1} 

%theta = [rand(hiddenSize{1}.I*visibleSize.I,1) * 2 * r - r; ...
%    rand(hiddenSize{1}.J*visibleSize.J,1) * 2 * r - r; zeros(hiddenSize{1}.I*hiddenSize{1}.J,1)];

theta = [];
for i=2:N
    if i==N && ~isfield(layerSize{i},'I')
        theta = [theta;rand(layerSize{i-1}.I*layerSize{i-1}.J*layerSize{i},1) * 2 * r - r;...
            rand(layerSize{i},1) * 2 * r - r];
    else
        theta = [theta;rand(layerSize{i}.I*layerSize{i-1}.I,1) * 2 * r - r;...
            rand(layerSize{i}.J*layerSize{i-1}.J,1) * 2 * r - r; rand(layerSize{i}.K*layerSize{i-1}.K,1) * 2 * r - r; zeros(layerSize{i}.I*layerSize{i}.J*layerSize{i}.K,1)];
    end
end;

