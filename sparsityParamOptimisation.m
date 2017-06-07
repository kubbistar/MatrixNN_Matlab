clear all
close all
rng('default');
rng(100) ;

Train_err = [];
Test_err = [];
sparsityParams=logspace(-3,1,100);
%lambdas= logspace(-6,-1,40); 
%maxIters = 1000;
maxIters = 100;
%% STEP 0: Preparing Data  
load('mtxydata.mat')
Y(isnan(Y)) = 0;
% We test on the first action data separately and put data in 3rd tensor form
Xtrain = squeeze(X(:,:,1,1,1,1:400)); 
Xtest = squeeze(X(:,:,1,1,1,401:433)); 

Ytrain = squeeze(Y(:,:,1,1:400));
Ytest = squeeze(Y(:,:,1,401:433));
 

for i = 1:length(sparsityParams)   
    sparsityParam=sparsityParams(i);
    %lambda = lambdas(i);%1e-3
    %sparsityParam = 0.1; %original optimal sparsityParam=0.5
                         % We cannot set sparsityParam = 0 as our program
                         % calculate log(sparsityParam)
                         % If we don't wish to use sparsity, the safest way
                         % is to set beta = 0
    beta = 0;    %3.0;
    lambda=4.6416e-05;
      layerSize{1}.I = 25; layerSize{1}.J = 25; % the size of input layer
      layerSize{2}.I = 40; layerSize{2}.J = 40; % the size of hidden layer
      layerSize{3}.I = 40; layerSize{3}.J = 40; % the size of hidden layer
      layerSize{4}.I = 25; layerSize{4}.J = 25; % the size of Output layer
%     layerSize{1}.I = 25; layerSize{1}.J = 25; % the size of input layer
%     layerSize{2}.I = 15; layerSize{2}.J = 15; % the size of hidden layer
%     layerSize{3}.I = 20; layerSize{3}.J = 20; % the size of hidden layer
%     layerSize{4}.I = 25; layerSize{4}.J = 25; % the size of Output layer
%     
    %%======================================================================
    %% Step 1:  Initializing the network
    % Obtain random parameters theta
    theta = InitialiseMatrixNeuralNetwork(layerSize);
    
    %%======================================================================
    %% STEP 2: Implement MatrixNeuralNetworkRegressionCost
 
    % [cost, grad] = MatrixNeuralNetworkRegressionCost(theta, layerSize, lambda, sparsityParam, beta, Xtrain, Ttrain, 'linear');

    %%======================================================================
    %% STEP 3: Gradient Checking
    % removed

    %%======================================================================
    %% STEP 4: After verifying that your implementation of
    % theta = InitialiseMatrixNeuralNetwork(layerSize); 
    if(isdeployed==false) % This is necessary if it is deployed. No addpath in deploy version. 
        % You need to change to your path to minFunc.
        %addpath('/Users/jbgao/Documents/Gaofiles/matlab/Hinton/UFLDL/starter/minFunc/')
        addpath('/Users/Yvonne/Documents/Matrix_NN/minFunc')
        %addpath('./minFunc/')
    end
    options.Method = 'lbfgs'; 
    % Here, we use L-BFGS to optimize our cost function.  
    options.maxIter = maxIters;	  % Maximum number of iterations of L-BFGS to run
    options.display = 'on';  %'final';    %not really good to show nothing of the program iterations, so better no 'off'
    options.MaxFunEvals = maxIters + 100;

    [opttheta, cost] = minFunc( @(p) MatrixNeuralNetworkRegressionCost(p, ...
        layerSize,lambda, sparsityParam, beta, Xtrain, Ytrain, 'linear'), ...
        theta, options);

    if(isdeployed==false)
        rmpath('/Users/Yvonne/Documents/Matrix_NN/minFunc')
        %rmpath('/Users/jbgao/Documents/Gaofiles/matlab/Hinton/UFLDL/starter/minFunc/')
        %rmpath('./minFunc/')
    end
    %%======================================================================
    %% STEP 5: Visualization
    L = length(layerSize)-1;
    count_start = 1;
    for i=2:L+1
          count_end = count_start + layerSize{i}.I*layerSize{i-1}.I - 1;
          U{i-1} = reshape(opttheta(count_start:count_end), layerSize{i}.I, layerSize{i-1}.I);

          count_start = count_end + 1;
          count_end = count_start+layerSize{i}.J*layerSize{i-1}.J - 1;
          V{i-1} = reshape(opttheta(count_start:count_end), layerSize{i}.J, layerSize{i-1}.J);

          count_start = count_end + 1;
          count_end = count_start+layerSize{i}.I*layerSize{i}.J - 1;
          B{i-1} = reshape(opttheta(count_start:count_end), layerSize{i}.I,layerSize{i}.J);

          count_start = count_end + 1;
    end;
    XX{1} = Xtrain; % The input is X_1.
    N = cell(1,L);
    for i=1:L
        N{i} =bsxfun(@plus, prod_mat_tensor_mat(U{i}, XX{i}, V{i}),  B{i});  % Summarising
        if (i+1>L)
            XX{i+1} = N{i};   % The last layer is linear
        else    
            XX{i+1} = 1./(1+exp(-N{i})); % Acting
        end    
    end;
    Train_err = [Train_err, sum(sum(sum((XX{L+1} - Ytrain).^2)))/(size(Ytrain,1)*size(Ytrain,2)*size(Ytrain,3))];


% We can simply run one step neural network to get the test cost, i.e., the error.j
    [test_cost, ~] = MatrixNeuralNetworkRegressionCost(opttheta, layerSize, lambda, sparsityParam, beta, Xtest, Ytest, 'linear');
   
    Test_err = [Test_err, test_cost(1)/(size(Ytrain,1)*size(Ytrain,2))];
   % disp(['The training error: ',num2str(Train_err)]);
   % disp(['The testing error: ',num2str(Test_err)]);
   
end
plot(sparsityParams, Test_err, '-r')
 
[best_sparsityParam_testerror,idx] = min(Test_err)   %--to find the smallest value and the position or index

% You can write another program to change the size
%  layerSize{1}.I = 25; layerSize{1}.J = 25; % the size of input layer
%    layerSize{2}.I = 40; layerSize{2}.J = 40; % the size of hidden layer
%    layerSize{3}.I = 40; layerSize{3}.J = 40; % the size of hidden layer
%    layerSize{4}.I = 25; layerSize{4}.J = 25; % the size of Output layer
%  or inner size to 100.  I think that is enough
% 
% For each case of sizes, find out the best lambda
%
% Finally with these best_lambda(s) (or close to it) write another program
% where you change sparsity.
%
% Use this strategy:
% 1. When inner size is small, sparsityParam should be small too. say
% sparsityParam = 0.01 (or 0.05); when you have size such as 100, then
% sparsityParam = 0.7 (0r 0.8) etc
% 2. After you have decided a sparsity, you can change beta.  However in
% general, beta is not sensitive, I may suggest you fix beat =2.5 or 3.
% Just test sparsityParam for different possible values.
