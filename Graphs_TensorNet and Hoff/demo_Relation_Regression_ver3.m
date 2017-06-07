function [U, V, W, B] = demo_Relation_Regression_ver3(sparsityParam,lambda,beta,maxIters)
% Version 1.000
%
% Author:  Professor Junbin Gao
% Copyright all reserved, last modified 24 November 2016

% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.
%%======================================================================

%% STEP 0: Here we provide the relevant parameters values that will
%  allow a sparse autoencoder to get good weights; you do not need to
%  change the parameters below.
%

% You may test different sizes
layerSize{1}.I = 25; layerSize{1}.J = 25; layerSize{1}.K = 4;% the size of input layer
layerSize{2}.I = 8; layerSize{2}.J = 8; layerSize{2}.K = 4;% the size of hidden layer
layerSize{3}.I = 12; layerSize{3}.J = 12; layerSize{3}.K = 4;% the size of hidden layer
layerSize{4}.I = 25; layerSize{4}.J = 25; layerSize{4}.K = 4;% the size of Output layer
% layerSize{1}.I = 10; layerSize{1}.J = 10; layerSize{1}.K = 4;% the size of input layer
% layerSize{2}.I = 5; layerSize{2}.J = 5; layerSize{2}.K = 4;% the size of hidden layer
% layerSize{3}.I = 10; layerSize{3}.J = 10; layerSize{3}.K = 4;% the size of hidden layer

sparsityParam = 0.5;         % desired average activation of the hidden units.
lambda = 1e-3;%2.3300e-86;%3e-3;         % weight decay parameter
beta = 0.1;  %3;               % weight of sparsity penalty term
% 
maxIters = 1000;    % 1000  Maximal iteration
MaxFunEvals = 2000;        % Maximal number of function evaluations
%%======================================================================
%% STEP 1: Preparing Data  
%  % Please change the path to the data set on your machine
load('/Users/Yvonne/Documents/Matrix_NN/mtxydata.mat')
%load('/Users/Yvonne/Documents/Matrix_NN/NLsimNorml.mat')
%load('/Users/Yvonne/Documents/Matrix_NN/LsimStudentT.mat')
Y(isnan(Y)) = 0;

Xtrain = X(:,:,:,1:400); %NLsim
Xtest = X(:,:,:,401:543); %NLsim
Ytrain = Y(:,:,:,1:400); %NLsim
Ytest =  Y(:,:,:,401:543); %NLsim

% Xtrain = X(:,:,:,1:40); %Lsim
% Xtest = X(:,:,:,41:48); %Lsim
% Ytrain = Y(:,:,:,1:40); %Lsim
% Ytest =  Y(:,:,:,41:48); %Lsim


% We test on the first action data separately and put data in 3rd tensor form
X = squeeze(X(:,:,:,1,1,:)); 
%X=1./(1+exp(-X));
Y = squeeze(Y(:,:,:,:));
%Y=1./(1+exp(-Y));
patches.Xtrain = Xtrain;  %(1:10, 1:10, 1:4, 1:2);   % input
patches.Ytrain = Ytrain;  %(1:10, 1:10, 1:4, 1:2);   % output

patches.Xtest = Xtest;  %(1:10, 1:10, 1:4, 1:2);   % input
patches.Ytest = Ytest;  %(1:10, 1:10, 1:4, 1:2);   % output
%%====================================================================== 
% Obtain random parameters theta
rng('default');
rng(0) ;
Train_err = [];
Test_err = [];
theta = InitialiseTensor3DNeuralNetwork(layerSize);

%%======================================================================
%% STEP 2: Implement MatrixNeuralNetworkRegressionCost
%
%  You can implement all of the components (squared error cost, weight decay term,
%  sparsity penalty) in the cost function at once, but it may be easier to do
%  it step-by-step and run gradient checking (see STEP 3) after each step.  We
%  suggest implementing the sparseAutoencoderCost function using the following steps:
%
%  (a) Implement forward propagation in your neural network, and implement the
%      squared error term of the cost function.  Implement backpropagation to
%      compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking
%      to verify that the calculations corresponding to the squared error cost
%      term are correct.
%
%  (b) Add in the weight decay term (in both the cost function and the derivative
%      calculations), then re-run Gradient Checking to verify correctness.
%
%  (c) Add in the sparsity penalty term, then re-run Gradient Checking to
%      verify correctness.
%
%  Feel free to change the training settings when debugging your
%  code.  (For example, reducing the training set size or
%  number of hidden units may make your code run faster; and setting beta
%  and/or lambda to zero may be helpful for debugging.)  However, in your
%  final submission of the visualized weights, please use parameters we
%  gave in Step 0 above.
[cost, grad] = Tensor3DNeuralNetworkRegressionCost(theta, layerSize, lambda, sparsityParam, beta, patches.Xtrain, patches.Ytrain, 'linear');

%%======================================================================
%% STEP 3: Gradient Checking
%
% Hint: If you are debugging your code, performing gradient checking on smaller models
% and smaller training sets (e.g., using only 10 training examples and 1-2 hidden
% units) may speed things up.

% First, lets make sure your numerical gradient computation is correct for a
% simple function.  After you have implemented computeNumericalGradient.m,
% run the following:
%checkNumericalGradient();

% Now we can use it to check your cost function and derivative calculations
% for the sparse autoencoder.
% numgrad = computeNumericalGradient( @(x) Tensor3DNeuralNetworkRegressionCost(x, layerSize, lambda,sparsityParam, beta, patches.X, patches.Y, 'linear'), theta);

% Use this to visually compare the gradients side by side
% disp([numgrad grad]);

% Compare numerically computed gradients with the ones obtained from backpropagation
% diff = norm(numgrad-grad)/norm(numgrad+grad);
% disp(diff); % Should be small. In our implementation, these values are
% usually less than 1e-9.

% When you got this working, Congratulations!!!

%%======================================================================
%% STEP 4: After verifying that your implementation of
%  sparseAutoencoderCost is correct, You can start training your sparse
%  autoencoder with minFunc (L-BFGS).

% Randomly initialize the parameters
theta = InitialiseTensor3DNeuralNetwork(layerSize); 

%  Use minFunc to minimize the function
%  Use mcc -m demo_MNIST_classification.m -a '~/research projects/starter/' instead to
%  include the path. Note the ' in the above command!!
% or to be more comprehensive 
% mcc -m demo_MNIST_classification.m -a '~/research projects/starter/' -d ./test/ -T compile:exe
% Issue the following to run it:
% ./run_demo_MNIST_classification.sh ~/MATLAB/MATLAB_Compiler_Runtime/v83

if(isdeployed==false) % This is necessary if it is deployed. No addpath in deploy version. 
    % You need to change to your path to minFunc.
    addpath('/Users/Yvonne/Documents/Matrix_NN/minFunc/')
end
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
% function. Generally, for minFunc to work, you
% need a function pointer with two outputs: the
% function value and the gradient. In our problem,
% MatrixNeuralNetworkRegressionCost.m satisfies this.
options.maxIter = maxIters;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';
options.TolFun=1e-10; %added
options.MaxFunEvals = MaxFunEvals;	

[opttheta, cost,~,info] = minFunc( @(p) Tensor3DNeuralNetworkRegressionCost(p, ...
    layerSize,lambda, sparsityParam, beta, patches.Xtrain,patches.Ytrain, 'linear'), ...
    theta, options);

if(isdeployed==false)
    %rmpath('~/research projects/starter/minFunc/')
    rmpath('/Users/Yvonne/Documents/Matrix_NN/minFunc/')
end
%%======================================================================
%% STEP 5: Visualization
L = length(layerSize)-1;
count_start = 1;
for i=2:L+1
    count_end = count_start + layerSize{i}.I*layerSize{i-1}.I - 1;
    U{i-1} = reshape(opttheta(count_start:count_end), layerSize{i}.I, layerSize{i-1}.I);
%     grad.U{i-1} = zeros(size(U{i-1}));
    
    count_start = count_end + 1;
    count_end = count_start+layerSize{i}.J*layerSize{i-1}.J - 1;
    V{i-1} = reshape(opttheta(count_start:count_end), layerSize{i}.J, layerSize{i-1}.J);
%     grad.V{i-1} = zeros(size(V{i-1}));

    count_start = count_end + 1;
    count_end = count_start+layerSize{i}.K*layerSize{i-1}.K - 1;
    W{i-1} = reshape(opttheta(count_start:count_end), layerSize{i}.K, layerSize{i-1}.K);
%     grad.V{i-1} = zeros(size(V{i-1}));
    
    count_start = count_end + 1;
    count_end = count_start+layerSize{i}.I*layerSize{i}.J*layerSize{i}.K - 1;
    B{i-1} = reshape(opttheta(count_start:count_end), layerSize{i}.I,layerSize{i}.J,layerSize{i}.K);
    
    count_start = count_end + 1;
%     grad.B{i-1} = zeros(size(B{i-1}));
end;

XX{1} = patches.Xtrain; % The input is X_1.
N=cell(1,L); % added
for i=1:L
    UU = {U{i}, V{i}, W{i}};
    N{i} = double(ttm(tensor(XX{i}), UU, (1:length(UU))) + ttm(tensor(B{i}, [size(B{i}) 1]), ones(1,size(Ytrain,4))', 4));  
 
    %N{i} =bsxfun(@plus, prod_mat_tensor_mat(U{i}, XX{i}, V{i}),  B{i});  % Summarising
    if (i+1>L)
        XX{i+1} = N{i};
    else    
        XX{i+1} = 1./(1+exp(-N{i})); % Acting
    end    
end;
Train_err = sum(sum(sum(sum((XX{L+1} - patches.Ytrain).^2))))/(size(Ytrain,1)*size(Ytrain,2)*size(Ytrain,3)*size(Ytrain,4));


% We can simply run one step neural network to get the cost, i.e., the
% error.
[test_cost, ~] = Tensor3DNeuralNetworkRegressionCost(opttheta, layerSize, lambda, sparsityParam, beta, patches.Xtest, patches.Ytest, 'linear');
Test_err = [Test_err, test_cost(1)/(size(Ytrain,1)*size(Ytrain,2)*size(Ytrain,3))];
disp(['The training error: ',num2str(Train_err)]);
disp(['The testing cost: ',num2str(Test_err)]);

B1 = U{3}*U{2}*U{1};
B2 = V{3}*V{2}*V{1};
figure
imagesc(B1)
colormap(gray)
figure
imagesc(B2)
colormap(gray)

countries = {'AFG', 'AUS', 'CHN', 'DEU', 'EGY', 'FRA', 'GBR', 'GEO', 'IND', 'IRN', 'IRQ', 'ISR'...
    ,  'JPN', 'KOR', 'LBN', 'PAK', 'PRK', 'PSE', 'RUS', 'SDN', 'SYR', 'TUR', 'TWN', 'UKR','USA'};

% % Setting a threshold
% threshold = 0.6*max(max(abs(B1)));
% B1(abs(B1)<threshold) = 0;
% [ix_row, ix_col] = find(B1);
% n = length(ix_row);
% 
% A  = rand(3, 25);
% figure
% scatter(A(1,:), A(2,:), A(3, :));
% %scatter3(Z(1,:), Z(2,:), Z(3, :));
% hold on
% % text(Z(1,1), Z(2,1), Z(3,1), countries{1});
% text(A(1,:), A(2,:), A(3,:), countries);
% 
% 
% G = digraph(B1,countries);
% 
% threshold = 0.6*max(max(abs(B2)));
% B2(abs(B2)<threshold) = 0;
% [ix_row, ix_col] = find(B2);
% n = length(ix_row);
% 
% Z  = rand(3, 25);
% figure
% scatter(Z(1,:), Z(2,:), Z(3, :));
% %scatter3(Z(1,:), Z(2,:), Z(3, :));
% hold on
% % text(Z(1,1), Z(2,1), Z(3,1), countries{1});
% text(Z(1,:), Z(2,:), Z(3,:), countries);
% 
% 
% K = digraph(B2,countries);
% figure
% plot(G)
% figure
% plot(K)

% Setting a threshold
thresholdB1 = 0.5*max(max(abs(B1)));
B1(abs(B1)<thresholdB1) = 0;

[ix_rowB1, ix_colB1] = find(B1);
nB1 = length(ix_rowB1);

figure('name','directed graph','NumberTitle','off');
GB1 = digraph(B1,countries);

% ------------------
% Uncomment the following lines if you want 2-d view

% plot(G);
% ------------------

thresholdB2 = 0.5*max(max(abs(B2)));
B2(abs(B2)<thresholdB2) = 0;

[ix_rowB2, ix_colB2] = find(B2);
nB2 = length(ix_rowB2);

figure('name','directed graph','NumberTitle','off');
GB2 = digraph(B2,countries);
% ------------------
% Comment the following lines if you do not want 3-d view

%% m = size(G.Edges,1);
XYZ  = rand(25,3);
x= XYZ(:,2);
y= XYZ(:,1);
z= XYZ(:,3);
figure
pB1 = plot(GB1,'XData',x,'YData',y,'ZData',z,'EdgeLabel',GB1.Edges.Weight);
pB1.LineWidth=4;
view([45 45 45]);
figure
pB2 = plot(GB2,'XData',x,'YData',y,'ZData',z,'EdgeLabel',GB2.Edges.Weight);
pB2.LineWidth=4;
view([45 45 45]);

% ------------------
 
% the last stage, the output layer can be generated/calculated using the
% theta function (activation function)
% 
 