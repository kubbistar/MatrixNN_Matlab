%% set seed: Linear
X=zeros(25,25,12,50);

X(:,:,1:4,1)=randn(25,25,4);

for i=5:8
    X(:,:,i,1)=X(:,:,i-4,1)';
end

X = calculate8_12(X);
% construct 25*25*12
XN=zeros(25,25,4,50);

for i=1:4
    XN(:,:,i,1)=0.4*X(:,:,i,1)+0.3*X(:,:,i+4,1)+0.3*X(:,:,i+8,1); %weighted pooling
end

X=XN;
%% set seed: Nonlinear
X=zeros(25,25,12,545);

X(:,:,1:4,1)=randn(25,25,4);

for i=5:8
    X(:,:,i,1)=X(:,:,i-4,1)';
end

X = calculate8_12(X);
% construct 25*25*12
XN=zeros(25,25,4,545);

for i=1:4
    XN(:,:,i,1)=0.4*X(:,:,i,1)+0.3*X(:,:,i+4,1)+0.3*X(:,:,i+8,1); %weighted pooling
end
X=XN;
%%   Linear relationship 
B1=randn(25,25)/10; 
B2= randn(25,25)/10;
B3= randn(4,4)/10;
B=randn(25,25,4)/10;%bias
UU = {B1, B2, B3};
patches.X(:,:,:,1)=X(:,:,:,1);
XX{1} = patches.X(:,:,:,1);

for h=2:50
    X(:,:,:,h) = double(ttm(tensor(XX{h-1}), UU, (1:length(UU)))...
        + ttm(tensor(B, size(B)), [1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1], 3)+0.01*chi2rnd(1,25,25,4)); 
    % 0.01*randn(25,25,4): assuming normal noise  
    % noise which is randn(25,25,4) can be changed to
    % 0.01*chi2rnd(1,25,25,4) Chi2 noise
    % 0.01*trnd(1,25,25,4)
    
           
           %ttm(tensor(B, [size(B) 1]), ones(1,545)', 4): original B
%     N{i} = double(ttm(tensor(XX{i}), UU, (1:length(UU))) ...
%         + ttm(tensor(B{i}, [size(B{i}) 1]), ones(1,size(Y,4))', 4));
%         (sample)
%     for k=5:8
%         X(:,:,k,h)=X(:,:,k-4,h)';
%     end
% 
%     X = calculate8_12(X);
%     
    patches.X(:,:,:,h)=X(:,:,:,h);
    XX{h}=patches.X(:,:,:,h);
end

Xi=X(:,:,:,2:49);
Y=X(:,:,:,3:50);
X=Xi;
%% Nonlinear relationship
%input layer: 25*25*4; hidden layer1: 8*8*4; hidden layer2: 12*12; output layer: 25*25*4
B1=cell(1,3);
B2=cell(1,3);
B3=cell(1,3);
B=cell(1,3);

B11=randn(8,25);
patches.B11=B11;
B1{1}=patches.B11;

B21=randn(8,25);
patches.B21=B21;
B2{1}= patches.B21;

B31=randn(4,4);
patches.B31=B31;
B3{1}= patches.B31;

BI1=randn(8,8,4);
patches.BI1=BI1;
B{1}=patches.BI1;%bias

B12=randn(12,8);
patches.B12=B12;
B1{2}=patches.B12;

B22= randn(12,8);
patches.B22=B22;
B2{2}=patches.B22;

B32= randn(4,4);
patches.B32=B32;
B3{2}=patches.B32;

BI2=randn(12,12,4);
patches.BI2=BI2;
B{2}=patches.BI2;%bias

B13=randn(25,12);
patches.B13=B13;
B1{3}=patches.B13;


B23= randn(25,12);
patches.B23=B23;
B2{3}=patches.B23;

B33= randn(4,4);
patches.B33=B33;
B3{3}=patches.B33;

BI3=randn(25,25,4);%bias
patches.BI3=BI3;
B{3}=patches.BI3;

%UU = {B1, B2, B3};
patches.X(:,:,:,1)=X(:,:,:,1); %X(:,:,:,1): 25*25*12
XX{1} = patches.X(:,:,:,1);


for o=2:545
    for p=1:3
        UU = {B1{p}, B2{p}, B3{p}};
%     X(:,:,:,k) = double(ttm(tensor(XX{i}), UU, (1:length(UU))) +...
%         + ttm(tensor(B, [size(B) 1]), ones(1,545)', 4)); 
    
        N{p} = double(ttm(tensor(XX{p}), UU, (1:length(UU)))...
            + ttm(tensor(B{p}, size(B{p}) ), eye(4,4), 3) );  
    %N{i} =bsxfun(@plus, prod_mat_tensor_mat(U{i}, X{i}, V{i}),  B{i});  % Summarising
%         if p+1>3
%            XX{p+1} = N{p};
%         else
           XX{p+1} = 1./(1+exp(-N{p}))+0.01*chi2rnd(1,size(N{p})); % Activating and adding noise
           % noise which is 0.01*randn(25,25,4) can be changed to
           % 0.01*chi2rnd(1,25,25,4), 0.01*trnd(1,25,25,4)
       
        
    end
    X(:,:,:,o)=XX{4}; %last X: 25*25*4  HELP!!!!!!!! HOW CAN I TRANSFER A CELL INTO AN ARRAY?
    
    patches.X(:,:,:,o)=X(:,:,:,o);
    XX{1}=patches.X(:,:,:,o); %% HELP!!!!!!! Confused if correct... Would Boyan mind reviewing this line?
end


Xi=X(:,:,:,2:544);
Y=X(:,:,:,3:545);
X=Xi;