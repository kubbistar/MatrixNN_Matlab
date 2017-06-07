%let us check gradient
randn('state',100);
rand('state',100);

visSize.X = 4;
visSize.Y = 4;
hidSize.X = 10;
hidSize.Y = 6;
penSize.X = 4;
penSize.Y = 5;
numlab = 10;
N = 100;

U11 = rand(hidSize.X, visSize.X);
V11 = rand(hidSize.Y, visSize.Y);
U12 = rand(hidSize.X, penSize.X);
V12 = rand(hidSize.Y, penSize.Y);
U2 = rand(penSize.X, hidSize.X);
V2 = rand(penSize.Y, hidSize.Y);
W_class = rand(penSize.X, penSize.Y, numlab);
hiddenB1 = rand(hidSize.X, hidSize.Y);
penB2 = rand(penSize.X, penSize.Y);
topB3 = rand(1, numlab);

VV = [topB3(:); W_class(:); penB2(:) ; U2(:) ; V2(:); hiddenB1(:); U11(:); V11(:); U12(:); V12(:)];
Dim = [visSize.X, visSize.Y, hidSize.X, hidSize.Y, penSize.X, penSize.Y, numlab]; 
data = rand(visSize.X, visSize.Y, N);
targets = zeros(numlab, N);
idx = randi([1 numlab], N,1);
for i = 1:N
    targets(idx(i), i) = 1;
end
temp_h2 = rand(penSize.X, penSize.Y, N);

numgrad = computeNumericalGradient( @(x) crossEntropyCost(x, visSize, hidSize, penSize, numlab, data, targets, temp_h2), VV);
[f, grad] = crossEntropyCost(VV, visSize, hidSize, penSize, numlab, data, targets, temp_h2);
plot(numgrad-grad)
diff = norm(numgrad-grad)/norm(numgrad+grad)