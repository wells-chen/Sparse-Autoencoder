function temp=temp()
% m是第一层滤波器数量，n,k是卷积后分割操作得到的patch大小
%输入数据数量为M2*N2
M2=size(traindata1,2);
N2=size(traindata1{1,1},2);
[m2,n2,k2] = size(traindata1{1,1}{1,1});
visibleSize2 = m2*n2*k2;
hiddenSize2 = 0.6*m2*n2*k2;
patches2 = zeros(visibleSize2,M2*N2);
for i=1:M2
    for j=1:N2
        temp=traindata1{1,i}{1,j};
        patches2(:,i)=temp(:);
    end
end
clear temp;
theta1 = initializeParameters(hiddenSize2, visibleSize2);
%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta2, cost2] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize2, hiddenSize2, ...
                                   lambda, sparsityParam, ...
                                   beta, patches2), ...
                              theta1, options);





end
