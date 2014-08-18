function [opttheta,cost] = optimize_layer_output(traindata,hidden_value_rate)
%���վ����maxpooling��contrast normalization ��ָ�����ݽ����Ż�
% m�ǵ�һ���˲���������n,k�Ǿ����ָ�����õ���patch��С
%������������ΪM2*N2
M=size(traindata,2);
N=size(traindata{1,1},2);
[m,n,k] = size(traindata{1,1}{1,1});
visibleSize = m*n*k;
hiddenSize = hidden_value_rate*m*n*k;
patches = zeros(visibleSize,M*N);
for i=1:M
    for j=1:N
        temp=traindata{1,i}{1,j};
        patches(:,i)=temp(:);
    end
end
clear temp;
theta1 = initializeParameters(hiddenSize, visibleSize);
%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta1, options);





end