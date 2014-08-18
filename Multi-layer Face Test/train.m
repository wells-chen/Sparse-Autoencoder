%% CS294A/CS294W Programming Assignment Starter Code

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  programming assignment. You will need to complete the code in sampleIMAGES.m,
%  sparseAutoencoderCost.m and computeNumericalGradient.m. 
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file. 
%
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.
clc;
clear all;
layer1patchsize=5;
layer2patchsize = 25*64;
hiddenSize1 = 64;     % number layer one of hidden units 
hiddenSize2 = 81;   % number layer two of hidden units 
visibleSize1 = layer1patchsize^2;   % number layer one of input units 
% number layer two of input units 
visibleSize2 = hiddenSize1*(layer2patchsize-layer1patchsize+1)^2;    
sparsityParam = 0.05;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Implement sampleIMAGES
%
%  After implementing sampleIMAGES, the display_network command should
%  display a random sample of 200 patches from the dataset
% patches1=sampleIMAGES(layer1patchsize);
% patches1 = loadData(layer1patchsize);
path = fullfile('K:','chenwei','programming','contrast','im_input_40by40.mat');
patches1 = Im_Input_conv(path);
% display_network(patches1(:,randi(size(patches1,2),200,1)),8);
display_network(patches1(:,1:100),12);

%  Obtain random parameters theta
%%======================================================================
%% STEP 4: After verifying that your implementation of
%  sparseAutoencoderCost is correct, You can start training your sparse
%  autoencoder with minFunc (L-BFGS).

%  Randomly initialize the parameters
theta1 = initializeParameters(hiddenSize1, visibleSize1);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 40;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta1, cost1] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize1, hiddenSize1, ...
                                   lambda, sparsityParam, ...
                                   beta, patches1), ...
                              theta1, options);

%%======================================================================
%% STEP 5: layer1
W1 = reshape(opttheta1(1:hiddenSize1*visibleSize1), hiddenSize1, visibleSize1);
W11= reshape(opttheta1(hiddenSize1*visibleSize1+1:hiddenSize1*visibleSize1*2), visibleSize1,hiddenSize1);
display_network(W1', 12);
data =Im_Input(path);
[traindata1,data_culve1]=layer_conv_max_contrast(data,W1,1);
% clear data;
% m是第一层滤波器数量，n,k是卷积后分割操作得到的patch大小
%输入数据数量为M2*N2
[M2,N2]=size(traindata1);
[m2,n2,k2] = size(traindata1{1,1});
visibleSize2 = m2*n2*k2;
hiddenSize2 = 36;
patches2 = zeros(visibleSize2,M2*N2);
for i=1:M2
    k=1;
    for j=1:N2
        temp=traindata1{i,j};
        patches2(:,k)=temp(:);
        k=k+1;
    end
end
clear temp;
save;
clearvars -except hiddenSize2 visibleSize2 sparsityParam lambda beta patches2 M2 N2;
theta2 = initializeParameters(hiddenSize2, visibleSize2);
%  Use minFunc to minimize the function
% for i=1:50
% patches2_temp = patches2(:,(i-1)*N2*M2/50+1:i*N2*M2/50);
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 40;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta2, cost2] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize2, hiddenSize2, ...
                                   lambda, sparsityParam, ...
                                   beta, patches2), ...
                              theta2, options);
% theta2 = opttheta2;
% end

%% layer2
load('matlab.mat');
W2_temp= reshape(opttheta2(1:hiddenSize2*visibleSize2), hiddenSize2, visibleSize2);
W22_temp = reshape(opttheta2(hiddenSize2*visibleSize2+1:hiddenSize2*visibleSize2*2),  visibleSize2,hiddenSize2);
b2_temp = opttheta2(2*hiddenSize2*visibleSize2+1:2*hiddenSize2*visibleSize2+hiddenSize2);
b22_temp = opttheta2(2*hiddenSize2*visibleSize2+hiddenSize2+1:end);
%暂时取10个
% for i=1:10
% patches2_temp = patches2(:,(i-1)*N2+1:i*N2);
% patches2_temp2 = W2_temp*patches2_temp;
% patches2_temp2 = patches2_temp2+repmat(b2_temp,1,size(patches2_temp2,2));
% patches2_temp = W22_temp*patches2_temp2;
% patches2_temp = patches2_temp + repmat(b22_temp,1,size(patches2_temp,2));
% data_original_cell = original_data_display(W11,patches2_temp,1,m2,i);
% end

W2{1,hiddenSize2}=[];
for i=1:hiddenSize2
    W2{1,i} = reshape(W2_temp(i,:),m2,n2,k2);
end

% W2__temp =W2;
% W2__temp{1,1}(4,:,:) = [0 0 1 0 0;0 0 1 0 0;0 0 0 0 0;0 0 0 0 0;0 0 1 0 0;]
W2_display(W11,W2);
flag=1;
for i=1:2:50-10+1
    for j=1:2:50-10+1
        data_back{flag} =data_culve{1}(i:i+10-1,j:j+10-1,)；
        flag=flag+1;
    end
end
W2_display(W1',data_back);
for i=1:3
    W2_replace{i}=data_back{118+i};
end




[traindata2,data_culve2]=layer_conv_max_contrast(data_culve1,W2,2);
% m是第一层滤波器数量，n,k是卷积后分割操作得到的patch大小
%输入数据数量为M2*N2
[M3,N3]=size(traindata2);
[m3,n3,k3] = size(traindata2{1,1});
visibleSize3 = m3*n3*k3;
hiddenSize3 = 0.6*m2*n2*k2;
patches3 = zeros(visibleSize3,M3*N3);
for i=1:M3
    for j=1:N3
        temp=traindata2{i,j};
        patches3(:,i)=temp(:);
    end
end
clear temp;
save;
clearvars -except hiddenSize3 visibleSize3 sparsityParam lambda beta patches3;
theta3 = initializeParameters(hiddenSize3, visibleSize3);
%  Use minFunc to minimize the function
% for i=1:50
% patches3_temp = patches3(:,(i-1)*N3*M3/50+1:i*N3*M3/50);
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 40;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

[opttheta3, cost3] = minFunc( @(p) sparseAutoencoderCost_sigal_loop(p, ...
                                   visibleSize3, hiddenSize3, ...
                                   lambda, sparsityParam, ...
                                   beta, patches3), ...
                              theta3, options);
% theta3=opttheta3;
% end

%% layer result
load('matlab.mat');
W3_temp= reshape(opttheta3(1:hiddenSize3*visibleSize3), hiddenSize3, visibleSize3);
W3{1,hiddenSize3}=[];
for i=1:hiddenSize3
    W3{1,i} = reshape(W3_temp(i,:),m3,n3,k3);
end
% [traindata3,data_culve3]=layer_conv_max_contrast(data_culve2,W3);

path_test = fullfile('K:','chenwei','programming','contrast','im_input_40by40_test.mat');
patches1_test = Im_Input(path_test);
[traindata1_test,data_culve1_test]=layer_conv_max_contrast(data,W1,1);
[traindata2_test,data_culve2_test]=layer_conv_max_contrast(data_culve1_test,W2,2);
[traindata3_test,data_culve3_test]=layer_conv_max_contrast(data_culve2_test,W3,3);
[R,P,opt_location] = Precision_conv(data_culve3_test);




                          
                          
%                           
% 
% %% layer 2
% b1 = opttheta1(2*hiddenSize1*visibleSize1+1:2*hiddenSize1*visibleSize1+hiddenSize1);
% patches2 = W1*patches1;
% patches2 = patches2+repmat(b1,1,size(patches2,2));
% % patches2 = maxpooling(patches2,3);
% % patches2 = contrast_nomalization(patches2,3);
% visibleSize2 =size(patches2,1);
% hiddenSize2 = 400;
% [opttheta2, cost2]=layer(visibleSize2, hiddenSize2, ...
%      lambda, sparsityParam,beta,patches2);
% %%============
% %% layer3
% W2= reshape(opttheta2(1:hiddenSize2*visibleSize2), hiddenSize2, visibleSize2);
% b2 = opttheta2(2*hiddenSize2*visibleSize2+1:2*hiddenSize2*visibleSize2+hiddenSize2);
% patches3 = W2*patches2;
% patches3 = patches3+repmat(b2,1,size(patches3,2));
% % patches3 = maxpooling(patches3,3);
% % patches3 = contrast_nomalization(patches3,3);
% visibleSize3 =size(patches3,1);
% hiddenSize3 = 64;
% [m,n]=size(patches3);
% for i=1:10 
%     k=1;
%     patches3_negative_sample{i} = patches3;
%     for j=i:10:n
%         patches3_sample{i}(:,k)=patches3(:,j);
%         patches3_negative_sample{i}(:,k)=[];
%         k = k+ 1;
%     end
% end
% [opttheta3, cost3]=layer(visibleSize3, hiddenSize3, ...
%      lambda, sparsityParam,beta,patches3_sample{5});
% %%=====================================
% %% result process
% path_test = fullfile('K:','chenwei','programming','contrast','im_input_40by40_test.mat');
% patches_test = Im_Input(path_test);
% display_network(patches_test(:,1:100),12);
% patches2_test = W1*patches_test;
% patches2_test = patches2_test+repmat(b1,1,size(patches2_test,2));
% patches3_test = W2*patches2_test;
% patches3_test = patches3_test+repmat(b2,1,size(patches3_test,2));
% W3= reshape(opttheta3(1:hiddenSize3*visibleSize3), hiddenSize3, visibleSize3);
% b3 = opttheta3(2*hiddenSize3*visibleSize3+1:2*hiddenSize3*visibleSize3+hiddenSize3);
% training_data = W3*patches3_test;
% training_data = training_data+repmat(b3,1,size(training_data,2));
% [R,P,Q] = Precision(training_data);
% 
% % W=W31*W2*W1;
% % figure,display_network(W',12);



%%==========================


%% prepared 
% W2 = reshape(opttheta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize*visibleSize), visibleSize,hiddenSize);
% b1 = opttheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
% b2 = opttheta(2*hiddenSize*visibleSize+hiddenSize+1:end);
% print -djpeg weights.jpg   % save the visualization to a file 
% [opttheta2,cost2] = layer2(layer2patchsize,visibleSize2,hiddenSize2,visibleSize1,hiddenSize1,lambda, ...
%                                      sparsityParam, beta,W1);
% W2 = reshape(opttheta2(1:hiddenSize2*visibleSize2), hiddenSize2, visibleSize2);
% figure,displaylayer2(hiddenSize1,W2,W1);
% figure,displaylayer2_new(hiddenSize1,W2,W1);
% figure,displaylayer2_new2(hiddenSize1,layer2patchsize,W2,W1);
%%=====================================