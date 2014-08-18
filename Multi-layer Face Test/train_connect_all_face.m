%% CS294A/CS294W Programming Assignment Starter Code

%  Instructions
%  ------------
%
%  This file contains code that use to train the second layers  and
%  dispaly the result of the train neurons by linear combination
%  the data have been proceessed (including
%  convolution,maxpooling,normalization).However it is not to be split.
%
%
% m是第一层滤波器数量，n,k是卷积后分割操作得到的patch大小
%输入数据数量为M2*N2
clc;
clear all;
sparsityParam = 0.2;   % desired average activation of the hidden units.
% (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
%  in the lecture notes).
lambda = 0.0001;     % weight decay parameter
beta = 3;            % weight of sparsity penalty term
path = fullfile('K:','chenwei','programming','face_data','neu_l1_3_12.mat');
load(path);
hiddenSize1=8;
layer1patchsize=13;
W1=zeros(hiddenSize1,layer1patchsize*layer1patchsize);
for i=1:hiddenSize1
    W1(i,:) = neu{i}.w(:);
end
display_network(W1',12);
path = fullfile('K:','chenwei','programming','face_data','yres_sam_l1_4_9_train.mat');
load(path);
data_culve1{300}=[];
for i=1:300
    temp=zeros(50,50,8);
    for j=1:8
        temp(:,:,j)=yres_sam{i}(:,:,:,j);
    end
    data_culve1{i} = temp;
end
clear temp yres_sam;


% m是第一层滤波器数量，n,k是卷积后分割操作得到的patch大小
%输入数据数量为M2*N2
[M2,N2]=size(data_culve1);
[m2,n2,k2] = size(data_culve1{1,1});
visibleSize2 = m2*n2*k2;
hiddenSize2 = 100;
patches2 = zeros(visibleSize2,M2*N2); k=1;
for i=1:M2
    temp=data_culve1{i};
    patches2(:,k)=temp(:);
    k=k+1;
end
clear temp k;
mkdir('data_save\all_con_W3_200\');
save 'data_save\all_con_W3_200\layerone';
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
options.maxIter = 60;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';

[opttheta2, cost2] = minFunc( @(p) sparseAutoencoderCost(p, ...
    visibleSize2, hiddenSize2, ...
    lambda, sparsityParam, ...
    beta, patches2), ...
    theta2, options);

% [opttheta2, cost2]=layer(visibleSize2, hiddenSize2,lambda, sparsityParam,beta, patches2);
%% layer2
load('data_save\all_con_W3_200\layerone.mat');
W2_temp= reshape(opttheta2(1:hiddenSize2*visibleSize2), hiddenSize2, visibleSize2);
W22_temp = reshape(opttheta2(hiddenSize2*visibleSize2+1:hiddenSize2*visibleSize2*2),  visibleSize2,hiddenSize2);
b2_temp = opttheta2(2*hiddenSize2*visibleSize2+1:2*hiddenSize2*visibleSize2+hiddenSize2);
b22_temp = opttheta2(2*hiddenSize2*visibleSize2+hiddenSize2+1:end);

W2{hiddenSize2}=[];
for i=1:hiddenSize2
    W2{i} = reshape(W2_temp(i,:),m2,n2,k2);
end

save 'data_save\all_con_W3_200\W2';
% load('data_save\W2.mat');
W2_display(W1',W2);
W2_display_new_one(W1',W2);
% data{1}=data_culve1{1};
% [traindata2,data_culve2]=layers_training_con_all(data_culve1,W2,2);
% save 'data_save\org_noise\final';
save ‘W_value’ W1 W2;
clearvars -except W1 W2;

%% positive data testing
% clear; 正样本1
path_testh = fullfile('K:','chenwei','programming','face_data','yres_sam_l1_4_9_test.mat');
load(path_testh);
% load('data_save\org\final.mat')
M=size(yres_sam,2);
[m,n,t,k]=size(yres_sam{1});
data_culve1_test{M}=[];
for i=1:M
    temp=zeros(m,n,k);
    for j=1:k
        temp(:,:,j)=yres_sam{i}(:,:,:,j);
    end
    data_culve1_test{i} = temp;
end
clear temp yres_sam;

[m,n,k]=size(data_culve1_test{1});
temp =zeros(m+8,n+8,k);
data_culve1_test_extend{size(data_culve1_test,2)}=[];
for i=1:size(data_culve1_test,2)
    temp(5:m+4,5:n+4,:)=data_culve1_test{i}(:,:,:);
    data_culve1_test_extend{i} = temp;
end
clear temp m n k;

[traindata2_test,data_culve2_test]=layers_training_con_all(data_culve1_test_extend,W2,3);
%%数据扩充
flag=1;
for i=1:312
data_culve2_test_more{flag}=data_culve2_test{i};
flag=flag+1;
end
save 'data_save\all_con_W3_200\data_culve3_no_noise_p' data_culve2_test;

%% 正样本2-噪声1
path_testh = fullfile('K:','chenwei','programming','face_data','data_noise','yres_sam_l1_5_19_jitter.mat');
load(path_testh);
% load('data_save\org\final.mat')
M=size(yres_sam,2);
[m,n,t,k]=size(yres_sam{1});
data_culve1_test{M}=[];
for i=1:M
    temp=zeros(m,n,k);
    for j=1:k
        temp(:,:,j)=yres_sam{i}(:,:,:,j);
    end
    data_culve1_test{i} = temp;
end
clear temp yres_sam;

[m,n,k]=size(data_culve1_test{1});
temp =zeros(m+8,n+8,k);
data_culve1_test_extend{size(data_culve1_test,2)}=[];
for i=1:size(data_culve1_test,2)
    temp(5:m+4,5:n+4,:)=data_culve1_test{i}(:,:,:);
    data_culve1_test_extend{i} = temp;
end
[traindata2_test,data_culve2_test]=layers_training_con_all(data_culve1_test_extend,W2,3);
%%数据扩充
for i=1:312
data_culve2_test_more{flag}=data_culve2_test{i};
flag=flag+1;
end
%% 正样本3-噪声2
path_testh = fullfile('K:','chenwei','programming','face_data','data_noise','yres_sam_l1_5_19_jitter1.mat');
load(path_testh);
% load('data_save\org\final.mat')
M=size(yres_sam,2);
[m,n,t,k]=size(yres_sam{1});
data_culve1_test{M}=[];
for i=1:M
    temp=zeros(m,n,k);
    for j=1:k
        temp(:,:,j)=yres_sam{i}(:,:,:,j);
    end
    data_culve1_test{i} = temp;
end
clear temp yres_sam;

[m,n,k]=size(data_culve1_test{1});
temp =zeros(m+8,n+8,k);
data_culve1_test_extend{size(data_culve1_test,2)}=[];
for i=1:size(data_culve1_test,2)
    temp(5:m+4,5:n+4,:)=data_culve1_test{i}(:,:,:);
    data_culve1_test_extend{i} = temp;
end
[traindata2_test,data_culve2_test]=layers_training_con_all(data_culve1_test_extend,W2,3);
%%数据扩充
for i=1:312
data_culve2_test_more{flag}=data_culve2_test{i};
flag=flag+1;
end

%% 正样本4-噪声3
path_testh = fullfile('K:','chenwei','programming','face_data','data_noise','yres_sam_l1_5_20_jitter2.mat');
load(path_testh);
% load('data_save\org\final.mat')
M=size(yres_sam,2);
[m,n,t,k]=size(yres_sam{1});
data_culve1_test{M}=[];
for i=1:M
    temp=zeros(m,n,k);
    for j=1:k
        temp(:,:,j)=yres_sam{i}(:,:,:,j);
    end
    data_culve1_test{i} = temp;
end
clear temp yres_sam;
[m,n,k]=size(data_culve1_test{1});
temp =zeros(m+8,n+8,k);
data_culve1_test_extend{size(data_culve1_test,2)}=[];
for i=1:size(data_culve1_test,2)
    temp(5:m+4,5:n+4,:)=data_culve1_test{i}(:,:,:);
    data_culve1_test_extend{i} = temp;
end

[traindata2_test,data_culve2_test]=layers_training_con_all(data_culve1_test_extend,W2,3);

%%数据扩充
for i=1:312
data_culve2_test_more{flag}=data_culve2_test{i};
flag=flag+1;
end
save 'data_save\all_con_W3_200\final_test_p' data_culve2_test_more;
%% negative data testing

path_test_n = fullfile('K:','chenwei','programming','face_data','yres_sam_l1_4_14_neg.mat');
load(path_test_n);
M=size(yres_sam,2);
[m,n,t,k]=size(yres_sam{1});
data_culve1_test_n{M}=[];
for i=1:size(yres_sam,2)
    temp=zeros(m,n,k);
    for j=1:k
        temp(:,:,j)=yres_sam{i}(:,:,:,j);
    end
    data_culve1_test_n{i} = temp;
end
clear temp yres_sam;
[m,n,k]=size(data_culve1_test_n{1});
temp =zeros(m+8,n+8,k);
data_culve1_test_n_extend{size(data_culve1_test_n,2)}=[];
for i=1:size(data_culve1_test_n,2)
    temp(5:m+4,5:n+4,:)=data_culve1_test_n{i}(:,:,:);
    data_culve1_test_n_extend{i}= temp;
end
clear temp m n k;
[traindata2_test_n,data_culve2_test_n]=layers_training_con_all(data_culve1_test_n_extend,W2,3);

save 'data_save\all_con_W3_200\final_test_n' data_culve2_test_n;
%% 正负样本R—P绘图
R_P(data_culve2_test_more,data_culve2_test_n,1);


%% 正样本的方位选择处理
flag1=1;flag2=1;
M=size(data_culve2_test_more,2);
data_p_1{M/3}=[];
data_n_1{M/3*2}=[];
for i=1:3:M
    data_p_1{flag1}=data_culve2_test_more{i};
    flag1=flag1+1;
    data_n_1{flag2}=data_culve2_test_more{i+1};
    flag2=flag2+1;
    data_n_1{flag2}=data_culve2_test_more{i+2};
    flag2=flag2+1;
end
flag1=1;flag2=1;
data_p_2{M/3}=[];
data_n_2{M/3*2}=[];
for i=2:3:M
    data_p_2{flag1}=data_culve2_test_more{i};
    flag1=flag1+1;
    data_n_2{flag2}=data_culve2_test_more{i-1};
    flag2=flag2+1;
    data_n_2{flag2}=data_culve2_test_more{i+1};
    flag2=flag2+1;
end
flag1=1;flag2=1;
data_p_3{M/3}=[];
data_n_3{M/3*2}=[];
for i=3:3:M
    data_p_3{flag1}=data_culve2_test_more{i};
    flag1=flag1+1;
    data_n_3{flag2}=data_culve2_test_more{i-2};
    flag2=flag2+1;
    data_n_3{flag2}=data_culve2_test_more{i-1};
    flag2=flag2+1;
end


R_P(data_p_1,data_n_1,2);
R_P(data_p_2,data_n_2,3);
R_P(data_p_3,data_n_3,4);

%%=====================================