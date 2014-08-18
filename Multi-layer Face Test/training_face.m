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
%数据分割
patches_dim=10;

traindata1=Data_division(data_culve1,patches_dim);

% m是第一层滤波器数量，n,k是卷积后分割操作得到的patch大小
%输入数据数量为M2*N2
[M2,N2]=size(traindata1);
[m2,n2,k2] = size(traindata1{1,1});
visibleSize2 = m2*n2*k2;
hiddenSize2 = 225;
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
mkdir('data_save\org_W3_200\');
save 'data_save\org_W3_200\layerone';
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
options.maxIter = 300;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';

[opttheta2, cost2] = minFunc( @(p) sparseAutoencoderCost(p, ...
    visibleSize2, hiddenSize2, ...
    lambda, sparsityParam, ...
    beta, patches2), ...
    theta2, options);

% [opttheta2, cost2]=layer(visibleSize2, hiddenSize2,lambda, sparsityParam,beta, patches2);
%% layer2
load('data_save\org_W3_200\layerone.mat');
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

W2{hiddenSize2}=[];
for i=1:hiddenSize2
    W2{i} = reshape(W2_temp(i,:),m2,n2,k2);
end
% for i=1:hiddenSize2
%     W2{i}=max(W2{i},0);
% end

%
% flag=1;
% for i=1:2:50-10+1
%     for j=1:2:50-10+1
%         Wat{flag} =data_culve1{4}(i:i+10-1,j:j+10-1,:);
%         flag=flag+1;
%     end
% end
% for i=1:flag-1
%     W2_replace{i}=max(Wat{i},0);
% end
% figure,W2_display(W1',W2_replace);
% for i=1:3
%     W2_replace{i}=data_back{118+i};
% end
save 'data_save\org_W3_200\W2';
% load('data_save\W2.mat');
W2_display(W1',W2,4);
% data{1}=data_culve1{1};
[traindata2,data_culve2]=layers_training(data_culve1,W2,2);
% m是第一层滤波器数量，n,k是卷积后分割操作得到的patch大小
%输入数据数量为M2*N2
[M3,N3]=size(traindata2);
[m3,n3,k3] = size(traindata2{1,1});
visibleSize3 = m3*n3*k3;
hiddenSize3 = 49;
patches3 = zeros(visibleSize3,M3*N3);
for i=1:M3
    for j=1:N3
        temp=traindata2{i,j};
        patches3(:,i)=temp(:);
    end
end
clear temp;
save 'data_save\org_W3_200\layertwo';
% load('data_save\layertwo.mat');
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
options.maxIter = 300;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';

[opttheta3, cost3] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize3, hiddenSize3, ...
                                   lambda, sparsityParam, ...
                                   beta, patches3), ...
                              theta3, options);
% theta3=opttheta3;
% % end
% [opttheta3, cost3]=layer(visibleSize3, hiddenSize3,lambda, sparsityParam,beta, patches3);
%% layer result
load('data_save\org_W3_200\layertwo.mat');
W3_temp= reshape(opttheta3(1:hiddenSize3*visibleSize3), hiddenSize3, visibleSize3);
W3{hiddenSize3}=[];
for i=1:hiddenSize3
    W3{i} = reshape(W3_temp(i,:),m3,n3,k3);
end
% [traindata3,data_culve3]=layer_conv_max_contrast(data_culve2,W3);

% save 'data_save\org_noise\final';
save ‘W_value’ W1 W2 W3;
clearvars -except W1 W2 W3;


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
% patches1_test = Im_Input(path_test);
% [traindata1_test,data_culve1_test]=layer_conv_max_contrast(data,W1,1);
[traindata2_test,data_culve2_test]=layers_training(data_culve1_test,W2,2);
%%正样本1最后层扩充
[m,n,k]=size(data_culve2_test{1});
temp =zeros(m+8,n+8,k);
data_culve2_test_extend{size(data_culve2_test,2)}=[];
for i=1:size(data_culve2_test,2)
    tem(5:m+4,5:n+4,:)=data_culve2_test{i}(:,:,:);
    data_culve2_test_extend{i} = tem;
end
clear temp m n k;
[traindata3_test,data_culve3_test]=layers_training(data_culve2_test_extend,W3,3);
%%数据扩充
flag=1;
for i=1:312
data_culve3_test_more{flag}=data_culve3_test{i};
flag=flag+1;
end

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
% patches1_test = Im_Input(path_test);
% [traindata1_test,data_culve1_test]=layer_conv_max_contrast(data,W1,1);
[traindata2_test,data_culve2_test]=layers_training(data_culve1_test,W2,2);
%%正样本2最后层扩充
[m,n,k]=size(data_culve2_test{1});
temp =zeros(m+8,n+8,k);
data_culve2_test_extend{size(data_culve2_test,2)}=[];
for i=1:size(data_culve2_test,2)
    tem(5:m+4,5:n+4,:)=data_culve2_test{i}(:,:,:);
    data_culve2_test_extend{i} = tem;
end
[traindata3_test,data_culve3_test]=layers_training(data_culve2_test_extend,W3,3);
% [R,P,opt_location] = Precision_conv(data_culve3_test);
% clearvars -except W1 W2 W3;
%%数据扩充
for i=1:312
data_culve3_test_more{flag}=data_culve3_test{i};
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
% patches1_test = Im_Input(path_test);
% [traindata1_test,data_culve1_test]=layer_conv_max_contrast(data,W1,1);
[traindata2_test,data_culve2_test]=layers_training(data_culve1_test,W2,2);
%%正样本3最后层扩充
[m,n,k]=size(data_culve2_test{1});
temp =zeros(m+8,n+8,k);
data_culve2_test_extend{size(data_culve2_test,2)}=[];
for i=1:size(data_culve2_test,2)
    tem(5:m+4,5:n+4,:)=data_culve2_test{i}(:,:,:);
    data_culve2_test_extend{i} = tem;
end
[traindata3_test,data_culve3_test]=layers_training(data_culve2_test_extend,W3,3);
% [R,P,opt_location] = Precision_conv(data_culve3_test);
% clearvars -except W1 W2 W3;
%%数据扩充
for i=1:312
data_culve3_test_more{flag}=data_culve3_test{i};
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
% patches1_test = Im_Input(path_test);
% [traindata1_test,data_culve1_test]=layer_conv_max_contrast(data,W1,1);
[traindata2_test,data_culve2_test]=layers_training(data_culve1_test,W2,2);
%%正样本4最后层扩充
[m,n,k]=size(data_culve2_test{1});
temp =zeros(m+8,n+8,k);
data_culve2_test_extend{size(data_culve2_test,2)}=[];
for i=1:size(data_culve2_test,2)
    tem(5:m+4,5:n+4,:)=data_culve2_test{i}(:,:,:);
    data_culve2_test_extend{i} = tem;
end
[traindata3_test,data_culve3_test]=layers_training(data_culve2_test_extend,W3,3);
% [R,P,opt_location] = Precision_conv(data_culve3_test);
% clearvars -except W1 W2 W3;
%%数据扩充
for i=1:312
data_culve3_test_more{flag}=data_culve3_test{i};
flag=flag+1;
end
save 'data_save\org_W3_200\final_test_p' data_culve3_test_more;
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
% save ('W2_tem','W2')
% patches1_test = Im_Input(path_test);
% [traindata1_test,data_culve1_test]=layer_conv_max_contrast(data,W1,1);
[traindata2_test_n,data_culve2_test_n]=layers_training(data_culve1_test_n,W2,2);
%% 负样本最后层的扩充
[m,n,k]=size(data_culve2_test_n{1});
temp =zeros(m+8,n+8,k);
data_culve2_test_n_extend{size(data_culve2_test_n,2)}=[];
for i=1:size(data_culve2_test_n,2)
    tem(5:m+4,5:n+4,:)=data_culve2_test_n{i}(:,:,:);
    data_culve2_test_n_extend{i}= tem;
end
clear temp m n k;

[traindata3_test_n,data_culve3_test_n]=layers_training(data_culve2_test_n_extend,W3,3);
% R_P(data_culve3_test,data_culve3_test_n,1);
% save ('data_save\org_noise_extend\final_test_n',data_culve3_test_n);
% mkdir('data_save\org_noise_extend\');

% save 'data_save\org_noise_extend\final_test_n';
% load('data_save\org_noise\final_test_p');

save 'data_save\org_W3_200\final_test_n' data_culve3_test_n;
%% 正负样本R—P绘图
R_P(data_culve3_test_more,data_culve3_test_n,1);


%% 正样本的方位选择处理
flag1=1;flag2=1;
M=size(data_culve3_test_more,2);
data_p_1{M/3}=[];
data_n_1{M/3*2}=[];
for i=1:3:M
    data_p_1{flag1}=data_culve3_test_more{i};
    flag1=flag1+1;
    data_n_1{flag2}=data_culve3_test_more{i+1};
    flag2=flag2+1;
    data_n_1{flag2}=data_culve3_test_more{i+2};
    flag2=flag2+1;
end
flag1=1;flag2=1;
data_p_2{M/3}=[];
data_n_2{M/3*2}=[];
for i=2:3:M
    data_p_2{flag1}=data_culve3_test_more{i};
    flag1=flag1+1;
    data_n_2{flag2}=data_culve3_test_more{i-1};
    flag2=flag2+1;
    data_n_2{flag2}=data_culve3_test_more{i+1};
    flag2=flag2+1;
end
flag1=1;flag2=1;
data_p_3{M/3}=[];
data_n_3{M/3*2}=[];
for i=3:3:M
    data_p_3{flag1}=data_culve3_test_more{i};
    flag1=flag1+1;
    data_n_3{flag2}=data_culve3_test_more{i-2};
    flag2=flag2+1;
    data_n_3{flag2}=data_culve3_test_more{i-1};
    flag2=flag2+1;
end





R_P(data_p_1,data_n_1,2);

% for i=1:300
%     data_temp{i}=data_n_2{i};
% end
R_P(data_p_2,data_n_2,3);



R_P(data_p_3,data_n_3,4);

% for i=1:312
%     temp=final_neu(:,:,i);
%     data_ttt_p{i}=temp(:);
% end
% 
% data_culve3_test=data_ttt_p;


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