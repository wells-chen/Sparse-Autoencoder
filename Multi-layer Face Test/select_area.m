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

flag_k=1;
for k=1:2:300
    flag=1;flag1=1;
    for i=1:2:50-10+1
        flag2=1;
        for j=1:2:50-10+1
            data_back{flag1,flag2} =data_culve1{k}(i:i+10-1,j:j+10-1,:);
            flag2=flag2+1;
        end
        flag1=flag1+1;
    end
    %区域1
    for i=5:8
        for j=4:8
            data_select{flag_k,flag} = data_back{i,j};
            flag=flag+1;
        end
    end
    %区域2
    for i=5:8
        for j=12:16
            data_select{flag_k,flag} = data_back{i,j};
            flag=flag+1;
        end
    end
    %区域3
    for i=11:16
        for j=9:14
            data_select{flag_k,flag} = data_back{i,j};
            flag=flag+1;
        end
    end
    flag_k=flag_k+1;
    clear data_back;
end
[Mt,Nt]=size(data_select);
[mt,nt,kt] = size(data_select{1,1});
visibleSize2 = mt*nt*kt;
hiddenSize2 = 225;
patchest = zeros(visibleSize2,Mt*Nt); k=1;
for i=1:Mt
    for j=1:Nt
        temp=data_select{i,j};
        patchest(:,k)=temp(:);
        k=k+1;
    end
end
clear temp;
save 'data_save\layerone_select';
% load('data_save\layerone_select.mat');
[optthetat, costt]=layer(visibleSize2, hiddenSize2,lambda, sparsityParam,beta, patchest);



W2_t= reshape(optthetat(1:hiddenSize2*visibleSize2), hiddenSize2, visibleSize2);
for i=1:hiddenSize2
    W2t{i} = reshape(W2_t(i,:),mt,nt,kt);
end
for i=1:hiddenSize2
    W2tt{i}=max(W2t{i},0);
end
  W2_display(W1',W2tt,4);
  W2=W2tt;
  [traindata2,data_culve2]=layers_training(data_culve1,W2,2);
  
  
  
% m是第一层滤波器数量，n,k是卷积后分割操作得到的patch大小
%输入数据数量为M2*N2
[M3,N3]=size(traindata2);
[m3,n3,k3] = size(traindata2{1,1});
visibleSize3 = m3*n3*k3;
hiddenSize3 = 100;
patches3 = zeros(visibleSize3,M3*N3);k=1;
for i=1:M3
    for j=1:N3
        temp=traindata2{i,j};
        patches3(:,k)=temp(:);
        k=k+1;
    end
end
clear temp;
save 'data_save\layertwo_0';
clearvars -except hiddenSize3 visibleSize3 sparsityParam lambda beta patches3;
theta3 = initializeParameters(hiddenSize3, visibleSize3);

[opttheta3, cost3]=layer(visibleSize3, hiddenSize3,lambda, sparsityParam,beta, patches3);
%% layer result
load('data_save\layertwo_0.mat');
W3_temp= reshape(opttheta3(1:hiddenSize3*visibleSize3), hiddenSize3, visibleSize3);
W3{hiddenSize3}=[];
for i=1:hiddenSize3
    W3{i} = reshape(W3_temp(i,:),m3,n3,k3);
end
% [traindata3,data_culve3]=layer_conv_max_contrast(data_culve2,W3);
W3_display(W1',W2,W3,4,2);
save 'data_save\final_0';
clearvars -except W1 W2 W3;
save 'W_value' W1 W2 W3;

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
save 'data_save\final_test_p' data_culve3_test_more;
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

save 'data_save\final_test_n' data_culve3_test_n;
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

R_P(data_p_2,data_n_2,3);

R_P(data_p_3,data_n_3,4);


  
  
