function [data_result,data_result_conv]=layers_training_con_all(data,W_old,layer_i)
%用于下一次迭代的数据预处理----处理神经元卷积，maxpooling，contrast normalization
%data为原始数据minist 40*40  
%data的行数位单样本数据维度，列数位样本数；W_old的列数位数据维度，行数为神经元数。
%训练数据的输入维度预定义为5*5
patches_dim = 10;
%maxpooling矩阵大小 预定义2*2
maxpool_size =4;
%contrast normalization dimension 预定义为5*5
contrast_dim =4;

if layer_i==1
    [M,N]=size(data);
    [m,n] =size(W_old);%W 的每一列为一个神经元
    %data变为方阵，此为二维
    data_dim = sqrt(M);
    data_temp{N} = [];
    for i=1:N
        data_temp{i}=reshape(data(:,i),data_dim,data_dim);
    end
    %W变为方阵
    W{n}=[];
    neuron_dim =sqrt(n);
    for i=1:n
        W{i}=reshape(W_old(:,i),neuron_dim,neuron_dim);
    end
    conv_temp = zeros(floor((data_dim-neuron_dim+1)/maxpool_size),floor((data_dim-neuron_dim+1)/maxpool_size),N);
    data_conv_temp{N}=[];
    for i=1:N
        for j=1:n
            %卷积操作
            temp_conv = Conv_self(data_temp(i,k:k+neuron_dim-1,l:l+neuron_dim-1),W{j},2);
            %maxpooling
            conv_temp(:,:,j) = maxpooling(temp_conv,maxpool_size);
        end
        data_conv_temp{i}=conv_temp;
    end
    data_result_conv{n}=[];
    for i=1:N
        %contrast normalization
        %每一个cell都是一个样本的三维矩阵 如36*36*64
        data_result_conv{i} = Normalization(data_conv_temp{i},contrast_dim);
    end
    %数据分割
    %每一个data_result_conv cell 都是一样样本数据的卷积结果
    data_result=Data_division(data_result_conv,patches_dim);    
    % i是输入数据的数量，j是上一层滤波器的数量，k*l是卷积操作形成的patch大小，可分割的行列数， 最后两位是 分割出来的行列维度
end
if layer_i==2
      %训练数据的输入维度预定义为5*5
    patches_dim = 7;
    %maxpooling矩阵大小 预定义2*2
    maxpool_size =3;
    %contrast normalization dimension 预定义为5*5
    contrast_dim = 4;
    simple_num=size(data,2);
    conv_num = size(W_old,2);
    data_conv_temp{simple_num}=[];
    for i=1:simple_num
        i
        tic
        for j=1:conv_num
            %卷积操作
%            temp_conv= Conv_self(data{i},W_old{j},3);
            temp_conv = Conv_self(data{i},W_old{j},3);
            %maxpooling
            conv_temp(:,:,j) = maxpooling(temp_conv,maxpool_size);
        end
        data_conv_temp{i} = conv_temp; 
        toc
        
        
    end
    data_result_conv{simple_num}=[];
    for i=1:simple_num
        i
        %contrast normalization
        data_result_conv{i} = Normalization(data_conv_temp{i},contrast_dim);
    end
    %数据分割
%     for i=1:simple_num
%       data_result{i,1}=data_result_conv{i};
%     end
    %每一个data_result_conv cell 都是一样样本数据的卷积结果
    data_result=Data_division(data_result_conv,patches_dim);
    % i是输入数据的数量，j是上一层滤波器的数量，k*l是卷积操作形成的patch大小，可分割的行列数， 最后两位是 分割出来的行列维度
end


if layer_i==3
    %训练数据的输入维度预定义为5*5
    patches_dim = 7;
    %maxpooling矩阵大小 预定义2*2
    maxpool_size =2;
    %contrast normalization dimension 预定义为5*5
    contrast_dim = 4;
    simple_num=size(data,2);
    conv_num = size(W_old,2);
    data_conv_temp{simple_num}=[];
    data_result{simple_num} =[];
    temp_conv=zeros(conv_num,1);
    for i=1:simple_num
        i
        for j=1:conv_num
            %卷积操作
            temp=Conv_self(data{i},W_old{j},3);
            temp_conv(j) = max(max(temp));
%             maxpooling
%              conv_temp(j,:,:) = maxpooling(temp_conv,maxpool_size);
             conv{i}(:,:,j)=temp;
        end
        data_result_conv{i}=temp_conv;
        data_result{i} = temp_conv;
    end
    save 'conv_data_4' conv data_result_conv;
%     data_result_conv{simple_num}=[];
%     for i=1:simple_num
%         %contrast normalization
%         data_result_conv{i} = Normalization(data_conv_temp{i},contrast_dim);
%     end
    %数据分割
    %每一个data_result_conv cell 都是一样样本数据的卷积结果
%     data_result=Data_division(data_result_conv{i},patches_dim);
end

clearvars -except data_result data_result_conv ;



end