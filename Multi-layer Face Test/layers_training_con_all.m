function [data_result,data_result_conv]=layers_training_con_all(data,W_old,layer_i)
%������һ�ε���������Ԥ����----������Ԫ�����maxpooling��contrast normalization
%dataΪԭʼ����minist 40*40  
%data������λ����������ά�ȣ�����λ��������W_old������λ����ά�ȣ�����Ϊ��Ԫ����
%ѵ�����ݵ�����ά��Ԥ����Ϊ5*5
patches_dim = 10;
%maxpooling�����С Ԥ����2*2
maxpool_size =4;
%contrast normalization dimension Ԥ����Ϊ5*5
contrast_dim =4;

if layer_i==1
    [M,N]=size(data);
    [m,n] =size(W_old);%W ��ÿһ��Ϊһ����Ԫ
    %data��Ϊ���󣬴�Ϊ��ά
    data_dim = sqrt(M);
    data_temp{N} = [];
    for i=1:N
        data_temp{i}=reshape(data(:,i),data_dim,data_dim);
    end
    %W��Ϊ����
    W{n}=[];
    neuron_dim =sqrt(n);
    for i=1:n
        W{i}=reshape(W_old(:,i),neuron_dim,neuron_dim);
    end
    conv_temp = zeros(floor((data_dim-neuron_dim+1)/maxpool_size),floor((data_dim-neuron_dim+1)/maxpool_size),N);
    data_conv_temp{N}=[];
    for i=1:N
        for j=1:n
            %�������
            temp_conv = Conv_self(data_temp(i,k:k+neuron_dim-1,l:l+neuron_dim-1),W{j},2);
            %maxpooling
            conv_temp(:,:,j) = maxpooling(temp_conv,maxpool_size);
        end
        data_conv_temp{i}=conv_temp;
    end
    data_result_conv{n}=[];
    for i=1:N
        %contrast normalization
        %ÿһ��cell����һ����������ά���� ��36*36*64
        data_result_conv{i} = Normalization(data_conv_temp{i},contrast_dim);
    end
    %���ݷָ�
    %ÿһ��data_result_conv cell ����һ���������ݵľ�����
    data_result=Data_division(data_result_conv,patches_dim);    
    % i���������ݵ�������j����һ���˲�����������k*l�Ǿ�������γɵ�patch��С���ɷָ���������� �����λ�� �ָ����������ά��
end
if layer_i==2
      %ѵ�����ݵ�����ά��Ԥ����Ϊ5*5
    patches_dim = 7;
    %maxpooling�����С Ԥ����2*2
    maxpool_size =3;
    %contrast normalization dimension Ԥ����Ϊ5*5
    contrast_dim = 4;
    simple_num=size(data,2);
    conv_num = size(W_old,2);
    data_conv_temp{simple_num}=[];
    for i=1:simple_num
        i
        tic
        for j=1:conv_num
            %�������
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
    %���ݷָ�
%     for i=1:simple_num
%       data_result{i,1}=data_result_conv{i};
%     end
    %ÿһ��data_result_conv cell ����һ���������ݵľ�����
    data_result=Data_division(data_result_conv,patches_dim);
    % i���������ݵ�������j����һ���˲�����������k*l�Ǿ�������γɵ�patch��С���ɷָ���������� �����λ�� �ָ����������ά��
end


if layer_i==3
    %ѵ�����ݵ�����ά��Ԥ����Ϊ5*5
    patches_dim = 7;
    %maxpooling�����С Ԥ����2*2
    maxpool_size =2;
    %contrast normalization dimension Ԥ����Ϊ5*5
    contrast_dim = 4;
    simple_num=size(data,2);
    conv_num = size(W_old,2);
    data_conv_temp{simple_num}=[];
    data_result{simple_num} =[];
    temp_conv=zeros(conv_num,1);
    for i=1:simple_num
        i
        for j=1:conv_num
            %�������
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
    %���ݷָ�
    %ÿһ��data_result_conv cell ����һ���������ݵľ�����
%     data_result=Data_division(data_result_conv{i},patches_dim);
end

clearvars -except data_result data_result_conv ;



end