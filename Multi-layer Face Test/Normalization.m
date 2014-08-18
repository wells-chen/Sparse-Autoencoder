function result = Normalization(data,dimension)
%dimension 用于生成高斯方阵的维度，也是控制对比归一化 划分的patch大小。
%data数据格式i j k  ; i为数据个数 ,j*k为单个单元的数据方阵
u = dimension/2;
[M,N,K]=size(data);
dim_add =floor(dimension/2);
%扩展data的维度，一实现对比归一化后矩阵维度与原有维度一样。
data_temp = zeros(M+dim_add*2,N+dim_add*2,K);
for i=1:K
    for j=1:M
        for k=1:N
            data_temp(j+dim_add,k+dim_add,i) = data(j,k,i);
        end
    end
end
%生成高斯矩阵，并进行约束，使和值为1
[X,Y] = meshgrid(1:dimension,1:dimension);
W = exp(-(X-u).^2-(Y-u).^2);
W = W./(sum(W(:))*K);
%定义标准差的变量
temp_count=zeros(M,N);
for j=1:M
    for k=1:N
        temp =0;
        for i=1:K        
            temp=temp+sum(sum(W.*data_temp(j:j+dimension-1,k:k+dimension-1,i)));
        end
        temp_count(j,k)=temp;
    end
end
clear temp;
for i=1:K
    for j=1:M
        for k=1:N
            %subtractive
            data_temp(j+dim_add,k+dim_add,i) = data_temp(j+dim_add,k+dim_add,i)-temp_count(j,k);
        end
    end
end
for j=1:M
    for k=1:N
        temp=0;
        for i=1:K    
            temp0 = data_temp(j:j+dimension-1,k:k+dimension-1,i).^2;
            temp=temp+sum(sum((W.*temp0)));
        end
        temp_count(j,k)=sqrt(temp);
    end
end
clear temp1 temp2 temp3 temp;
%计算用于控制分母的全局标准差的均值
mean_var = mean(temp_count(:));
for i=1:K
    for j=1:M
        for k=1:N
            if max(mean_var, temp_count(j,k))==0
               data(j,k,i)=data_temp(j+dim_add,k+dim_add,i)/10^-10;
            else
                 data(j,k,i)=data_temp(j+dim_add,k+dim_add,i)/max(mean_var, temp_count(j,k));
        end
    end
end
result= data;

end