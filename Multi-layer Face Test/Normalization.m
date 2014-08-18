function result = Normalization(data,dimension)
%dimension �������ɸ�˹�����ά�ȣ�Ҳ�ǿ��ƶԱȹ�һ�� ���ֵ�patch��С��
%data���ݸ�ʽi j k  ; iΪ���ݸ��� ,j*kΪ������Ԫ�����ݷ���
u = dimension/2;
[M,N,K]=size(data);
dim_add =floor(dimension/2);
%��չdata��ά�ȣ�һʵ�ֶԱȹ�һ�������ά����ԭ��ά��һ����
data_temp = zeros(M+dim_add*2,N+dim_add*2,K);
for i=1:K
    for j=1:M
        for k=1:N
            data_temp(j+dim_add,k+dim_add,i) = data(j,k,i);
        end
    end
end
%���ɸ�˹���󣬲�����Լ����ʹ��ֵΪ1
[X,Y] = meshgrid(1:dimension,1:dimension);
W = exp(-(X-u).^2-(Y-u).^2);
W = W./(sum(W(:))*K);
%�����׼��ı���
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
%�������ڿ��Ʒ�ĸ��ȫ�ֱ�׼��ľ�ֵ
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