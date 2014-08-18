function data_original_cell = original_data_display(W11,data_in,simple_num,filter_num,ttt)
%used for display for the back computation of trainning data. compare original data to the data trainning result.
%setting the maxpooling_size : 2*2
%原始数据格式为 64*5*5 *N个

maxpooling_size = 2;

%转化为要的数据格式 要求data格式为data{样本数}（三维 64*5*5）
[m,n]=size(data_in);
M=size(data_in,2);
N=M/simple_num;
data{1,simple_num}{1,N}=[];
num_flag =1;
temp_size = sqrt(m/filter_num);
for i=1:simple_num
    for j=1:N
        temp = data_in(:,num_flag);
        data{1,i}{1,j} = reshape(temp,filter_num,temp_size ,temp_size );
        num_flag=num_flag+1;
    end
end
clear temp num_flag data_in temp_size;

M=simple_num;
[S,K,L]=size(data{1,1}{1,1});
former_size = sqrt(N)+L-1;
former_data = zeros(S,former_size ,former_size);
former_data_cell{1,M} = [];

%orgainize the minidata to be a original data patches.
for i=1:M
    for j=1:N
        for k=1:former_size-K+1
            for l=1:former_size-L+1
                former_data(:,k:k+K-1,l:l+L-1) = data{1,i}{1,j};
            end
        end
    end
    former_data_cell{1,i} = former_data;
end

% back for maxpooling
data_max = zeros(former_size*maxpooling_size,former_size*maxpooling_size);
data_max_cell{1,M}{1,S} = [];
for i=1:M
    former_data = former_data_cell{1,i};
    for j=1:S
        for k=1:former_size
            for l=1:former_size
                data_max(k*maxpooling_size,l*maxpooling_size) = former_data(j,k,l);
            end
        end
        data_max_cell{1,i}{1,j}=data_max;
    end
end
%译码器解码
filter_size = sqrt(size(W11,1));
data_original_cell{1,M}{1,S}=[];
data_toshow = zeros((former_size*maxpooling_size+filter_size-1)*(former_size*maxpooling_size+filter_size-1),S);
for i=1:M
    path_name = fullfile('..','test_result','data_back','\');
    mkdir(path_name);
    for j=1:S
        data_original = zeros(former_size*maxpooling_size+filter_size-1,former_size*maxpooling_size+filter_size-1);
        for k=1:former_size*maxpooling_size
            for l=1:former_size*maxpooling_size
                data_original(k:k+filter_size-1 ,l:l+filter_size-1) = data_max_cell{1,i}{1,j}(k,l)*(reshape(W11(:,j),filter_size ,filter_size ))+data_original(k:k+filter_size-1 ,l:l+filter_size-1);
            end
        end
        data_original_cell{1,i}{1,j}=data_original;
        data_toshow(:,j) = data_original(:);
    end
    display_network(data_toshow,12);
    filename =['Figure',num2str(ttt),'.','jpg'];
    saveas(gcf,[path_name,filename]);
    close all;
end
% for i=1:M
%     for j=1:S
%         path_name = fullfile('..','test_result','data_back',num2str(ttt),'\');
%         mkdir(path_name);
%         filename =['Figure',num2str(j),'.','jpg'];
%         imwrite(data_original_cell{1,i}{1,j},[path_name,filename]);
%     end
% end


end