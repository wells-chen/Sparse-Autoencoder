function  W2_display(W11,W2,maxpooling_size)
%used for display for the back computation of trainning data. compare original data to the data trainning result.
%setting the maxpooling_size : 2*2

%W2原始数据格式为 64*5*5 {1,N}个 W11 为25*64
[M1,N1] = size(W11);
%转化为要的数据格式 要求data格式为data{样本数} {1,i} 36个  {1,j}64个5*5）
M=size(W2,2);
[K,L,S]=size(W2{1});
W2_or_cell{M,S}=[];
for i=1:M
    for j=1:S
        W2_or_cell{i,j}= reshape(W2{i}(:,:,j),K,L);
    end
end
clear temp temp2;


% back for maxpooling
W2_max = zeros(K*maxpooling_size,L*maxpooling_size);
W2_max_cell{M,S} = [];
for i=1:M
    for j=1:S
        for k=1:K
            for l=1:L
                W2_max(k*maxpooling_size,l*maxpooling_size) = W2_or_cell{i,j}(k,l);
            end
        end
        W2_max_cell{i,j}=W2_max;
    end
end
clear W2_or_cell;

%译码器解码 线性叠加
filter_size = sqrt(size(W11,1));
W2_toshow = zeros((K*maxpooling_size+filter_size-1)*(L*maxpooling_size+filter_size-1),M);
%M第二层滤波器个数
path_name = fullfile('..','test_result','W2_W1_Display','\');
mkdir(path_name);
for i=1:M
    %N1 第一层滤波器个数
    W2_Line = zeros(K*maxpooling_size+filter_size-1,L*maxpooling_size+filter_size-1);
    for t=1:N1
        W2_temp = zeros(K*maxpooling_size+filter_size-1,L*maxpooling_size+filter_size-1);
        for k=1:K*maxpooling_size
            for l=1:L*maxpooling_size
                W2_temp(k:k+filter_size-1 ,l:l+filter_size-1) = W2_max_cell{i,t}(k,l)*(reshape(W11(:,t),filter_size ,filter_size))+W2_temp(k:k+filter_size-1 ,l:l+filter_size-1);
            end
        end
        W2_Line=W2_Line+W2_temp;
    end
    W2_toshow(:,i) = W2_Line(:);
    
end
figure,display_network(W2_toshow,12);
filename =['Filter','W2_Dispaly','.','jpg'];
saveas(gcf,[path_name,filename]);

% for i=1:M
%     for j=1:S
%         path_name = fullfile('..','test_result','data_back',num2str(ttt),'\');
%         mkdir(path_name);
%         filename =['Figure',num2str(j),'.','jpg'];
%         imwrite(data_original_cell{1,i}{1,j},[path_name,filename]);
%     end
% end
mkdir('data_save');
save 'data_save\W2_display';

end