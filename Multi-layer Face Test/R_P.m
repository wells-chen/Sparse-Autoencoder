function R_P(data_p,data_n,fig_num)
%传入的参数为cell{i} M*N*num--神经元

M_p =size(data_p,2);

[m,n,k]=size(data_p{1});
n_neuron_x=sqrt(m*n);n_neuron_y=sqrt(m*n);
temp=zeros(1,k);
final_neu_pos=zeros(n_neuron_x ,n_neuron_y,M_p);
for i=1:M_p
%     for j=1:k
%        temp(j)=max(max(max(data_culve3_test_0{i}(:,:,j))),0);
%     end
    final_neu_pos(: ,:,i)=reshape(data_p{i},n_neuron_x,n_neuron_y);
end
clear data_p temp;

M_N =size(data_n,2);
[m,n,k]=size(data_n{1});
temp=zeros(1,k);
final_neu_neg=zeros(n_neuron_x,n_neuron_y,M_N);
for i=1:M_N
%      for j=1:k
%        temp(j)=max(max(max(data_culve3_test_n_0{i}(:,:,j))),0);
%     end
  final_neu_neg(:,:,i) = reshape(data_n{i},n_neuron_x,n_neuron_y);
end
clear data_culve3_test_n_0 temp;
%% ??????
max_value=max(max(final_neu_pos(:),max(final_neu_neg(:))));
final_neu_pos=max(final_neu_pos,0);
final_neu_neg=max(final_neu_neg,0);
final_neu_pos=final_neu_pos./max_value;
final_neu_neg =final_neu_neg./max_value;

final_neu(:,:,1:M_p)=final_neu_pos;
final_neu(:,:,M_p+1:M_p+M_N)=final_neu_neg(:,:,1:M_N);
% for nnn=1:1
max_neu=final_neu;num=M_p+M_N;
p_top=0.99;Pnum=M_p;
k=1;
clear recall
clear precision
for p_th=0.02:0.01:0.9
    p_th
    for th=0.01:0.02:1
        P=0;
        N=0;
        p_t_0=zeros(n_neuron_x,n_neuron_y);n_t_0=zeros(n_neuron_x,n_neuron_y);
        task_related_flag=zeros(n_neuron_x,n_neuron_y);
        for i=1:n_neuron_x
            for j=1:n_neuron_y 
                for t=1:num

                    if(t<=M_p && max_neu(i,j,t)>th)
                        p_t_0(i,j)=p_t_0(i,j)+1;
                    end
                    if(t>M_p && max_neu(i,j,t)>th)
                        n_t_0(i,j)=n_t_0(i,j)+1;
                    end


                end
                Ps(i,j)=p_t_0(i,j)/(p_t_0(i,j)+n_t_0(i,j));
                if(Ps(i,j)>p_th)
                    task_related_flag(i,j)=1;
                end
            end 
        end 
        clear judgev;
        for t=1:num
            judgev=max_neu(:,:,t).*task_related_flag;
            if(t<=M_p && max(max(judgev))>th)
                P=P+1;
            end
            if(t>M_p && max(max(judgev))>th)
                N=N+1;
            end
        end
        precision(k)=P/(P+N+1e-10);
        recall(k)=P/(Pnum);
        k=k+1;
    end
end
for i=1:k-1
    for j=1:k-1        
        if(precision(i)<precision(j) && recall(i)<recall(j))
            precision(i)=0;recall(i)=0;
            break;
        end
        
    end
end
save 'data_p_n' precision recall;
figure;scatter(recall,precision,'filled','blue')
grid on
path_name = fullfile('..','test_result','P_R','\');
mkdir(path_name);
filename =['Filter',num2str(fig_num),'.','jpg'];
saveas(gcf,[path_name,filename]);


end