%%%%%%%%%%%%%%%%%%%%%    Pose estimation %%%%%%%%%%%%%%%%%%%%%%%%%%
% function pose_estimation()
% clear;
% load neu_l3_4_9;
% load yres_sam_l2_4_9_test;
% yres_sam_pos=yres_sam_l;
% % load yres_sam_l2_4_9_train;
% % yres_sam_neg=yres_sam_l;
% num=size(yres_sam_l,2);
% % max_map=zeros(n_neuron_x,n_neuron_y);
% num_error=num;
% error_y=zeros(1,num_error);
% lnum=size(yres_sam_pos{1},2);
% % neuin{1,1}=neu{2,3};neuin{1,2}=neu{3,3};neuin{1,3}=neu{2,2};neuin{1,4}=neu{1,1};neuin{1,5}=neu{3,1};
% R=2;
% 
% [n_neuron_x,n_neuron_y]=size(neu);
% for t=1:1:num
%     
%     [siz_n,siz_m,n_neuron_x_l,n_neuron_y_l]=size(yres_sam_pos{t}{1});
%     im_t=zeros(siz_n+2*R,siz_m+2*R,n_neuron_x_l,n_neuron_y_l);
%     
%     for l=1:lnum
%         
%         im_t(R+1:siz_n+R,R+1:siz_m+R,:,:)=yres_sam_pos{t}{l};
% 
%         [y,yorg,yvar,ysum,Ynoliner]=response_pal(im_t,neu,'non_sum');
%         [sx,sy]=size(ysum);
%         for i=1:n_neuron_x
%             for j=1:n_neuron_y
%                 
%                 max_neul(i,j,l)=max(max(yorg(:,:,i,j)));
%                 
%             end
%         end
%         
%         
%     end
%     
%      for i=1:n_neuron_x
%         for j=1:n_neuron_y  
%             max_neu(i,j,t)=max(max_neul(i,j,:));
%         end 
%      end
%      t
% end
% for p=1:6
%     for q=1:6
%  k=0;
%  for i=1:3000
% k=k+1;if(mod(i,10)==6)continue;end; bbb(k)=final_neu(p,q,i);
% end
% k=0;
% for i=6:10:3000
% k=k+1;aaa(k)=final_neu(p,q,i);
% end
% figure; plot(hist(aaa,[0:0.01:1]),'red');hold on; plot(hist(bbb,[0:0.01:1]));
%     end
% end
clc;
clear;
load('data_save\final_test_p.mat');
clearvars -except data_culve3_test;
load('data_save\final_test_n.mat');
clearvars -except data_culve3_test data_culve3_test_n;
M_p =size(data_culve3_test,2);
n_neuron_x=7;n_neuron_y=7;
[m,n,k]=size(data_culve3_test{1});
temp=zeros(1,k);
final_neu_pos=zeros(n_neuron_x ,n_neuron_y,M_p);
for i=1:M_p
%     for j=1:k
%        temp(j)=max(max(max(data_culve3_test{i}(:,:,j))),0);
%     end
    final_neu_pos(: ,:,i)=reshape(data_culve3_test{i},n_neuron_x,n_neuron_y);
end
clear data_culve3_test temp;

M_N =size(data_culve3_test_n,2);
[m,n,k]=size(data_culve3_test_n{1});
temp=zeros(1,k);
final_neu_neg=zeros(n_neuron_x,n_neuron_y,M_N);
for i=1:M_N
%      for j=1:k
%        temp(j)=max(max(max(data_culve3_test_n{i}(:,:,j))),0);
%     end
  final_neu_neg(:,:,i) = reshape(data_culve3_test_n{i},n_neuron_x,n_neuron_y);
end
clear data_culve3_test_n temp;
%% πÈ“ªªØ
max_value=max(max(final_neu_pos(:),max(final_neu_neg(:))));
final_neu_pos=final_neu_pos./max_value;
final_neu_neg =final_neu_neg./max_value;

% load('Dis')
% load('final_neu_5_8_pos');
% final_neu_pos=final_neu;
% load('final_neu_5_8_neg');
% final_neu_neg=final_neu;
% clear final_neu;
% final_neu=zeros(n_neuron_x,n_neuron_y,3312);

% nn=0;
% for t=1:312
%     if(max(max(final_neu(:,:,t)))<0.3)
%         final_neu(:,:,t)=0;
%         nn=nn+1;
% %         Dis(1,:,t)=0;
%         imm{t}=imread(['E:\body\visualization\',num2str(t),'.jpg']);
%         imwrite(imm{t},['E:\body\visualization3\',num2str(t),'.jpg']);
%     end
% end
final_neu(:,:,1:312)=final_neu_pos;
final_neu(:,:,313:3312)=final_neu_neg;
% for nnn=1:1
max_neu=final_neu;num=3312;
p_top=0.99;Pnum=312;
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

                    if(t<=312 && max_neu(i,j,t)>th)
                        p_t_0(i,j)=p_t_0(i,j)+1;
                    end
                    if(t>312 && max_neu(i,j,t)>th)
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
            if(t<=312 && max(max(judgev))>th)
                P=P+1;
            end
            if(t>312 && max(max(judgev))>th)
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
figure;scatter(recall,precision,'filled','blue')
grid on

max_neu=Dis;n_neuron_x=1;n_neuron_y=1;num=1600;
p_top=0.99;Pnum=600-nn;
k=1;
clear recall
clear precision
for p_th=0.4:0.01:0.9
    for th=0.1:0.02:1
        P=0;
        N=0;
        p_t_0=zeros(n_neuron_x,n_neuron_y);n_t_0=zeros(n_neuron_x,n_neuron_y);
        task_related_flag=zeros(n_neuron_x,n_neuron_y);
        for i=1:n_neuron_x
            for j=1:n_neuron_y 
                for t=1:num

                    if(t<=600 && max_neu(i,j,t)>th)
                        p_t_0(i,j)=p_t_0(i,j)+1;
                    end
                    if(t>600 && max_neu(i,j,t)>th)
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
            if(t<=600 && max(max(judgev))>th)
                P=P+1;
            end
            if(t>600 && max(max(judgev))>th)
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

hold ;scatter(recall,precision,'filled','red')

% end