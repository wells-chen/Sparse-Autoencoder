function [R,P,opt_location] = Precision(training_data) 
%evaluation  the training data
[M,N]=size(training_data);
 Xmax = max(training_data(:));
 Xmin = min(training_data(:));
training_data = (training_data-Xmin)./(Xmax-Xmin);
negative_sample{10}=[];
positive_sample{10}=[];
for i=1:10 
    k=1;
    negative_sample{i} = training_data;
    for j=i:10:N
        positive_sample{i}(:,k)=training_data(:,j);
        negative_sample{i}(:,k)=[];
        k = k+ 1;
    end
end
R=0;
opt_location.x=1;
opt.location.y=1;
presion_temp=zeros(M,100);

 
for i= 1:10
    for j=1:M
        
        data_temp=positive_sample{i}(j,:);
        mun_count{j} = hist(floor(data_temp*100),0:1:100);
        max_positive = max(mun_count{j});
        data_temp_negative=negative_sample{i}(j,:);
        mun_count_negative{j} = hist(floor(data_temp_negative*100),0:1:100);
        max_negative = max(mun_count_negative{j});
        Y_max = max_negative+10;
        if max_positive>max_negative
            Y_max = max_positive+10;
        end
        scrsz = get(0,'ScreenSize');
        figure1=figure('Position',[0 30 scrsz(3) scrsz(4)-95]);
        bar(mun_count_negative{j},'g');
        hold on
        %         precision_rate = length(find((data_temp*100>=limit_value)))/(M*N/10);
        %         text(10,60,['greater than ',num2str(limit_value), ' is ',num2str(precision_rate)]);
        %         text(10,50,['greater than ',num2str(limit_value), ' is ',num2str(length(find((data_temp*100>=limit_value))))]);
        %         text(10,40,['smaller than ',num2str(limit_value), ' is ',num2str(length(find((data_temp*100<limit_value))))]);
        
        bar(mun_count{j},'b');
        set(gca,'xlim',[0 100],'ylim',[0 Y_max]);
        grid on;
        title(['Figure',num2str(j)]);
        xlabel('Response Value');
        ylabel('Count for neure');
%         path_name = fullfile('D:','algorithm_learning','sparseae_exercise','contrast','starter','test_result','testone_histogram',num2str(i),'\');
        path_name = fullfile('..','test_result','testone_histogram',num2str(i),'\')
        mkdir(path_name);
        filename =['Figure',num2str(j),'.','jpg'];
        saveas(gcf,[path_name,filename]);
        close all;
        %         limit_value = 70;
        %         precision_rate_negative = length(find((data_temp_negative*100>=limit_value)))/(M*N/10);
        %         text(10,60,['negative greater than ',num2str(limit_value), ' is ',num2str(precision_rate_negative)]);
        %         text(10,50,['negative greater than ',num2str(limit_value), ' is ',num2str(length(find((data_temp_negative*100>=limit_value))))]);
        %         text(10,40,['negative smaller than ',num2str(limit_value), ' is ',num2str(length(find((data_temp_negative*100<limit_value))))]);
        
        scrsz = get(0,'ScreenSize');
        figure1=figure('Position',[0 30 scrsz(3) scrsz(4)-95]);
        title(['Figure',num2str(j)]);
        plot(mun_count{j},'Color','g' );
        hold on;
        plot(mun_count_negative{j},'Color','b');
        hold off;
        xlabel('Response Value');
        ylabel('Count for neure');
        set(gca,'xlim',[0 100],'ylim',[0 Y_max]);
        text(10,Y_max/2,'Gree is Positive Sample');
        text(10,Y_max/2-5,'Blue is Negative Sample');
%         path_name = fullfile('D:','algorithm_learning','sparseae_exercise','contrast','starter','test_result','testone',num2str(i),'\');
        path_name = fullfile('..','test_result','testone',num2str(i),'\')
        mkdir(path_name);
        filename =['Figure',num2str(j),'.','jpg'];
        saveas(gcf,[path_name,filename]);
        close all;
        for k=1:100
            presion_temp(j,k)=length(find((data_temp*100>=k)))/(length(find((data_temp_negative*100>=k)))+length(find((data_temp*100>=k))));
            if(presion_temp(j,k)>R)
                R=presion_temp(j,k);
                opt_location.x=j;
                opt.location.y=k;
            end
        end
    end
    surf(presion_temp);
    path_name = fullfile('..','test_result','testone_presion\');
    mkdir(path_name);
    filename =['Figure',num2str(j),'.','jpg'];
    saveas(gcf,[path_name,filename]);
    close all;
end




% for i=1:10 
%     k=1;
%     for j=i:10:N
%         value{i}(:,k)=training_data(:,j);
%         k = k+ 1;
%     end
% end
% n




% for i=1:10 
%     Xmax = max(value{i}(:));
%     Xmin = min(value{i}(:));
%     value{i}=(value{i}-Xmin)./(Xmax-Xmin);
% %     mun_count{i} = hist(value{i}(:)*100,100);
%     mun_count{i} = hist(floor(value{i}(:)*100),0:1:100);
%     figure,bar(mun_count{i});
%     set(gca,'xlim',[0 100],'ylim',[0 300]);
%     grid on;
%     title(['Figure',num2str(i)]);
%     xlabel('Response Value');
%     ylabel('Count for neure');
%     limit_value = 70;
%     precision_rate = length(find((value{i}(:)*100>=limit_value)))/(M*N/10);
%     text(10,150,['greater than ',num2str(limit_value), ' is ',num2str(precision_rate)]);
% end

% for Single sample respnose  
% for i=1:10 
%     Xmax = max(value{i}(:));
%     Xmin = min(value{i}(:));
%     value{i}=(value{i}-Xmin)./(Xmax-Xmin);
%     for j=1:150
%         mun_count{i} = hist(floor(value{i}(:,j)*100),0:1:100);
% %         mun_count{i} = hist(value{i}(:,j)*100,100);
%         figure,bar(mun_count{i});
%         set(gca,'xlim',[0 100],'ylim',[0 10]);
%         grid on;
%         title(['Figure',num2str(j)]);
%         ylabel('Number of Neure');
%         xlabel('Response Value ');
%         limit_value = 70;
%         text(50,5,['greater than ',num2str(limit_value), ' is ',num2str(length(find((value{i}(:,j)*100>=limit_value))))]); 
%     end
% end
P=presion_temp;


    
end