clear data_temp data_select min_value max_value data_p_2_temp;
[m,n]=size(data_p_2);
 min_value = min(min(data_p_2{1}));
 max_value =max(max(data_p_2{1}));
for i=2:n
    if max_value<max(max(data_p_2{i}))
        max_value =max(max(data_p_2{i}));
    end
    if min_value > min(min(data_p_2{i}))
    min_value = min(min(data_p_2{i}));
    end
end
for i=1:n
data_p_2_temp{i}=(data_p_2{i}-min_value)./(max_value-min_value);
end
flag =1 ;
for i=1:n
    if max(data_p_2_temp{i}(:))>0.79
        data_select{flag}=data_p_2{i};
        flag=flag+1;
    end
end
R_P(data_select,data_n_2,3);
i=100;
temp_p=final_neu_pos(:,:,i);
temp_n=final_neu_neg(:,:,i);
count_p=hist(temp_p(:),0:0.01:1,'blue');
count_n=hist(temp_n(:),0:0.01:1,'red');
bar(count_n,'g');
hold on
bar(count_p,'b');
