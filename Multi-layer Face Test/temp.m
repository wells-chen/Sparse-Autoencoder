
data=randn(10,10,8);
patch=randn(4,4,8);
[M,N,K]=size(data);
[m,n,K]=size(patch);
result=zeros(M-m+1,N-n+1);
resultconv=result;
for i=1:K
   temp =filter2(patch(:,:,i),data(:,:,i)); 
   result=result+temp(floor((m+1)/2):floor((m+1)/2)+M-m,floor((n+1)/2):floor((n+1)/2)+N-n);
end
for i=1:K
   patchin=patch(m:-1:1,n:-1:1,i);
   temp =conv2(patchin,data(:,:,i),'same'); 
   resultconv=resultconv+temp(floor((m+1)/2):floor((m+1)/2)+M-m,floor((n+1)/2):floor((n+1)/2)+N-n);
end
% D=zeros(M-m+1,N-n+1);
% for k=1:M-m+1
%     for l=1:N-n+1
%         temp =data(k:k+m-1,l:l+n-1,:).*patch;
%         D(k,l) = sum(temp(:));
%     end
% end
 sum(sum(result-resultconv))
 
 
 

X=ones(1000,1000);
pat=ones(100,100);
tic
for i=1:100
    patin=pat(100:-1:1,100:-1:1);
    Y=conv2(patin,X,'same');
end

toc

tic
for i=1:100
    Y=filter2(pat,X,'same');
end

toc