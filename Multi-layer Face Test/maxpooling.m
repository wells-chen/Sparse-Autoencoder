function max_pool = maxpooling(patch,dimension)
[m,n] = size(patch);
%judge if the dimension can be divided exactly
max_pool = zeros(floor(m/dimension),floor(n/dimension));
i_limit = floor(m/dimension);
j_limit = floor(n/dimension);
for i=1:i_limit
    for j=1:j_limit
        max_pool(i,j) = max(max(patch((i-1)*dimension+1:i*dimension,(j-1)*dimension+1:j*dimension)));
    end
end
end