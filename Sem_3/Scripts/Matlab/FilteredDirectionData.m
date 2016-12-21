% This Script generates only the responses corrosponding to stimuli and
% eliminates the response for blank screen

clear;
dirName = '';
A = StimuliResponse(dirName);

M = mean(A,3);

n = size(A,3);
for i=1:n
    A(:,:,i) = imsubtract(A(:,:,i),M);
    A(:,:,i) = mat2gray(A(:,:,i));
    A(:,:,i) = im2double(A(:,:,i));
end

m = n/40;

k = m - floor(m);

flag = false;

index = 1;
filteredResponse = [];
while(index<n)
    if(flag==false)
        flag = true;
        index = index + 40;
    else
        flag = false;
        filteredResponse = cat(3,filteredResponse,A(:,:,index:index+39));
        index = index + 40;
    end
end