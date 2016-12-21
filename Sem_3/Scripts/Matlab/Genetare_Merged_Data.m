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
B = [];
while(index<n)
    if(flag==false)
        flag = true;
        index = index + 40;
    else
        flag = false;
        B = cat(3,B,A(:,:,index:index+39));
        index = index + 40;
    end
end