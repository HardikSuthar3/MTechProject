function [ F,ind ] = DownsampleFrames( frames,rate,seconds )
%UNTITLED Takes high frequency frames and returns the downsampled frames
%   Detailed explanation goes here

n = size(frames,3);

d = n/(rate*seconds); 

ind = downsample(1:n,d);

F = frames(:,:,ind);

end

