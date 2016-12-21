function [hog_features,hog_labels] = GenerateHOGFeaturesForResponseMapping()
%% Get the Stimulus Frame
frames = GetNMStimulusFrames();

%% Generate HOG Features

n = size(frames,3);
hog_features=[];
for i=1:n
    I = frames(:,:,i);
    [hog_8x8, vis8x8] = extractHOGFeatures(I,'CellSize',[8 8]);
    hog_features = cat(1,hog_features,hog_8x8);
end

%% Creating Label Vector

hog_labels = zeros(size(hog_features,1),1);
for i=0:3
    hog_labels(80*i+1:80*(i+1),1)= i;
end

end
