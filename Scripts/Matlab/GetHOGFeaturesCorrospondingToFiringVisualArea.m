function [ area_feature,area_label ] = GetHOGFeaturesCorrospondingToFiringVisualArea( area_number )
%   This function will return the HOG Features of The Stimulus for the time
%   when a perticular visual area(area_number) was fired.

%% Get the HOG Features of Stimulus
[hog_features,hog_labels] = GenerateHOGFeaturesForResponseMapping();
lbl = repmat(hog_labels,91,1);
lbl = [lbl;hog_labels(1:80)];

%% Get The Visual Area Average Firing
load('/home/hardik/Desktop/MTech_Project/Data/Feature/NaturalMovies.mat')
nm_feature = nm_feature';
[~,ind] = max(nm_feature,[],2);

%% Get Visual Area Features

n = size(lbl,1);
area_feature =[];
area_label = [];
for i=1:n
    if(ind(i)==area_number)
        index = mod(i,320); 
        if(index==0)
            index=320;
        end
        area_feature = cat(1,area_feature,hog_features(index,:));
        area_label = cat(1,area_label,lbl(i));
    end
end
end

