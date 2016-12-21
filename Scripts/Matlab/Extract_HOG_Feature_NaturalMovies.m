%% Get The Stimulus Frames
frames = [];

load('/home/hardik/Desktop/MTech_Project/Data/NM_Stimulus/MovsMat/mov1.mat');
frames = cat(3,frames,movnew);
load('/home/hardik/Desktop/MTech_Project/Data/NM_Stimulus/MovsMat/mov2.mat');
frames = cat(3,frames,movnew);
load('/home/hardik/Desktop/MTech_Project/Data/NM_Stimulus/MovsMat/mov3.mat');
frames = cat(3,frames,movnew);
load('/home/hardik/Desktop/MTech_Project/Data/NM_Stimulus/MovsMat/mov4.mat');
frames = cat(3,frames,movnew);
clearvars movnew;

%% Extract HOG Features for Each Frame

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
    hog_labels(240*i+1:240*(i+1),1)=i;
end

%% Save The Result
save('/home/hardik/Desktop/MTech_Project/Data/HOG_Feature_Data/natural_movies_hog_features.mat','hog_features','hog_labels');