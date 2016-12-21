clear;
for i=1:9
    [area_feature,area_label] = GetHOGFeaturesCorrospondingToFiringVisualArea(i);
    fileName = sprintf('/home/hardik/Desktop/MTech_Project/Data/HOG_Feature_Data/area%d',i);
    save(fileName,'area_feature','area_label');
end