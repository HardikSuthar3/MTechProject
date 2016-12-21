function [ frames ] = GetNMStimulusFrames()
%GETNMSTIMULUSFRAMES Generates downsampled 4 types of natural stimulus
%frames for Image Feature Extraction

frames = [];

load('/home/hardik/Desktop/MTech_Project/Data/NM_Stimulus/MovsMat/mov1.mat');
[F,~] = DownsampleFrames(movnew,20,4);
frames = cat(3,frames,F);


load('/home/hardik/Desktop/MTech_Project/Data/NM_Stimulus/MovsMat/mov2.mat');
[F,~] = DownsampleFrames(movnew,20,4);
frames = cat(3,frames,F);


load('/home/hardik/Desktop/MTech_Project/Data/NM_Stimulus/MovsMat/mov3.mat');
[F,~] = DownsampleFrames(movnew,20,4);
frames = cat(3,frames,F);


load('/home/hardik/Desktop/MTech_Project/Data/NM_Stimulus/MovsMat/mov4.mat');
[F,~] = DownsampleFrames(movnew,20,4);
frames = cat(3,frames,F);


end

