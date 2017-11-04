%%
close all;clear all;clc;
addpath('/Users/jasoncjs/Documents/MATLAB/QSM/view3dgui');
addpath('/Users/jasoncjs/Documents/MATLAB/QSM/utils');
addpath('/Users/jasoncjs/Documents/MATLAB/QSM/MEDI_toolbox');

load('spatial_res.mat')
load('temp.mat')
total_size = size(temp);
samples = total_size(1);
N = total_size(2:4);
% n = zeros(samples,3);
% n(:,1) = randi([15 30], 1, samples);
% n(:,2) = randi([15 30], 1, samples);
% n(:,3) = randi([5 15], 1, samples);
% nn = [30, 30, 20];


voxel_size = spatial_res;
val = rand(1, samples);
phantom = double(temp);


D = dipole_kernel(N,voxel_size,[0;0;1]);
for s = 1:samples
    phase(s,:,:,:) = ifftn(D .* fftn(squeeze(phantom(s,:,:,:))));
end

DD = 1 ./ D;
DD(find(abs(DD)>1e5)) = 0;
for s = 1:samples
    train(s,:,:,:) = ifftn(fftn(squeeze(phase(s,:,:,:))) .* DD);
end
% view3dgui(phase);
% view3dgui(input);

save('train.mat', 'train');
save('target.mat', 'phantom');
% 
% lambda = 1e-4;
% QSM = admm_qsm(squeeze(phase(end,:,:,:)), N, ones(N), voxel_size, lambda);
% view3dgui(QSM)


