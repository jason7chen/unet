close all;clear all;clc;
addpath('/Users/jasoncjs/Documents/MATLAB/QSM/view3dgui');
addpath('/Users/jasoncjs/Documents/MATLAB/QSM/utils');
addpath('/Users/jasoncjs/Documents/MATLAB/QSM/MEDI_toolbox');

N = [256,256,128];
samples = 6;
n = zeros(samples,3);
n(:,1) = randi([10 50], 1, samples);
n(:,2) = randi([10 50], 1, samples);
n(:,3) = randi([10 30], 1, samples);
% nn = [30, 30, 20];

voxel_size = [1,1,1];
chi1 = 0.2;
chi2 = 0.5;
truth = chi2 - chi1;

phantom = zeros([samples, N]);

%% cuboid
% phantom(((-n(1)/2+1):(n(1)/2))+N(1)/2, ((-n(2)/2+1):(n(2)/2))+N(2)/2, ((-n(3)/2+1):(n(3)/2))+N(3)/2) = chi1;
% phantom(((-nn(1)/2+1):(nn(1)/2))+N(1)/2, ((-nn(2)/2+1):(nn(2)/2))+N(2)/2, ((-nn(3)/2+1):(nn(3)/2))+N(3)/2) = chi2;

%% ellipsoids
for s = 1:samples
    for i = 1:N(1)
        for j = 1:N(2)
            for k = 1:N(3)
                if((i - N(1)/2)^2 / n(s,1)^2 + (j - N(2)/2)^2 / n(s,2)^2 + (k - N(3)/2)^2 / n(s,3)^2 <= 1)
                    phantom(s, i, j, k) = chi1;
                end
    %             if((i - N(1)/2)^2 / nn(1)^2 + (j - N(2)/2)^2 / nn(2)^2 + (k - N(3)/2)^2 / nn(3)^2 <= 1)
    %                 phantom(i, j, k) = chi2;
    %             end
            end
        end
    end    
end



% mask1 = zeros(N);
% mask1(((-n(1)/2+1):(n(1)/2))+N(1)/2, ((-n(2)/2+1):(n(2)/2))+N(2)/2, ((-n(3)/2+1):(n(3)/2))+N(3)/2) = 1;
% mask1 = logical(mask1);
% 
% mask2 = zeros(N);
% mask2(((-nn(1)/2+1):(nn(1)/2))+N(1)/2, ((-nn(2)/2+1):(nn(2)/2))+N(2)/2, ((-nn(3)/2+1):(nn(3)/2))+N(3)/2) = 1;
% mask2 = logical(mask2);

D = dipole_kernel(N,voxel_size,[0;0;1]);
% view3dgui(phantom);
for s = 1:samples
    phase(s,:,:,:) = ifftn(D .* fftn(squeeze(phantom(s,:,:,:))));
end

% view3dgui(phase);

save('train.mat', 'phase');
save('target.mat', 'phantom');

% lambda = 1e-4;
% QSM = admm_qsm(phase, N, ones(N), voxel_size, lambda);
% view3dgui(QSM)


