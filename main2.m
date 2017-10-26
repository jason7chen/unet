close all;clear all;clc;
addpath('/Users/jasoncjs/Documents/MATLAB/QSM/view3dgui');
addpath('/Users/jasoncjs/Documents/MATLAB/QSM/utils');
addpath('/Users/jasoncjs/Documents/MATLAB/QSM/MEDI_toolbox');

N = [64,64,32];
samples = 100;
n = zeros(samples,3);
n(:,1) = randi([10 20], 1, samples);
n(:,2) = randi([10 20], 1, samples);
n(:,3) = randi([5 10], 1, samples);
% nn = [30, 30, 20];

voxel_size = [1,1,1];
val = rand(1, samples);

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
                    phantom(s, i, j, k) = val(s);
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
for s = 1:samples
    phase(s,:,:,:) = ifftn(D .* fftn(squeeze(phantom(s,:,:,:))));
end

D(find(abs(D)<1e-5)) = 1e-3;
for s = 1:samples
    train(s,:,:,:) = ifftn(fftn(squeeze(phase(s,:,:,:))) ./ D);
end
% view3dgui(phase);
% view3dgui(input);

save('train.mat', 'train');
save('target.mat', 'phantom');

% lambda = 1e-4;
% QSM = admm_qsm(phase, N, ones(N), voxel_size, lambda);
% view3dgui(QSM)


