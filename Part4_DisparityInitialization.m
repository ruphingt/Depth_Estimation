%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%               Depth Initialization 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
close all;clear all;
addpath('GCMex\GCMex')

video = dir('Road\Road\src\*.jpg')
loc = 'Road\Road\src\';
numIm = length(video); %141

% Load the K, R and T
% Generated using save_camera_matrices.m
load('camera_params.mat')
% Load the pairwise sparse matrix for each frame
% This would be the pairwise matrix already multiplied with lambda
% It was generated using multiply_lambda_pairwise.m script
load('pairwise_frames.mat', 'pairwise_array');
% Number of frames to process
numframes = 10;

% Number of classes
C = 820
% Disparities to check
d = linspace(0.00030,0.0085,C); 

% Geenerate Prior term
[X Y] = meshgrid(1:C, 1:C);
etha = 200 % Threshold for labelcost
labelcost = min(etha,abs(X - Y)); % 820x820 double
wr = 100 % Smoothness strength

% Array to store initial disparity maps for 10 frames
Dt_init = cell(numframes,1);

for  t1 = 1:numframes
    K1 = K(:,:,t1); % 3x3
    R1 = R(:,:,t1); % 3x3
    T1 = T(:,t1); % 3x1
    
    % Load first image and get the size
    I1 = im2double(imread(strcat(loc,video(t1).name))); % 540x960x3
    [H,W,channels] = size(I1); 
    totalpixels = H*W;
    figure(1)
    subplot(2,5,t1); imshow(I1);
    I1 = reshape(I1,[],channels); % 518400 x 3

    pairwise = pairwise_array{t1}*wr;
    
    [coordx, coordy] = meshgrid(1:W,1:H); % coordx: 960x540 coordy: 960x540
    coordx = (reshape(coordx,[],1))'; % 1x518400
    coordy = (reshape(coordy,[],1))'; % 1x518400
    xh = [coordx; coordy; ones(1,totalpixels)]; % 3x518400
    ind1 = sub2ind([H,W],xh(2,:),xh(1,:)); %1x518400
    clearvars X Y coordx coordy
    
    % Data terms settings
    sigma = 1
    datacost = zeros(C,totalpixels); %820x518400
    
    for  t2 = 1:numframes
        if t2 == t1
            continue
        end
        
        K2 = K(:,:,t2);
        R2 = R(:,:,t2);
        T2 = T(:,t2);
        e_prime = K2*R2'*(T1-T2);
        x_prime_inf = K2*R2'*R1*inv(K1);

        imgname = video(t2).name;
        I2 = im2double(imread(strcat(loc,imgname)));
        I2 = reshape(I2,[],channels); % 168750 x 3

        for class = 1:C
           xh_prime = x_prime_inf*xh + d(class)*e_prime; % 3x518400
           xh_prime = xh_prime./xh_prime(3,:); % normalize to (x/z; y/z; 1)

           % Can access the matrix using integer numbers only
           xh_prime = round(xh_prime);
           % Replace all out of bound coordinates
           xh_prime(1,find(xh_prime(1,:) > W)) = W;
           xh_prime(2,find(xh_prime(2,:) > H)) = H;
           xh_prime(find(xh_prime<1)) = 1;

           ind2 = sub2ind([H,W],xh_prime(2,:),xh_prime(1,:)); 

           datacost(class,:) = datacost(class,:) + ...
               sigma./(vecnorm((I1(ind1,:) - I2(ind2,:))')+sigma); % 1x518400
        end

    end

    U_x = 1./max(datacost(:,:)); % 1x518400
    datacost = 1 - U_x.*datacost; %820 x 518400 double
    [~,segclass] = min(datacost);
    segclass = (segclass-1)'; % 518400x1 double

    [labels, ~, ~] = GCMex(segclass, single(datacost), pairwise, single(labelcost),1);
    Inew = reshape(labels,[H,W]);
    Dt_init{t1} = labels;
    
end

save('Dt_init.mat', 'Dt_init');
figure(2)

for t = 1:numframes
    strarray = ["disp_init/test", int2str(t), "_init.png"];
    imageName = join(strarray,"");
    A = mat2gray(reshape(Dt_init{t},[H,W]));
    figure(1+t); imshow(A); 
    imwrite(A,imageName, 'BitDepth', 16);
end
