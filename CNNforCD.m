%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convolutional Neural Networks for the prediction of force coefficients
%
% This program is developed using MATLAB 2017b and MATLAB - Neural Networks
% toolbox. It utilizes the CNN-based stochastic gradient method for the 
% training and prediction of fluid forces on bluff bodies. To use this, the 
% full-order results should be available for the training bluff body set.
% The input function matrices should be available for both the training and 
% intended prediction bluff body set.

disp(' ');
disp(' ================================================================');
disp('             WELCOME TO CNN-BASED LEARNING FOR CFD               ');
disp(' ================================================================');
disp(' ');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input data

clear
load TrainSetSmooth          % Loads the input function set
load CDFOM                   % Loads the FOM result set
load TestSet                 % Loads the intended predction set (optional)

% The input function has to be constructed as a 4D tensor. For 'S' inputs of
% size 'm x n', the input tensor will be of (m,n,1,S). In this example, the
% 'TrainSetSmooth.mat' file contains 13 input functions with the index 'S'
% representing the following cases:
%
% Ellipses
% S = 1:Circular cylinder, S = 2:Aspect Ratio=2, S = 3:AR=3, S = 4:AR=4, 
% S = 5:AR=5, S = 6:AR=7, S = 7:AR=10,
%
% Rounded squares
% S = 8:Rounding angle = 10 degrees, S = 9: 20 deg, S = 10: 30 deg,
% S = 11: 40 deg, S = 12: 50 deg, S = 13: 60 deg.
%
% The vector 'CDFOM.mat' contains the drag coefficient obtained from the 
% full order simulations for the 13 cases in 'TrainSetSmooth.mat'
%
% 'TestSet.mat' file contains 14 input functions with the index 'S'
% representing the following cases:
%
% Ellipses
% S = 1:AR=1.1, S = 2:AR=1.25, S = 3:AR=1.75, S = 4:AR=1.5, 
% S = 5:AR=2.25, S = 6:AR=2.5, S = 7:AR=2.75, S = 8:AR=3.5
%
% Rounded squares
% S = 9:Rounding angle = 5 degrees, S = 10: 15 deg, S = 11: 25 deg,
% S = 12: 35 deg, S = 13: 45 deg, S = 14: 55 deg.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training process

% Training data for input-output process
InputSet = [1 2 3 7 8 10 13];                % Select the training set
TrainInput = TrainSetSmooth(:,:,1,InputSet); 
TrainOutputr = CD(InputSet);                 % Select the FOM result set
TrainOutput = TrainOutputr-mean(TrainOutputr); 
TrainOutput = TrainOutput';

% Call for built-in CNN function
layersSmooth = [ ...
    imageInputLayer([201 301 1])            % Input layer
    convolution2dLayer(4,50,'Stride',1)     % Convolution layer
    reluLayer                               % Non-linearization layer
%   maxPooling2dLayer(4,'Stride',4)         % Down-sampling layer                      
    fullyConnectedLayer(1)                  % Fully-connected layer
    regressionLayer];                       % Output layer

% Tuning of hyperparameters
optionsSmooth = trainingOptions('sgdm','VerboseFrequency',1,...
                'InitialLearnRate',0.01, ... % Learning rate
                'MiniBatchSize', 7, ...      % Stochastic batch size   
                'MaxEpochs',100);            % Maximum training iterations
                            
% Training step
netSmooth = trainNetwork(TrainInput,TrainOutput,layersSmooth,optionsSmooth);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prediction

% Predict the entire input set
CDPre = predict(netSmooth,TrainSetSmooth)+mean(TrainOutputr);

% Predict the training set
CDPreIn = predict(netSmooth,TrainInput)+mean(TrainOutputr);

% Predict the test set
CDPreTe = predict(netSmooth,TestSet)+mean(TrainOutputr);

% Calculate the percentage error
Error = abs(CDPre-CD')./CD'*100;

% Display the maximum error
ErrorMax = max(Error)

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                          OUTPUT: POSTPROCESSING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the CD predictions vs FOM results

figure
L1 = line (AR,CD(1:7),'Color','k','Marker','o','LineStyle',...
    '-','MarkerFaceColor','k'); % FOM results
L2 = line (AR,CDPre(1:7),'Color','r','Marker','s',...
    'LineStyle','--'); % CNN prediction
xlim([0.5 10.5])
ylim([1.2 2.6])
ylabel ('$$\overline{C_D}$$','FontSize',20,'Interpreter','latex')
xlabel ('Ellipses: Aspect Ratio','FontSize',20)
set(gca,'box','off')
ax1=gca;
ax1_pos = ax1.Position;
ax2 = axes('Position',ax1_pos,'XAxisLocation','top','YAxisLocation',...
    'right','YTick',[],'Color','none');

% % Rounded Squares
phi = [10 20 30 40 50 60];

L3 = line (phi,CD(8:13),'Parent',ax2,'Color','k','Marker','^',...
    'LineStyle','-','MarkerFaceColor','k');    % FOM results
L4 = line (phi,CDPre(8:13),'Parent',ax2,'Color','b',...
    'Marker','d','LineStyle','--');            % CNN prediction
xlim([9.5 60.5])
ylim([1.2 2.6])
xlabel ('Rounded Squares: Rounding angle','FontSize',20)
legend([L1; L2; L3; L4],{'FOM-Ellipses','CNN-Ellipses',...
    'FOM-Rounded Sq.','CNN-Rounded Sq.'},'FontSize',16,'Location','East')