%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================
%===============================Description=====================================
%
%   Binary implementation of the LeNet convolutional network for the analysis
%   of images.
%
%===============================================================================

clear; clc; close all:

%========================== Constants Definition ===============================

epochs = 400; %Number of epochs the network will be strained. For good results it must be 1000,  but we will start lower
alpha = 0.2; %Learning rate for the training process

width=28; %Width of the picture
height=28; %height of the picture

%======================= Constants Definition ==================================

fprintf('Reading and Preparing Training Data  \n')

%Picture that will be ued for training
inputPicture = imread('');
%Picture with labels that corresponds to input
labelPicture = imread('');

%Looking at the loaded pictures before training
imshow(inputPicture);
figure();
imshow(labelPicture);

%Prepare the data for the training
inputPicture = cast(inputPicture, 'double'); %casting from uint into double

%Create a matrix with the dimensions of the picture for the later labelPicture
labels = zeros(height,width);

%==================================Generate Network=============================
fprintf('Generate Network \n')

% Define the network structure as a vector
% [784 128 10]
%  - Input layer has 784 neurons for all pixels of the 28x28 image
%  - 128 hidden layers
%  - 10 output layers, because there are 10 classes where we can classify an image

networkStructure = [784 128 10];

%Create network
network = generateNetwork(networkStructure);

%=================================Training======================================
fprintf('Start Training\n');

%Train the network.
[trainedNetwork,costLog,accuracyLog]= trainedNetwork(inputPicture,labels,network,'epochs',epochs,'alpha',alpha);

%Plot the cost log from training
figCostLog=figure();
plot(costLog);
ylabel('Loss');
xlabel('epochs');

fprintf('Training Done\n')

%============================Prediction=========================================
fprintf('Using Trained Network for Test Prediciton\n')

%Use the trained network on the inputPicture to see results
predOutput = networkPrediciton(inputPicture, trainedNetwork);

%============================Generate Results===================================

fprintf('Results \n')

%Remove the last line from the first matrix, because those are wiegths for
% connections that go to the bias in the hidden layer, which are not needed for VHDL IM=Mplementation
nnParams = trainedNetwork;
nnParams{1} = nnParams{1}(1:end-1,:); %ignore last column

frpintf('\nWeight Matrix from the Input to the HiddenLayer\n')
disp(nnParams{1});
fprintf('Weight Matrix from the Hidden to the Output Layer\n')
disp(nnParams{2});

save('bb_lenet.mat','trainedNetwork','networkStructure','nnParams');
fprintf('\nFinished Script\n')


