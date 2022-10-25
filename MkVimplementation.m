
 clear;

 % Implementation of BNN mk V
 % image: 28h x 28w x 1 channel --- 784 Binarization
 % convolution 5x5 kernel + 0 padding 24h x 24w = 576
 % average pooling 2x2 kernel + 2 stride 12h x 12w = 144
 % binarization
 % convolution 2x2 kernel + 0 padding 11h x 11w = 121
 % binarization
 % flatten
 % dense: 121 fully connected neurons
 % sigmoid
 % dense: 84 fully connected neurons
 % sigmoid
 % dense:10 fully connected neurons
 % soft max function
 % output: 1 0f 10 classes


 data = load('../mnist/mnist_test.csv');
 labels = data(:,1); % labels for the samples
 images = data(:,2:785);
 size(images)
 imgTest = reshape(images(2,:),28,28)' % this selects the sample used from the data base
 imgTestLabel = labels(2)
 kernel = [0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0] % check for vertical features
 % kernel1_horizontal = [0 0 0 0 0; 0 0 0 0 0; 1 1 1 1 1; 0 0 0 0 0; 0 0 0 0 0] check for horizontal features
 kernel2 = [0 0; 1 1] % check for horizontal features
 % kernel2_vertical = [0 1; 0 1] check for vertical features

  % work on binarizing the image by calling the deterministic binarization  function
  % binaryImage = []
  binaryImage = deterministic_binarization(imgTest)

  binaryImage = convolution2(binaryImage,kernel);

  % turn a row vector into a matrix

  binaryImage = reshape(binaryImage,24,24)'

  binaryImage = avrgPooling(binaryImage,2);

  binaryImage = reshape(binaryImage,12,12)';

  binaryImage = deterministic_binarization(binaryImage);

  binaryImage = convolution(binaryImage,kernel2);

  % turn a row into a matrix

  binaryImage = reshape(binaryImage,11,11)';

  binaryImage =deterministic_binarization(binaryImage);

  % flatten

  binaryImage = reshape(binaryImage',1,[])

  structure = [121 84 10];

  network = generateNetwork(structure);

  epochs = 150;

  alpha = 0.0001;

  [trainedNetwork,costLog,accuracyLog]=trainNetworkBinary(binaryImage,imgTestLabel,network,'epochs',epochs, 'alpha',alpha);

  np=networkPredictions(binaryImage,trainedNetwork)

  r = softMax(np)

  max(r)

 %Plot the cost log from training

 figCostLog=figure();

 plot(costLog);

 ylabel('loss');

 xlabel('epochs');

 imgTestLabel

 % use the same network to train al binaryImage and the result should be an accurate trainedNetwork
%  load(binaryDataImages.mat) to call the mat file with the test images turned binary
% reshape(binaryImage{2},27,27)'  show one of the binary images
 % convolving with binary
 %for i = 8001:10000
  % img = reshape(binaryData{i},28,28)';
   %img = convolution2(img,kernel);
   %img = reshape(img,24,24)';
   %img = avrgPooling(img,2);
   %img = reshape(img,12,12)';
   %img = deterministic_binarization(img);
   %img = convolution(img,kernel2);
   %img = reshape(img,11,11)';
   %img =deterministic_binarization(img);
   %convolutedImagesMkIV{i} = reshape(img',1,[]);
 %endfor


 %load('MkIVConvolvedData.mat')  to call the mat file with the test images that went throught the convolution process of the mIK implementation
 %reshape(convolutedImagesMkIV{2},11,11)'  to look at one sample of the data

 % fully connected network

 % the trained network is used to get the predictions, it is the fully connected network. The convolutedImegesMkIv are input
 % load('arrayLabels.mat') get file with labels placed as arrays
 % load('networkmkIV.mat') get file with network trained with test samples using the mkIv algorithm the name of the variable is trained network
 %for i = 1:length(j)
%np=networkPredictions(convolutedImagesMkIV{j(i)},trainedNetwork);
%r = softMax(np)
%arrayLabels{j(i)}
%end


%% for testing results of training
%j= [916 4416 2604 7337 4958 4976 5820 5763 773 7359 196 4594 5362 2646 6061 2546 5919 2120 726 1256 6223 7222 1112 108 5422 12 5700 5851 5187 5096 806 1589 521 1834 2504 4103 4909 449 4616 3999 7978 6953 620 140 2517 1649 6347 2889 4952 6459]

% j = [7400 9163 9002 6659 8291 7882 6729 8397 6831 9340 7639 8235 8038 7090 9509 7176 8053 7706 9242 6187 7193 6223 8121 8311 9709 6610 8292 7448 8186 7391 9995 6194 9296 7829 9153 6266 7555 9566 8536 7034 8879 6669 9987 6867 9009 9635 9964 8632 9805 6163]

% for getting the accuracy of the model
% accuracyCalculator (j, trainedNetwork, convolutedImagesMkIV, arrayLabels)
