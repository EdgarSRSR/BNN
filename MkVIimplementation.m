
 clear;

 % Implementation of BNN mk VI
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
 %load('networkMkVI')
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

  %network = networkMkVI;

  epochs = 200;

  alpha = 0.0001;

  [trainedNetwork,costLog,accuracyLog]=trainNetworkSTE(binaryImage,imgTestLabel,network,'epochs',epochs, 'alpha',alpha);

  np=networkPredictions(binaryImage,trainedNetwork)

  r = softMax(np)

  max(r)

 %Plot the cost log from training

 figCostLog=figure();

 plot(costLog);

 ylabel('loss');

 xlabel('epochs');

 imgTestLabel

 trainedNetwork

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


 %load('MkVIConvolvedData.mat')  to call the mat file with the test images that went throught the convolution process of the mIK implementation
 %reshape(convolutedImagesMkVI{2},11,11)'  to look at one sample of the data

 % fully connected network

 % the trained network is used to get the predictions, it is the fully connected network. The convolutedImegesMkIv are input
 % load('arrayLabels.mat') get file with labels placed as arrays
 % load('networkMkVI.mat') get file with network trained with test samples using the mkIv algorithm the name of the variable is trained network
 %for i = 1:length(j)
%np=networkPredictions(convolutedImagesMkIV{j(i)},trainedNetwork);
%r = softMax(np)
%arrayLabels{j(i)}
%end


%% for testing results of training
%j= [916 4416 2604 7337 4958 4976 5820 5763 773 7359 196 4594 5362 2646 6061 2546 5919 2120 726 1256 6223 7222 1112 108 5422 12 5700 5851 5187 5096 806 1589 521 1834 2504 4103 4909 449 4616 3999 7978 6953 620 140 2517 1649 6347 2889 4952 6459]
%j = [1798	1543	1591	470	502	723	1203	180	1750	1763	197	736	1112	1697	977	1883	440	635	1039	1895	1910	1933	183	1524	1570	109	678	355	920	887	19	185	1953	117	169	82	664	116	1839	198	876	1281	533	1640	7	542	1669	1399	1932	277	1672	1314	1089	645	2000	1766	337	1955	1591	367	982	984	812	1494	695	214	533	215	596	662	1078	1278	1521	1134	1427 1314	1489	1530	578	1120	1002	619	1422	1846	637	588	1161	1242	81	29	1793	1305	173	28	1502	1818	883	1166	1191	407]
% j = [7400 9163 9002 6659 8291 7882 6729 8397 6831 9340 7639 8235 8038 7090 9509 7176 8053 7706 9242 6187 7193 6223 8121 8311 9709 6610 8292 7448 8186 7391 9995 6194 9296 7829 9153 6266 7555 9566 8536 7034 8879 6669 9987 6867 9009 9635 9964 8632 9805 6163]
% 100 samples under 1000
% j = [835 827 687 283 942 151 818 392 441 475 707 923 968 53 568 631 220 635 131 754 914 281 967 288 363 455 250 682 240 483 918 177 657 824 364 985 728 886 318 666 206 522 545 741 935 689 171 470 105 92 674 670 586 884 106 737 200 999 88 767 188 50 306 408 246 898 930 906 266 242 730 52 944 892 112 491 20 12 535 864 848 561 186 176 75 532 73 229 128 103 785 119 851 932 870 979 168 662 834 623]
% 100 samples OVER 4000
% j =[4053 6852 3995 7605 8505 6964 3379 7422 2043 6494 7398 3994 4601 2892 6287 7625 5949 6271 3006 7644 3446 8159 7888 2276 3616 7028 5296 8614 5981 8102 6848 7572 6374 4428 3118 3539 3989 5947 8517 8373 8876 3864 8900 8445 4095 4297 5004 6699 2379 8000 7528 8301 6585 4951 4748 6815 6157 3624 2614 8748 8985 8895 7304 8463 8232 7645 6331 2869 6704 8303 6044 5739 4735 8567 2603 3760 2830 6594 4511 8361 4255 6105 7903 3523 6381 6810 6869 8476 6190 8016 8374 4913 3179 5398 5917 8639 7121 5584 6009 4322]
% for getting the accuracy of the model
% accuracyCalculator (j, networkMkVI, convolutedImagesMkIV, arrayLabels)


% Training of the network
%load arrayLabels.mat
%  labels = data(:,1);
%load MkIVConvolvedData.mat
 %load networkMkVI.mat
 %epochs = 200;
 %alpha = 0.0001;
 %labels = data(:,1);

 %for i=1:2000
 %  [networkMkVI,costLog,accuracyLog]=trainNetworkSTE(convolutedImagesMkIV{i},labels(i),networkMkVI,'epochs',epochs, 'alpha',alpha);
 % printf("Data sample %d training complete", i);
 %endfor



% labels = data(:,1);
% load MkIVConvolvedData.mat
% load networkMkVI.mat
% epochs = 200;
% alpha = 0.0001;
%structure = [121 84 10];
% network = generateNetwork(structure);
% for i=1:1000
%[network,costLog,accuracyLog]=trainNetworkSTE(convolutedImagesMkIV{i},labels(i),network,'epochs',epochs, 'alpha',alpha);
%printf("Data sample %d training complete", i);
%endfor
