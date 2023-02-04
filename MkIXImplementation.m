
 clear;

 % Implementation of BNN mk IX
 % uses values 0 and 1 so that octave's logic gates work correctly
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
 %



 data = load('../mnist/mnist_test.csv');
 labels = data(:,1); % labels for the samples
 images = data(:,2:785);
 size(images)
 imgTest = reshape(images(200,:),28,28)' % this selects the sample used from the data base
 imgTestLabel = labels(200)
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

  [trainedNetwork,costLog,accuracyLog]=trainNetworkBinaryBatch(binaryImage,imgTestLabel,network,'epochs',epochs, 'alpha',alpha);

  np=networkPredictions(binaryImage,trainedNetwork)

  r = softMax(np)

  max(r)

 %Plot the cost log from training

 figCostLog=figure();

 plot(costLog);

 ylabel('loss');

 xlabel('epochs');

 imgTestLabel

 trainedNetwork;

 % use the same network to train al binaryImage and the result should be an accurate trainedNetwork
%  load(binaryDataImages.mat) to call the mat file with the test images turned binary
% reshape(binaryImage{2},27,27)'  show one of the binary images
 % convolving with binary
 %for i = 8001:10000
   %img = reshape(binaryData{i},28,28)';
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
% j = [41 359 39 429 621 800 410 770 894 944 68 129 768 92 204 737 523 725 984 35 217 596 21 343 506 595 605 967 796 674 455 658 61 377 198 367 696 80 387 143 496 327 271 482 111 895 495 483 282 491 736 928 412 813 817 852 539 814 650 235 850 815 999 824 189 867 753 546 208 113 627 32 171 956 313 53 180 723 865 344 583 55 501 959 310 164 260 72 474 985 58 46 754 694 8 953 660 851 612 159]
% 100 samples OVER 4000
% j = [1637 1696 1024 2576 1371 2262 1621 3134 2344 3523 4008 1240 3229 4375 1616 2359 3903 3087 2219 1929 2975 3233 4849 2377 2265 1624 4142 3121 3096 4653 2411 3748 2055 1871 1551 3630 3431 2286 4840 4829 2535 2319 4172 1169 4873 1766 4420 3145 2459 1016 3576 1533 1481 1694 3065 2462 4728 2847 3513 4065 3951 4664 3704 3175 4330 4718 2808 4789 4574 1061 4373 4957 4625 2636 2383 2948 2940 1889 2404 2011 2673 4821 2301 3054 3475 4599 2695 1529 2877 4335 4261 2211 3698 3552 2049 4298 2258 3181 2350 1490]
%j = [2900 2285 4883 2571 4373 4991 4437 3100 3942 3813 4521 2976 4970 2380 2287 4618 2210 4753 3899 3324 2369 2746 2036 2224 2213 3595 2472 3162 3916 3386 2432 4546 2454 2263 4852 4040 3953 4943 4868 4561 2216 3221 4233 3013 3141 3305 2814 4756 4174 2450 3923 2303 4998 4683 3301 2233 4461 4654 2035 2567 3174 4787 3919 3965 4821 4208 2054 2508 2352 2850 3462 2914 4997 3865 2634 3318 4788 4740 4898 2958 2240 3659 3863 3277 3832 4812 2295 2374 2389 3972 2615 2732 2005 4506 3063 4695 2149 3612 4183 2974]
% for getting the accuracy of the model
% accuracyCalculator (j, networkMkVI, convolutedImagesMkIV, arrayLabels)
j=[37 59 28 103 139 10 79 31 70 118 134 128 58 33 113 30 137 1 132 115 14 56 18 65 110 116 24 22 11 129 123 75 119 21 92 44 69 52 46 49 74 15 101 6 107 50 130 142 117 63 109 39 66 90 93 102 96 81 141 12 54 77 64 148 16 86 105 150 3 40 4 48 98 73 97 78 85 60 57 17 106 108 61 149 112 53 13 127 131 34 72 42 138 38 147 91 8 20 7 80]


% Training of the network
%load arrayLabels.mat
%  labels = data(:,1);
%load MkIVConvolvedData.mat
%load MkVIIConvolvedData.mat
 %load networkMkVI.mat
 %epochs = 200;
 %alpha = 0.0001;
 %labels = data(:,1);

 %for i=1:2000
 %  [networkMkVI,costLog,accuracyLog]=trainNetworkSTE(convolutedImagesMkIV{i},arrayLabels{i},networkMkVI,'epochs',epochs, 'alpha',alpha);
 % printf("Data sample %d training complete ", i);
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
%printf("Data sample %d training complete ", i);
%endfor
