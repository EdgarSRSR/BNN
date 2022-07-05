 clear;

 % Implementation of BNN mk III
 % image: 28h x 28w x 1 channel --- 784 Binarization
 % convolution 2x2 kernel + 0 padding 27h x 27w = 729
 % average pooling 2x2 kernel + 1 stride 26h x 26w = 676
 % binarization
 % convolution 2x2 kernel + 0 padding 25h x 25w = 625
 % average pooling 2x2 kernel + 1 stride 24h x 24w = 576
 % binarization
 % flatten
 % dense: 576 fully connected neurons
 % sigmoid
 % dense: 120 fully connected neurons
 % sigmoid
 % dense:10 fully connected neurons
 % soft max function
 % output: 1 0f 10 classes


 data = load('../mnist/mnist_test.csv');
 labels = data(:,1);
 images = data(:,2:785);
 size(images)
 imgTest = reshape(images(1,:),28,28)';
 % load binaryDataImages.mat % to load binaryDataImages.mat file
 % btest = reshape(binaryData{1},28,28)' % to call images from binaryData directly
 imgTestLabel = labels(1)
 kernel =  [0 1; 0 1] % check for vertical features
 kernel2 = [0 0; 1 1] % check for horizontal features
 btest = binaryConvolutionalLayers(imgTest,kernel,kernel2)
 % load ConvBinaryImgMkIII.mat
 % btest = convolutedBinaryImg{1} % call image that has already gone through the whole binaryConvolutionalLayers function
 structure = [576 120 10];
 network = generateNetwork(structure);
 epochs = 50;
 alpha = 0.00001;
 [trainedNetwork,costLog,accuracyLog]=trainNetwork(btest,imgTestLabel,network,'epochs',epochs, 'alpha',alpha);
 np=networkPredictions(btest,trainedNetwork)
 r = softMax(np)
 max(r)

 %Plot the cost log from training
 figCostLog=figure();
 plot(costLog);
 ylabel('loss');
 xlabel('epochs');


 % experiment with different kernels for each convolution
 % us the same network to train al btest and the result should be an accurate trainedNetwork



% for i =1:6000
%[trainedNetwork,costLog,accuracyLog]=trainNetwork(convolvedDataMkIII{i},arrayLabels{i},trainedNetwork,'epochs',epochs, 'alpha',alpha);

%end

%% for testing results of training
%j= [916 4416 2604 7337 4958 4976 5820 5763 773 7359 196 4594 5362 2646 6061 2546 5919 2120 726 1256 6223 7222 1112 108 5422 12 5700 5851 5187 5096 806 1589 521 1834 2504 4103 4909 449 4616 3999 7978 6953 620 140 2517 1649 6347 2889 4952 6459]

%for i = 1:length(j)
%np=networkPredictions(convolvedDataMkIII{j(i)},trainedNetwork);
%r = softMax(np)
%arrayLabels{j(i)}
%end

 % j = [7400 9163 9002 6659 8291 7882 6729 8397 6831 9340 7639 8235 8038 7090 9509 7176 8053 7706 9242 6187 7193 6223 8121 8311 9709 6610 8292 7448 8186 7391 9995 6194 9296 7829 9153 6266 7555 9566 8536 7034 8879 6669 9987 6867 9009 9635 9964 8632 9805 6163]

