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
 imgTestLabel = labels(1)
 kernel =  [0 1; 0 1] % check for vertical features
 kernel2 = [0 0; 1 1] % check for horizontal features
 btest = binaryConvolutionalLayers(imgTest,kernel,kernel2)
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
