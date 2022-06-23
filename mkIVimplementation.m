
 clear;

 % Implementation of BNN mk IV
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
 labels = data(:,1);
 images = data(:,2:785);
 size(images)
 imgTest = reshape(images(1,:),28,28)'
 imgTestLabel = labels(1)
 kernel = [0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0] % check for vertical features
 kernel2 = [0 0; 1 1] % check for horizontal features

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

  epochs = 5;

  alpha = 0.00001;

  [trainedNetwork,costLog,accuracyLog]=trainNetwork(binaryImage,imgTestLabel,network,'epochs',epochs, 'alpha',alpha);

  np=networkPredictions(binaryImage,trainedNetwork)

  softMax(np)

 % use the same network to train al binaryImage and the result should be an accurate trainedNetwork
