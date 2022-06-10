%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

% Processes the convolutional layers. Includes convolution, average pooling and binarization


function binaryImage = binaryConvolutionalLayers (image, kernel)

  % work on binarizing the image by calling the deterministic binarization  function
  binaryImage = []
  binaryImage =deterministic_binarization(image);


  binaryImage = convolution(binaryImage,kernel);

  binaryImage = avrgPooling(binaryImage,2);

  binaryImage =deterministic_binarization(image);


  binaryImage = convolution(binaryImage,kernel);

  binaryImage(i,j) =deterministic_binarization(image);




endfunction
