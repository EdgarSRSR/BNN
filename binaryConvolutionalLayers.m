%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

% Processes the convolutional layers. Includes convolution, average pooling and binarization


function binaryImage = binaryConvolutionalLayers (image, kernel, kernel2)

  % work on binarizing the image by calling the deterministic binarization  function
  % binaryImage = []
  binaryImage = deterministic_binarization(image);

  binaryImage = convolution(binaryImage,kernel);

  % turn a row vector into a matrix

  binaryImage = reshape(binaryImage,27,27)';

  binaryImage = avrgPooling(binaryImage,1);

  binaryImage = deterministic_binarization(binaryImage);

  binaryImage = convolution(binaryImage,kernel2);


  % turn a row into a matrix

  %binaryImage = reshape(binaryImage,26,26)'; % if not using avrgPooling

  binaryImage = reshape(binaryImage,25,25)';

  binaryImage = avrgPooling(binaryImage,1);

  binaryImage =deterministic_binarization(binaryImage);

  % flatten

  binaryImage = reshape(binaryImage',1,[])


endfunction
