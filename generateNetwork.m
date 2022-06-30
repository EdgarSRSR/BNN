%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

% GENERATENETWORK Generates the weight matrices for the fully connected neural network and fills them
% with binarized random numbers.
% A = GENERATENETWORK(structure) generates the matrices for the given structure network
% [784 128 10]
%  - Input layer has 784 neurons for all pixels of the 28x28 image
%  - 128 hidden layers
%  - 10 output layers, because there are 10 classes where we can classify an image

function[network] = generateNetwork(structure)
  numberOfLayer = length(structure);
  numberOfThetas = numberOfLayer -1; %Number of matrices between the layer
  offset = 1; %Add Bias/Offset
  theta{numberOfThetas} = {}; %Initialize the array for matrices that connect the layer

  %Create and fill the weight matrices
  for i = 1:numberOfThetas
    %matrices Range for: -0.5 to 0.5. This values will be later binarized
    theta{i} = rand(structure(i+1)+offset,structure(i)+offset)-0.5;
    % binarized thetas
    %theta{i} = deterministic_binarization(rand(structure(i+1)+offset,structure(i)+offset)-0.5);
  endfor

  % Remove the offset from the last (output) layer
  theta{numberOfThetas}(end,:) =[];

  network = theta;

 end


