%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

% NETWORKPREDICTIONS Calculates the output of a neural network to a given input.
% Y = NETWORKPREDICTION(X,network) Calculates response of the given network for X.

function [prediction] = networkPredictions(X, network)

  % number of weight matrices and layers
  numberOfThetas = length(network);
  numberOfLayers = numberOfThetas +1;

  % matrices for the Output of each Layer
  layer=cell(1,numberOfLayers);

  layer{1} = X';
  % Add offset to the layers
  layer{1}=[layer{1}; ones(1,size(layer{1},2))];

  % forward propagation to calculate output using sigmoid function
  for j=1:numberOfThetas
    % By the forward calculation the offset neuron gets inserted
    % into the activation function. This needs to be reversed before the next layer is calculated.
    layer{j}(end,:)=1;
    layer{j+1} = sigmoid(network{j} * layer{j});
    % to use the softmax activation function for the result
    %layer{j+1} = softMax(network{j} * layer{j});
  endfor

  prediction = layer{numberOfLayers};

 endfunction

