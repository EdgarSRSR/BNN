%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

% CONVOLUTION
% Apply XNOR AND POPCOUNT

function c = convolution(x,y)

  for i = length(x)
    c(i) = not(xor(x(i),y(i)));
  endfor
end
