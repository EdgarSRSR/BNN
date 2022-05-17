%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

% CONVOLUTION
% Apply XNOR AND POPCOUNT

function c = convolution(x,y)
  %c = [0 0 0 0];
  for i = 1:length(x)
    % x nor process
    c(i) = not(xor(x(i),y(i)));
    %fprintf(' x array %d\n', x(i) )
    %fprintf(' y array %d\n', y(i) )


  end
end
