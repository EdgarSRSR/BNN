%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

% XNOR GATE

function c = xnor(x,y)

  for i = 1:length(x)

    c(i) = not(xor(x(i),y(i)));
  end
end
