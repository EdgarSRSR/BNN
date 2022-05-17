%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

% Calculate the result of the cross correlation


function bcc = binaryCrossCorrelation(x,y)
  xnr = xnor(x,y);
  n = length(xnr);
  p = popcount(xnr);
  bcc = 2*p-n;
 end
