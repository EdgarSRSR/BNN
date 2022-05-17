%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

% DETERMINISTIC_BINARIZATION Biniarizes the value by turning it into a +1 value if it is equal or above 0
% or to -1 if it below 0

function v = deterministic_binarization(x)
  if (x > 0)
    v = 1;
  else
    v = 0;
end

