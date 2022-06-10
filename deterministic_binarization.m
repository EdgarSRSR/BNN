%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

% DETERMINISTIC_BINARIZATION Biniarizes the value by turning it into a +1 value if it is equal or above 0
% or to -1 if it below 0

function binaryImage = deterministic_binarization(image)

  binaryImage = []
  for i = 1:rows(image)
    for j = 1:columns(image)
      if (image(i,j) > 0)
        binaryImage(i,j)  = 1;
      else
        binaryImage(i,j)  = 0;
      endif
     endfor
  endfor
  %if (x > 0)
  %  v = 1;
  %else
  %  v = 0;
end

