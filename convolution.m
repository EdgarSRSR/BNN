%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

% CONVOLUTION
% Apply XNOR AND POPCOUNT

function rmatrix = convolution(matrix, kernel)
  % turn the kernel into a transposed vector array
  krow = reshape(kernel',1,[])
  % turn the kernel into a transposed vector array
  m = reshape(matrix',1,[])
  wy = columns(matrix) - columns(kernel) + 1
  hy = rows(matrix) - rows(kernel) + 1
  wx = columns(matrix)
  hx = rows(matrix)
  rmatrix = [0];
  for i = 1:(wx*hx - (wx + 1))
    if mod(i,wx) != 0
      %display([(i),(i+1),(i+wx),(i+wx+1)]);
      %rmatrix(i) = i;
      rmatrix(end+1) = binaryCrossCorrelation([(i),(i+1),(i+wx),(i+wx+1)],krow);
    endif
    if mod(i, wx) == 0
      continue;
    endif
  end
  rmatrix(1) = [];
  rmatrix = reshape(rmatrix,3,3)';
end
