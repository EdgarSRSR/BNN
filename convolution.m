%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

% CONVOLUTION
% THIS CONVOLUTION ONLY WORKS FOR 2X2 KERNELS
% Apply XNOR AND POPCOUNT

function rmatrix = convolution(matrix, kernel)
  % turn the kernel into a transposed vector array
  krow = reshape(kernel',1,[])
  % turn the matrix into a transposed vector array
  m = reshape(matrix',1,[]);
  wy = columns(matrix) - columns(kernel) + 1
  hy = rows(matrix) - rows(kernel) + 1
  wx = columns(matrix)
  hx = rows(matrix)
  rmatrix = [0];
  for i = 1:(wx*hx - (wx + 1))
    if mod(i,wx) != 0
      %display([m(i),m(i+1),m(i+wx),m(i+wx+1)]);
      %rmatrix(i) = i;
      rmatrix(end+1) = binaryCrossCorrelation([m(i),m(i+1),m(i+wx),m(i+wx+1)],krow);
    endif
    if mod(i, wx) == 0
      continue;
    endif
  end
  rmatrix(1) = [];
  %rmatrix = reshape(rmatrix,wy,hy)';
end
