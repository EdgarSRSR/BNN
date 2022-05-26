%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

% CONVOLUTION
% Apply XNOR AND POPCOUNT

function rmatrix = convolution(matrix, kernel)
  % turn the matrix into a transposed vector array
  krow = reshape(kernel',1,[])
  % turn the kernel into a transposed vector array
  m = reshape(matrix',1,[])
  wy = columns(matrix) - columns(kernel) + 1
  hy = rows(matrix) - rows(kernel) + 1

  for i = 1:(wy*hy)
      rmatrix(i) = 1;
  end
end
