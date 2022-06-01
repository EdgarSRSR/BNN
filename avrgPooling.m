%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================


% SAverage Pooling compute Pooling by calculating average of values in filters

function avrg = avrgPooling (matrix,stride)

  wx = columns(matrix)
  hx = rows(matrix)
  avrg = [0];
  for i=1:stride:hx
    for j=1:stride:wx
      display([matrix(i,j),matrix(i,j+1),matrix(i+1,j),matrix(i+1,j+1)])
      avrg(end+1)= mean([matrix(i,j),matrix(i,j+1),matrix(i+1,j),matrix(i+1,j+1)]);
    endfor
  endfor
  avrg(1) = [];
  display(avrg)
  avrg = reshape(avrg,(wx/stride),(hx/stride))';

endfunction
