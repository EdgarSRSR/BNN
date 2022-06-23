## Copyright (C) 2022 edgar
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {} {@var{retval} =} convolution2 (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: edgar <edgar@DESKTOP-VS5G4BO>
## Created: 2022-06-23

function rmatrix = convolution2 (matrix, kernel)

  wy = columns(matrix) - columns(kernel) + 1;
  hy = rows(matrix) - rows(kernel) + 1;
  wx = columns(matrix);
  hx = rows(matrix);
  ki= rows(kernel);
  kj= columns(kernel);
  krow = reshape(kernel',1,[]);
  m = [];
  rmatrix = [0];

  for i=1:wy
    for j=1:hy
      m = [matrix(i:i+(ki-1),j:j+(kj-1))];
      m = reshape(m',1,[]);
      rmatrix(end+1) = binaryCrossCorrelation(m,krow);
    endfor
  endfor
  rmatrix(1) = [];

endfunction
