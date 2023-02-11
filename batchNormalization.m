## Copyright (C) 2023 Admin
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
## @deftypefn {} {@var{retval} =} batchNormalization (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Admin <Admin@DESKTOP-0KMA9I8>
## Created: 2023-01-17

function retval = batchNormalization (batch, epsilon, gamma, beta)

  % var: Computes the variance of the elements of the vector x. The variance is defined as var (x) = 1/(N-1) SUM_i (x(i) - mean(x))^2
  % Batch Normalization Forward Pass
  retval = gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta;

endfunction
