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
## @deftypefn {} {@var{retval} =} batchnorm_forward (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Admin <Admin@DESKTOP-0KMA9I8>
## Created: 2023-01-27

function [xhat, xmu, ivar, sqrtvar, var] = batchnorm_forward (batch, epsilon, gamma, beta)

  N = rows(batch)
  D = columns(batch)

  # step1: calculate mean
  mu = 1./N*sum(batch)

  #step2: substract mean vector of every trainings example
  xmu = batch - mu

  #step3:
  sq = xmu^2

  #step4:calculate variance
  var = 1./N * sum(sq)

  #step5: add eps for numerical stability, then sqrt
  sqrtvar = sqrt(var + epsilon)

  #step6: invert sqrtvar
  ivar = 1./sqrtvar

  #step7: execute normalization
  xhat = xmu * ivar

  #step8: Nor the two transformation steps
  gammax = gamma * xhat

  #step9:
  out = gammax + beta


endfunction
