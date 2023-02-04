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
## @deftypefn {} {@var{retval} =} batchnorm_backward (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Admin <Admin@DESKTOP-0KMA9I8>
## Created: 2023-01-27

function [dx, dgamma, dbeta] = batchnorm_backward (dout, xhat, gamma, epsilon, xmu, ivar, sqrtvar, var)

  N = rows(dout)
  D = columns(dout)

  #step9
  dbeta = sum(dout)

  #step8
  dgamma = sum(dout*xhat)
  dxhat = dout * gamma

  #step7
  divar = sum(dxhat*xmu)
  dxmu1 = dxhat * ivar

  #step6
  dsqrtvar =-1./(sqrtvar^2)*divar

  #step5
  dvar = 0.5*1. / sqrt(var+epsilon) * dsqrtvar

  #step4
  dsq = 1./N* ones(N,D) * dvar

  #steps3
  dxmu2 = 2 * xmu * dsq

  #step2
  dx1 = (dxmu1 + dxmu2)
  dmu = -1 * sum(dx1)

  #step1
  dx2 = 1./N * ones(N,D) * dmu

  #step0
  dx = dx1 + dx2

endfunction
