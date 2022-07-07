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
## @deftypefn {} {@var{retval} =} accuracyCalculator (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: edgar <edgar@DESKTOP-VS5G4BO>
## Created: 2022-07-07

function [accPercentage, correctGuesses, incorrectGuesses, totalGuesses] = accuracyCalculator (n, trainedNetwork, data, arrayLabels)

  % n set of random numbers used to select the data to be tested

  sum = 0;


  for i = 1:length(n)
    np = networkPredictions(data{n(i)}, trainedNetwork);
    r = softMax(np)
    arrayLabels{n(i)}
    [max_valuesR, indexR] = max(r);
    [max_valuesL, indexL] = max(arrayLabels{n(i)});
    if(indexL == indexR)
     sum = sum+1;
    endif

  endfor

    fprintf('number of correct guesses \n')
    correctGuesses = sum
    fprintf('number of incorrect guesses \n')
    incorrectGuesses = length(n)-sum
    fprintf('total number of guesses \n')
    totalGuesses = length(n)

    fprintf('accuracy percentage \n')
    accPercentage = (sum/length(n))*100

endfunction



