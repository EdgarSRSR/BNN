%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

## -*- texinfo -*-
## @deftypefn {} {@var{retval} =} softMax (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

% SOFTMAX compute SOFTMAX function.


function retval = softMax (x)

  denom = sum(exp(x(:)))

  for i = 1:length(x)

    retval(i)=  (exp(x(i))) / denom;

  endfor
  display(retval)
  % if we want to give the biggest values instead of the array
   retval = max(retval);

endfunction
