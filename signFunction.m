%===============================================================================
%
%   Author: Edgar Solis Romeu
%
%===============================================================================

% SIGN compute sign function.

function retval = signFunction (z)

  retval = 0;

  if (z > 0)
    retval = 1;
  else
    retval = 0;
  endif

endfunction
