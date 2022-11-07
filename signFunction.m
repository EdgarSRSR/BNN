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
  elseif (z == 0)
    retval = 0;
  elseif (z < 0)
    retval = -1;
  endif

endfunction
