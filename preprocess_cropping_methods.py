import numpy as np
import pandas as pd

def crop_file(y, shift_lim):
    
    """Crop the raman shift window.
    
    Parameters
    ----------
    y : pandas dataframe
        [0] = raman shift to be cropped 
        [1] = raman intensity
    shift_lim : array_like (dim 2)
        0 : lower bond (cropped_window)
        1 : upper bond (cropped_window)
    
    Returns
    -------
    z : pandas Dataframe
        [0] cropped raman shift
        [1] raman intensity
        
    References
    ----------
    Eilers, P.H., 2003. A perfect smoother. Analytical chemistry, 75(14), pp.3631-3636.
    """   
    cropped_f = y[y[0]>=shift_lim[0]]
    cropped_f = cropped_f[cropped_f[0]<=shift_lim[1]]

    return cropped_f