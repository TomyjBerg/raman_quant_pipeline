import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def perform_smoothing(cropped_files,index_smoothing_method,alleles_smoothing_parameters):
    """Perform the smoothing of the cropped raman spectra using a specific method and parameters).

    Parameters
    ---------- 
 
    cropped_files : array like of pandas dataframe
        List of the spectra of the raman experiement after cropping (pandas dataframe)
    
    index_smoothing_method : int
        index of the smoothing methods (1rt gene of the chromosome during GA optimization)

    alleles_smoothing_parameters : array like
        list of all the smoothing methods that are a list of parameters / range of parameter

    Returns
    -------
    smoothed_data : array like of pandas dataframe
        List of the spectra of the raman experiement after smoothing (pandas dataframe)
    """
    param = alleles_smoothing_parameters[:-1]
    if index_smoothing_method  == 1:
        smoothed_data =  [whittaker_smoother(d, param[0], param[1]) for d in cropped_files]
    elif index_smoothing_method == 2:
        smoothed_data =  [sg_filter(d,param[0],param[1]) for d in cropped_files]
    else: 
        smoothed_data =  [d for d in cropped_files]
    return smoothed_data



def whittaker_smoother(y, lambda_whittaker, degree_whittaker):
    """Smooth data according to Whittaker.
    
    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity to be smoothed
    lambda_whittaker : float
        smoothing parameter lambda
    degree_whittaker : int
        order of the penalty
    
    Returns
    -------
    z : pandas Dataframe
        [0] raman shift
        [1] smoothed intensity
        
    References
    ----------
    Eilers, P.H., 2003. A perfect smoother. Analytical chemistry, 75(14), pp.3631-3636.
    """   
    index_y =  y.index
    line = y[0]
    y = np.array(y[1])
    m = len(y)
    
    E = np.eye(m)
    D = np.diff(E, degree_whittaker)
    z = np.linalg.solve((E + lambda_whittaker*np.dot(D, D.T)),y)
    z = pd.DataFrame(z,index = index_y , columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)

    return z



def sg_filter(y, order_SG, window_SG):
    """
    Smooth data according to Savitzky Golay, using the sav_gol filter from the scipy library.
    
    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity to be smoothed
    
    order_SG : int
        order of the polynomial to be fit
    window_SG : int (odd)
        window where the fit can perform
    
    Returns
    -------
    z : pandas Dataframe
        [0] raman shift
        [1] smoothed intensity
        
    """
       
    index_y =  y.index
    line = y[0]
    y = np.array(y[1])
    z = savgol_filter(y,window_SG,order_SG)
    z = pd.DataFrame(z,index = index_y, columns = [1] )
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

