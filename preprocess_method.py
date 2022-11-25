import numpy as np
import pandas as pd
import pybaselines.whittaker as pywh
import pybaselines.polynomial as pypoly
import pybaselines.spline as pyspline
import pybaselines.morphological as pymorph
import pybaselines.smooth as pysmooth
from scipy.signal import savgol_filter

##### CROPPING #####

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



##### SMOOTHING #####

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


#####  BASELINE CORRECTION #######

# Asymetric Penalised least Square method

def perform_baseline_correction(y, lambda_whittaker, penalty, degree, epsilon=1e-5):
    """Perform baseline correction of data by applying an asymmetric Whittaker smoother
    based on the publication by Eilers and Boelens (2005) described in Ye et al. (2020).
    
    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_whittaker : float
        parameter labmda
    penalty : float
        parameter p. 0 < p < 1
        It is suggested by Ye et al. (2020) that p should lie between 1e-1 and 1e-3.
    degree : int
        order of differential matrix D
    epsilon: float, default 1e-5
        value of residuals
        
    Returns
    -------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity without baseline
        
    References
    ----------
    Eilers, P.H. and Boelens, H.F., 2005. Baseline correction with asymmetric 
      least squares smoothing. Leiden University Medical Centre Report, 1(1), 
      p.5.
    Ye, J., Tian, Z., Wei, H. and Li, Y., 2020. Baseline correction method 
      based on improved asymmetrically reweighted penalized least squares for 
      the Raman spectrum. Applied Optics, 59(34), pp.10933-10943.
    He, S., Zhang, W., Liu, L., Huang, Y., He, J., Xie, W., Wu, P. and Du, C., 
      2014. Baseline correction for Raman spectra using an improved asymmetric 
      least squares method. Analytical Methods, 6(12), pp.4402-4407.
    """
    index_y =  y.index
    line = y[0]
    data = y[1]
    y = np.array(y[1])
    m = len(y)
    E = np.eye(m)
    D = np.diff(E, degree)


    z = np.linalg.solve((E+ lambda_whittaker*np.dot(D, D.T)), np.dot(E, y))
    w = y-z
    w[w>=0] = penalty
    w[w<0] = 1-penalty
    W = E*w

    z = np.linalg.solve((W + lambda_whittaker*np.dot(D, D.T)), np.dot(W, y))
    z = data-z
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def airpls(y,lambda_air,m_order):
    """Perform baseline correction of data by applying an adaptative iteratively reweighted penalized 
    least squares (airPLS) for baseline fiiting using the airpls function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_air : int
        more lambda_air is high, more the baseline will be smoothed
    m_order : int
        order of the difference of penalties
        
    Returns
    -------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity without baseline
        
    References
    ----------
    Martyna et al. 2020. Improving discrimation of Raman spectra by optimising 
        preprocessing strategies of the basis of the ability to refine the 
        relationship between variance components
    """
    index_y =  y.index
    line = y[0]
    data = y[1]
    x = np.array(y[1])
    ba, param = pywh.airpls(x,lambda_air,m_order)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def asls(y,lamda_as,penalty_as,m_order_as):
    """Perform baseline correction of data by applying an asymetric least squares (asls) 
        for baseline fiiting using the airpls function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_as : float
        The smoothing parameter, Larger values will create smoother baselines.
    penalty_as : float (between 0 and 1)
        The penalizing weighting factor. Must be between 0 and 1.
    m_order_as : int
        The order of the differential matrix
        
    Returns
    -------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity without baseline
        
    References
    ----------
    Martyna et al. 2020. Improving discrimation of Raman spectra by optimising 
        preprocessing strategies of the basis of the ability to refine the 
        relationship between variance components

    pybaseline package litterature : https://pybaselines.readthedocs.io/
    """
    index_y =  y.index
    line = y[0]
    data = y[1]
    x = np.array(y[1])
    ba, param = pywh.asls(x,lamda_as,penalty_as,m_order_as)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def arpls(y,lambda_ar,m_order_ar):
    """Perform baseline correction of data by applying an asymetric reweighted penalized least squares (asls) 
        for baseline fiiting using the airpls function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_ar : float
        The smoothing parameter, Larger values will create smoother baselines.
    m_order_ar : int
        The order of the differential matrix
        
    Returns
    -------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity without baseline
        
    References
    ----------
    Martyna et al. 2020. Improving discrimation of Raman spectra by optimising 
        preprocessing strategies of the basis of the ability to refine the 
        relationship between variance components

    pybaseline package litterature : https://pybaselines.readthedocs.io/
    """
    index_y =  y.index
    line = y[0]
    data = y[1]
    x = np.array(y[1])
    ba, param = pywh.arpls(x,lambda_ar,m_order_ar)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def aspls(y,lamda_aspls,m_order_aspls):
    """Perform baseline correction of data by applying an adaptative smoothness penalized least squares(aspls) 
        for baseline fiiting using the airpls function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_aspls : float
        The smoothing parameter, Larger values will create smoother baselines.
    m_order_aspls : int
        The order of the differential matrix
        
    Returns
    -------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity without baseline
        
    References
    ----------
    Martyna et al. 2020. Improving discrimation of Raman spectra by optimising 
        preprocessing strategies of the basis of the ability to refine the 
        relationship between variance components

    pybaseline package litterature : https://pybaselines.readthedocs.io/
    """
    index_y =  y.index
    line = y[0]
    data = y[1]
    x = np.array(y[1])
    ba, param = pywh.aspls(x,lamda_aspls,m_order_aspls)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def drpls(y,lamda_dr,eta_dr,m_order_dr):
    """Perform baseline correction of data by applying an doubly reweighted penalized least squares (drpls) 
        for baseline fiiting using the airpls function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_dr : float
        The smoothing parameter, Larger values will create smoother baselines.
    eta_dr : float (between 0 and 1)
        A term for controlling the value of lamda_dr; should be between 0 and 1. 
        Low values will produce smoother baselines, while higher values will more aggressively fit peaks.
    m_order_dr : int
        The order of the differential matrix
        
    Returns
    -------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity without baseline
        
    References
    ----------
    Martyna et al. 2020. Improving discrimation of Raman spectra by optimising 
        preprocessing strategies of the basis of the ability to refine the 
        relationship between variance components

    pybaseline package litterature : https://pybaselines.readthedocs.io/
    """
    index_y =  y.index
    line = y[0]
    data = y[1]
    x = np.array(y[1])
    ba, param = pywh.drpls(x,lamda_dr,eta_dr,m_order_dr)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def improve_arpls(y,lamda_iar,m_order_iar,max_iter_iar):
    """Perform baseline correction of data by applying an improve asymmetrical reweighted 
        penalized least squares (iarpls) for baseline fiiting using the airpls function 
        from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_iar : float
        The smoothing parameter, Larger values will create smoother baselines..
    m_order_iar : int
        The order of the differential matrix
    max_iter_iar : int
        Number of max fit iteration
        
    Returns
    -------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity without baseline
        
    References
    ----------
    Martyna et al. 2020. Improving discrimation of Raman spectra by optimising 
        preprocessing strategies of the basis of the ability to refine the 
        relationship between variance components

    pybaseline package litterature : https://pybaselines.readthedocs.io/
    """
    index_y =  y.index
    line = y[0]
    data = y[1]
    x = np.array(y[1])
    ba, param = pywh.iarpls(x,lamda_iar,m_order_iar,max_iter_iar)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def improve_asls(y,lamda_ias,penalty_ias,lamdaDer1_ias,max_iter_ias):
    """Perform baseline correction of data by applying an improve asymmetrical least squares (iasls) 
        for baseline fiiting using the airpls function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_ias : float
        The smoothing parameter, Larger values will create smoother baselines.
    penalty_ias : 
        The penalizing weighting factor. Must be between 0 and 1.
    lamdaDer1_ias :
        The smoothing parameter for the first derivative of the residual
    max_iter_ias : int
        Number of max fit iteration
        
    Returns
    -------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity without baseline
        
    References
    ----------
    Martyna et al. 2020. Improving discrimation of Raman spectra by optimising 
        preprocessing strategies of the basis of the ability to refine the 
        relationship between variance components

    pybaseline package litterature : https://pybaselines.readthedocs.io/
    """

    index_y =  y.index
    line = y[0]
    data = y[1]
    x = np.array(y[1])
    ba, param = pywh.iasls(x,lam=lamda_ias,p=penalty_ias,lam_1=lamda_ias,max_iter=max_iter_ias)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

#Polynomial Method

def normal_poly(y,p_order):
    """Apply polynimoal basline correction.
    
    Parameters
    ----------
    y : pandas DataFrame
    p_order: int

    Returns
    -------
    z : pandas DataFrame
        Baseline corrected data.
    """
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pypoly.poly(x,k,p_order)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def mod_poly(y,p_order):
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pypoly.modpoly(x,k,p_order)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def improve_mod_poly(y,p_order):
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pypoly.imodpoly(x,k,p_order)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def penalized_poly(y,p_order):
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pypoly.penalized_poly(x,k,p_order)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def quantile_poly(y,p_order,q):
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pypoly.quant_reg(x,k,p_order,q)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


#Spline Method

def quantile_spline(y,lam,q,knots,spline_deg,m_order):
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pyspline.irsqr(x,lam,q,knots,spline_deg,m_order,x_data = k)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def spline_airpls(y,lam,knots,spline_deg,m_order):
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pyspline.pspline_airpls(x,lam,knots,spline_deg,m_order,x_data = k)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def spline_asls(y,lam,p,knots,spline_deg,m_order):
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pyspline.pspline_asls(x,lam,p,knots,spline_deg,m_order,x_data = k)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def spline_arpls(y,lam,knots,spline_deg,m_order):
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pyspline.pspline_arpls(x,lam,knots,spline_deg,m_order,x_data = k)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def improve_spline_arpls(y,lam,knots,spline_deg,m_order,max_iter):
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pyspline.pspline_iarpls(x,lam,knots,spline_deg,m_order,x_data = k,max_iter =max_iter)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def improve_spline_asls(y,lam,p,lam1,knots,spline_deg,max_iter):
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pyspline.pspline_iasls(x,k,lam,p,lam1,knots,spline_deg,max_iter =max_iter)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

#Morphological Approach

def amormol(y,window):
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pymorph.amormol(x,window)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

#Statistic Sensitive method

def snip(y,window):
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pysmooth.snip(x,window)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

######## NORMALIZATION #############

def msc(input_datas, replicant,reference=None):
    """
        :msc: Scatter Correction technique performed with mean of the sample data as the reference.
        :param input_data: Array of spectral data
        :type input_data: DataFrame
        :returns: data_msc (ndarray): Scatter corrected spectra data
    """
    comb = []
    input_data = []
    index_y = input_datas[0].index

    if reference is not None:
      input_ref = []
      for i in range(len(reference)):
        input_ref.append(reference[i][1])
      input_ref = np.array(input_ref, dtype=np.float64)


  
    for i in range(len(input_datas)):
        comb.append(input_datas[i][0])
        input_data.append(input_datas[i][1])
        
    eps = np.finfo(np.float32).eps
    input_data = np.array(input_data, dtype=np.float64)
    ref = []
    sampleCount = int(len(input_data))

    # Get the reference spectrum. If not given, estimate it from the mean
    # Define a new array and populate it with the corrected data    
    data_msc = np.zeros_like(input_data)
    
    for i in range(input_data.shape[0]):
      if reference is None:
        if i%replicant == 0 and i != input_data.shape[0]-1:
          ref = np.mean(input_data[i:i+replicant], axis=0)
      else:
        ref = np.mean(input_ref[0:len(input_ref)],axis=0)
        # Run regression
      fit = np.polyfit(ref, input_data[i,:], 1, full=True)
      # Apply correction
      data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0]
    data_fin_order = []
    for k in range(len(data_msc)):
      z = pd.DataFrame(data_msc[k],index = index_y , columns = [1])
      z = pd.concat([comb[k], z],axis =1,ignore_index = False)
      data_fin_order.append(z)


    return (data_fin_order)


def snv(input_datas):
    """
        :snv: A correction technique which is done on each
        individual spectrum, a reference spectrum is not
        required
        :param input_data: Array of spectral data
        :type input_data: DataFrame
        
        :returns: data_snv (ndarray): Scatter corrected spectra
    """

    comb = []
    input_data = []
    index_y = input_datas[0].index
  
    for i in range(len(input_datas)):
        comb.append(input_datas[i][0])
        input_data.append(input_datas[i][1])
    
    input_data = np.asarray(input_data)
    
    # Define a new array and populate it with the corrected data  
    data_snv = np.zeros_like(input_data)
    for i in range(data_snv.shape[0]):
    # Apply correction
        data_snv[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
    
    data_fin_order = []
    for k in range(len(data_snv)):
        z = pd.DataFrame(data_snv[k],index = index_y , columns = [1])
        z = pd.concat([comb[k], z],axis =1,ignore_index = False)
        data_fin_order.append(z)    
    return (data_fin_order)


