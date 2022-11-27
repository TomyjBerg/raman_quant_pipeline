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

def perform_baseline_correction(y, lambda_whittaker, penalty_whittaker):
    """Perform baseline correction of data by applying an asymmetric Whittaker smoother
    based on the publication by Eilers and Boelens (2005) described in Ye et al. (2020).
    
    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_whittaker : float
        parameter labmda
    penalty_whittaker : float
        parameter p. 0 < p < 1
        It is suggested by Ye et al. (2020) that p should lie between 1e-1 and 1e-3.
    degree_whittaker : int
        order of differential matrix D
        
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
    degree_whittaker = 2
    index_y =  y.index
    line = y[0]
    data = y[1]
    y = np.array(y[1])
    m = len(y)
    E = np.eye(m)
    D = np.diff(E, degree_whittaker)


    z = np.linalg.solve((E+ lambda_whittaker*np.dot(D, D.T)), np.dot(E, y))
    w = y-z
    w[w>=0] = penalty_whittaker
    w[w<0] = 1-penalty_whittaker
    W = E*w

    z = np.linalg.solve((W + lambda_whittaker*np.dot(D, D.T)), np.dot(W, y))
    z = data-z
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def airpls(y,lambda_air):
    """Perform baseline correction of data by applying an adaptative iteratively reweighted penalized 
    least squares (airPLS) for baseline fiiting using the airpls function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_air : int
        more lambda_air is high, more the baseline will be smoothed
        
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
    m_order_air = 2
    index_y =  y.index
    line = y[0]
    data = y[1]
    x = np.array(y[1])
    ba, param = pywh.airpls(x,lambda_air,m_order_air)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def asls(y,lambda_as,penalty_as):
    """Perform baseline correction of data by applying an asymetric least squares (asls) 
        for baseline fiiting using the asls function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_as : float
        The smoothing parameter, Larger values will create smoother baselines.
    penalty_as : float (between 0 and 1)
        The penalizing weighting factor. Must be between 0 and 1.
        
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
    m_order_as = 2
    index_y =  y.index
    line = y[0]
    data = y[1]
    x = np.array(y[1])
    ba, param = pywh.asls(x,lambda_as,penalty_as,m_order_as)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def arpls(y,lambda_ar):
    """Perform baseline correction of data by applying an asymetric reweighted penalized least squares (arpls) 
        for baseline fiiting using the arpls function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_ar : float
        The smoothing parameter, Larger values will create smoother baselines.

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
    m_order_ar = 2
    index_y =  y.index
    line = y[0]
    data = y[1]
    x = np.array(y[1])
    ba, param = pywh.arpls(x,lambda_ar,m_order_ar)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def aspls(y,lambda_aspls):
    """Perform baseline correction of data by applying an adaptative smoothness penalized least squares(aspls) 
        for baseline fiiting using the aspls function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_aspls : float
        The smoothing parameter, Larger values will create smoother baselines.
        
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
    m_order_aspls = 2
    index_y =  y.index
    line = y[0]
    data = y[1]
    x = np.array(y[1])
    ba, param = pywh.aspls(x,lambda_aspls,m_order_aspls)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def drpls(y,lambda_dr,eta_dr):
    """Perform baseline correction of data by applying an doubly reweighted penalized least squares (drpls) 
        for baseline fiiting using the drpls function from the pybaseline package

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
    m_order_dr = 2
    index_y =  y.index
    line = y[0]
    data = y[1]
    x = np.array(y[1])
    ba, param = pywh.drpls(x,lambda_dr,eta_dr,m_order_dr)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def improve_arpls(y,lambda_iar):
    """Perform baseline correction of data by applying an improve asymmetrical reweighted 
        penalized least squares (iarpls) for baseline fiiting using the iarpls function 
        from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_iar : float
        The smoothing parameter, Larger values will create smoother baselines..
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
    m_order_iar = 2
    max_iter_iar = 20
    index_y =  y.index
    line = y[0]
    data = y[1]
    x = np.array(y[1])
    ba, param = pywh.iarpls(x,lambda_iar,m_order_iar,max_iter_iar)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def improve_asls(y,lambda_ias,penalty_ias,lambdaDer1_ias):
    """Perform baseline correction of data by applying an improve asymmetrical least squares (iasls) 
        for baseline fiiting using the iasls function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_ias : float
        The smoothing parameter, Larger values will create smoother baselines.
    penalty_ias : 
        The penalizing weighting factor. Must be between 0 and 1.
    lambdaDer1_ias :
        The smoothing parameter for the first derivative of the residual
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
    max_iter_ias = 50
    index_y =  y.index
    line = y[0]
    data = y[1]
    x = np.array(y[1])
    ba, param = pywh.iasls(x,lam=lambda_ias,p=penalty_ias,lam_1=lambdaDer1_ias,max_iter=max_iter_ias)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

#Polynomial Method

def normal_poly(y,normpoly_order):
    """Perform baseline correction of data by computing a polynomial that fits 
    the baseline of the data by the "poly" function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    
    normpoly_order : int
        The polynomial order for fitting the baseline.
        
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
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pypoly.poly(x,k,normpoly_order)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def mod_poly(y,modpoly_order):
    """Perform baseline correction of data by computing a modified polynomial that fits 
    the baseline of the data by the "modpoly" function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    
    modpoly_order : int
        The polynomial order for fitting the baseline.
    
    
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
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pypoly.modpoly(x,k,modpoly_order)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def improve_mod_poly(y,imodpoly_order):
    """Perform baseline correction of data by computing an improved modified polynomial that fits 
    the baseline of the data by the "imodpoly" function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    
    imodpoly_order : int
        The polynomial order for fitting the baseline.
        
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
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pypoly.imodpoly(x,k,imodpoly_order)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def penalized_poly(y,penpoly_order):
    """Perform baseline correction of data by computing a polynomial that fits 
    the baseline using a non-quadratic cost function. This algorithm use the "penalized_poly" function 
    from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    
    penpoly_order : int
        The polynomial order for fitting the baseline.
        
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
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pypoly.penalized_poly(x,k,penpoly_order)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def quantile_poly(y,qpoly_order,poly_quantile):
    """Perform baseline correction of data by computing a polynomial that fits 
    the baseline using a quantile regression. This algorithm use the "quant_reg" function 
    from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    
    qpoly_order : int
        The polynomial order for fitting the baseline.

    poly_quantile : float
        The quantile at which to fit the baseline

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
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pypoly.quant_reg(x,k,qpoly_order,poly_quantile)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


# Spline Method

def quantile_spline(y,lambda_irsqr,quantile_irsqr,knots_irsqr,spline_deg_irsqr):
    """Perform baseline correction of data by applying an Iterative Reweighted Spline 
        Quantile Regression (IRSQR) for the baseline fiiting using the "irsqr" function 
        from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    
    lambda_irsqr: float
        The smoothing parameter. Larger values will create smoother baselines

    quantile_irsqr : float
        The quantile at which to fit the baseline

    knots_irsqr : int
        The number of knots for the spline

    spline_deg_irsqr : int
        The degree of the spline (libear,cubic,quadratic)

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
    m_order_irsqr = 2
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pyspline.irsqr(x,lambda_irsqr,quantile_irsqr,knots_irsqr,spline_deg_irsqr,m_order_irsqr,x_data = k)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def spline_airpls(y,lambda_spline_air,knots_air,spline_deg_air):
    """Perform baseline correction of data by applying a penalized spline version of the
        Adaptive iteratively reweighted penalized least squares (airPLS) for the baseline fiiting 
        using the "pspline_airpls" function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    
    lambda_spline_air: float
        The smoothing parameter. Larger values will create smoother baselines

    knots_air : int
        The number of knots for the spline

    spline_air: int
        The degree of the spline (libear,cubic,quadratic)

    
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
    m_order_spline_air = 2
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pyspline.pspline_airpls(x,lambda_spline_air,knots_air,spline_deg_air,m_order_spline_air,x_data = k)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def spline_asls(y,lambda_spline_asls,penalty_spline_asls,knots_asls,spline_deg_asls):
    """Perform baseline correction of data by applying a penalized spline version of the
        asymmetric least squares(AsLS) algorithm for the baseline fiiting 
        using the "pspline_asls" function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    
    lambda_spline_asls: float
        The smoothing parameter. Larger values will create smoother baselines

    penalty_spline_asls : float
        The penalizing weighting factor. Must be between 0 and 1.

    knots_asls : int
        The number of knots for the spline

    spline_asls: int
        The degree of the spline (libear,cubic,quadratic)
    
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
    m_order_spline_asls = 2
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pyspline.pspline_asls(x,lambda_spline_asls,penalty_spline_asls,knots_asls,spline_deg_asls,m_order_spline_asls,x_data = k)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def spline_arpls(y,lambda_spline_ar,knots_ar,spline_deg_ar):
    """Perform baseline correction of data by applying a penalized spline version of the
        Asymmetrically reweighted penalized least squares smoothing (arPLS) algorithm 
        for the baseline fiiting using the "pspline_arpls" function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    
    lambda_spline_ar: float
        The smoothing parameter. Larger values will create smoother baselines

    knots_ar : int
        The number of knots for the spline

    spline_ar: int
        The degree of the spline (libear,cubic,quadratic)
    
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
    m_order_spline_ar = 2
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pyspline.pspline_arpls(x,lambda_spline_ar,knots_ar,spline_deg_ar,m_order_spline_ar,x_data = k)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def improve_spline_arpls(y,lambda_spline_iar,knots_iar,spline_deg_iar):
    """Perform baseline correction of data by applying a penalized spline version of the
        Improved asymmetrically reweighted penalized least squares smoothing (IarPLS) algorithm 
        for the baseline fiiting using the "pspline_arpls" function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    
    lambda_spline_iar: float
        The smoothing parameter. Larger values will create smoother baselines

    knots_iar : int
        The number of knots for the spline

    spline_iar: int
        The degree of the spline (libear,cubic,quadratic)

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
    m_order_spline_iar = 2
    max_iter_spline_iar = 10
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pyspline.pspline_iarpls(x,lambda_spline_iar,knots_iar,spline_deg_iar,m_order_spline_iar,
                                        x_data = k,max_iter =max_iter_spline_iar)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def improve_spline_asls(y,lambda_spline_ias,penalty_spline_ias,lambdaDer1_spline_ias,knots_ias,spline_deg_ias,
    max_iter_spline_ias):
    """Perform baseline correction of data by applying a penalized spline version of the
        Fits the baseline using the improved asymmetric least squares (IAsLS) algorithm 
        for the baseline fiiting using the "pspline_arpls" function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    
    lambda_spline_ias : float
        The smoothing parameter, Larger values will create smoother baselines.

    penalty_spline_ias : 
        The penalizing weighting factor. Must be between 0 and 1.

    lambdaDer1_spline_ias :
        The smoothing parameter for the first derivative of the residual

    knots_ias : int
        The number of knots for the spline

    spline_deg_ias: int
        The degree of the spline (libear,cubic,quadratic)
    
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
    max_iter_spline_ias = 10
    index_y =  y.index
    line = y[0]
    data = y[1]
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pyspline.pspline_iasls(x,k,lambda_spline_ias,penalty_spline_ias,lambdaDer1_spline_ias,
                                        knots_ias,spline_deg_ias,max_iter =max_iter_spline_ias)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

#Morphological Approach



def amormol(y,half_window_amormol):
    """Perform baseline correction of data by applying an iteratively averaging morphological 
        and mollified (aMorMol) for the baseline fiiting using the "amormol" function 
        from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed

    half_window_amormol : int
        The half-window used for the morphology functions

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
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pymorph.amormol(x,half_window_amormol)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

#Statistic Sensitive method

def snip(y,max_half_window_snip):
    """Perform baseline correction of data by applying a Statistics-sensitive Non-linear 
        Iterative Peak-clipping (SNIP) for the baseline fiiting using the "snip" function 
        from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed

    max_half_window_snip : int
        The maximum number of iterations. Should be set such that max_half_window is approxiamtely 
        (w-1)/2, where w is the index-based width of a feature or peak. 

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
    k = np.array(y[0])
    x = np.array(y[1])
    ba, param = pysmooth.snip(x,max_half_window_snip)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

######## NORMALIZATION #############

def msc(input_datas, replicant,reference=None):
    """Scatter Correction technique performed with mean of the sample data as the reference.

    Parameters
    ----------
    input_datas : Array like of pandas dataframe (y) (order by the name of the sample)
                    [0] = raman_shift 
                    [1] = intensity to be 
        List of the raman spectra dataframe to be normalized
    
    replicants : int
        Number of replication measurement

    reference : panda dataframe (Default is None)
        [0] = raman_shift 
        [1] = intensity
        Reference spectra for the normalization ("BlanK")


    Returns
    -------
    output_data : Array like of pandas dataframe
                    [0] = raman_shift 
                    [1] = intensity normalized
        List of spectra dataframe normalized
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
    """ Perform Normalization technique using the Standard Normal Variate (SNV) algorithm 
        which is done on each individual spectrum, a reference spectrum is not required

    Parameters
    ----------
    input_datas : Array like of pandas dataframe (y) (order by the name of the sample)
                    [0] = raman_shift 
                    [1] = intensity to be 
        List of the raman spectra dataframe to be normalized


    Returns
    -------
    output_data : Array like of pandas dataframe
                    [0] = raman_shift 
                    [1] = intensity normalized
        List of spectra dataframe normalized
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


