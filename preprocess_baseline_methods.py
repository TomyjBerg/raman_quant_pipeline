import numpy as np
import pandas as pd
import pybaselines.whittaker as pywh
import pybaselines.polynomial as pypoly
import pybaselines.spline as pyspline
import pybaselines.morphological as pymorph
import pybaselines.smooth as pysmooth
from scipy.signal import savgol_filter



def perform_baseline(smoothed_data,index_baseline_method,alleles_baseline_parameters,verbose):
    """Perform the baseline of the raman spectra using a specific method and parameters).

    Parameters
    ---------- 
 
    smoothed_data : array like of pandas dataframe
        List of the spectra of the raman experiement (pandas dataframe) after cropping 
        and possibly the smoothing 
    
    index_baseline_method : int
        index of the baseline correction methods (2nd gene of the chromosome during GA optimization)

    alleles_baseline_parameters : array like
        list of all the smoothing methods that are a list of parameters / range of parameter
    
    verbose : Bool
        if True : Print baseline correction method indexes

    Returns
    -------
    base_data : array like of pandas dataframe
        List of the spectra of the raman experiement after baseline correction (pandas dataframe)
    """
    param = alleles_baseline_parameters[:-1]
    if verbose:
        print(index_baseline_method)
    if index_baseline_method== 0:
        base_data = [d for d in smoothed_data]
    elif index_baseline_method == 1:
        base_data =  [perform_baseline_correction(d, param[0], param[1]) for d in smoothed_data]
    elif index_baseline_method == 2:
        base_data =  [asls(d, param[0], param[1]) for d in smoothed_data]
    elif index_baseline_method == 3:
        base_data =  [airpls(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 4:
        base_data =  [arpls(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 5:
        base_data =  [aspls(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 6:
        base_data =  [drpls(d, param[0], param[1]) for d in smoothed_data]
    elif index_baseline_method == 7:
        base_data =  [improved_arpls(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 8:
        base_data =  [improved_asls(d,param[0], param[1],param[2]) for d in smoothed_data]
    elif index_baseline_method == 9:
        base_data =  [normal_poly(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 10:
        base_data = [mod_poly(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 11:
        base_data =  [improved_mod_poly(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 12:
        base_data =  [penalized_poly(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 13:
        base_data =  [quantile_poly(d,param[0],param[1]) for d in smoothed_data]
    elif index_baseline_method == 14:
        base_data =  [quantile_spline(d,param[0],param[1],param[2],param[3]) for d in smoothed_data]
    elif index_baseline_method == 15:
        base_data =  [spline_airpls(d,param[0],param[1],param[2]) for d in smoothed_data]
    elif index_baseline_method == 16:
        base_data =  [spline_arpls(d,param[0],param[1],param[2]) for d in smoothed_data]
    elif index_baseline_method == 17:
        base_data =  [spline_asls(d,param[0],param[1],param[2],param[3]) for d in smoothed_data]
    elif index_baseline_method == 18:
        base_data =  [improved_spline_arpls(d,param[0],param[1],param[2]) for d in smoothed_data]
    elif index_baseline_method == 19:
        base_data =  [improved_spline_asls(d,param[0],param[1],param[2],param[3],param[4]) for d in smoothed_data]
    elif index_baseline_method == 20:
        base_data =  [amormol(d,param[0]) for d in smoothed_data]
    else:
        base_data =  [snip(d,param[0]) for d in smoothed_data]

    return base_data


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


def airpls(y,lambda_air,m_order_air=2):
    """Perform baseline correction of data by applying an adaptative iteratively reweighted penalized 
    least squares (airPLS) for baseline fiiting using the airpls function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_air : int
        more lambda_air is high, more the baseline will be smoothed
    m_order_air : int
        The order of the differential matrix. Must be greater than 0. Default is 2

        
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
    ba, param = pywh.airpls(x,lambda_air,m_order_air)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def asls(y,lambda_as,penalty_as,m_order_as=2):
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
    m_order_as : int
        The order of the differential matrix. Must be greater than 0. Default is 2
    
        
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
    ba, param = pywh.asls(x,lambda_as,penalty_as,m_order_as)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def arpls(y,lambda_ar,m_order_ar=2):
    """Perform baseline correction of data by applying an asymetric reweighted penalized least squares (arpls) 
        for baseline fiiting using the arpls function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_ar : float
        The smoothing parameter, Larger values will create smoother baselines.
    m_order_ar : int
        The order of the differential matrix. Must be greater than 0. Default is 2

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

def aspls(y,lambda_aspls,m_order_aspls=2):
    """Perform baseline correction of data by applying an adaptative smoothness penalized least squares(aspls) 
        for baseline fiiting using the aspls function from the pybaseline package

    Parameters
    ----------
    y : pandas dataframe
        [0] = raman_shift 
        [1] = intensity with baseline to be removed
    lambda_aspls : float
        The smoothing parameter, Larger values will create smoother baselines.
    m_order_aspls : int
        The order of the differential matrix. Must be greater than 0. Default is 2
        
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
    ba, param = pywh.aspls(x,lambda_aspls,m_order_aspls)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def drpls(y,lambda_dr,eta_dr,m_order_dr=2):
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
    m_order_dr : int
        The order of the differential matrix. Must be greater than 0. Default is 2  
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
    ba, param = pywh.drpls(x,lambda_dr,eta_dr,m_order_dr)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def improved_arpls(y,lambda_iar,m_order_iar=2,max_iter_iar=20):
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
    m_order_iar : int
        The order of the differential matrix. Must be greater than 0. Default is 2
    max_iter_iar :
        The max number of fit iterations. Default is 20.
    
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
    ba, param = pywh.iarpls(x,lambda_iar,m_order_iar,max_iter_iar)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def improved_asls(y,lambda_ias,penalty_ias,lambdaDer1_ias,max_iter_ias=50):
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
    max_iter_ias :
        The max number of fit iterations. Default is 50.
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


def improved_mod_poly(y,imodpoly_order):
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

def quantile_spline(y,lambda_irsqr,quantile_irsqr,knots_irsqr,spline_deg_irsqr,m_order_irsqr=2):
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

    m_order_irsqr : int
        The order of the differential matrix. Must be greater than 0. Default is 2

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
    ba, param = pyspline.irsqr(x,lambda_irsqr,quantile_irsqr,knots_irsqr,spline_deg_irsqr,m_order_irsqr,x_data = k)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def spline_airpls(y,lambda_spline_air,knots_air,spline_deg_air,m_order_spline_air=2):
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
    
    m_order_spline_air : int
        The order of the differential matrix. Must be greater than 0. Default is 2

    
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
    ba, param = pyspline.pspline_airpls(x,lambda_spline_air,knots_air,spline_deg_air,m_order_spline_air,x_data = k)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def spline_asls(y,lambda_spline_asls,penalty_spline_asls,knots_asls,spline_deg_asls,m_order_spline_asls=2):
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
    
    m_order_spline_asls : int
        The order of the differential matrix. Must be greater than 0. Default is 2
    
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
    ba, param = pyspline.pspline_asls(x,lambda_spline_asls,penalty_spline_asls,knots_asls,spline_deg_asls,m_order_spline_asls,x_data = k)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z


def spline_arpls(y,lambda_spline_ar,knots_ar,spline_deg_ar,m_order_spline_ar=2):
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
    
    m_order_spline_ar : int
        The order of the differential matrix. Must be greater than 0. Default is 2
    
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
    ba, param = pyspline.pspline_arpls(x,lambda_spline_ar,knots_ar,spline_deg_ar,m_order_spline_ar,x_data = k)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def improved_spline_arpls(y,lambda_spline_iar,knots_iar,spline_deg_iar,m_order_spline_iar=2,max_iter_spline_iar=10):
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
    
    m_order_spline_iar : int
        The order of the differential matrix. Must be greater than 0. Default is 2
    
    max_iter_spline_iar :
        The max number of fit iterations. Default is 10.

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
    ba, param = pyspline.pspline_iarpls(x,lambda_spline_iar,knots_iar,spline_deg_iar,m_order_spline_iar,
                                        x_data = k,max_iter =max_iter_spline_iar)
    z = data-ba
    z = pd.DataFrame(z,index = index_y, columns = [1])
    z = pd.concat([line, z],axis =1,ignore_index = False)
    return z

def improved_spline_asls(y,lambda_spline_ias,penalty_spline_ias,lambdaDer1_spline_ias,knots_ias,spline_deg_ias,max_iter_spline_ias=10):
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
    
     max_iter_spline_ias :
        The max number of fit iterations. Default is 10.
    
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
