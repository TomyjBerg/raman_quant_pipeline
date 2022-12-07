import numpy as np

### Smoothhing ###

def get_default_smoothing():
    """Get all the parameters and the values'range of those for the smoothing methods implemented.

    Parameters
    ----------
    None
    
    Returns
    -------
    Allele_smoothing : array like 
        list of all the smoothing methods that are a list of parameters / range of parameter
    """

    #### No smoothing ####

    no_smoothParam_default = 0
    no_smoothParam_range = range(0,1)

    no_smooth_allele = [no_smoothParam_default,[no_smoothParam_range]]

    #### Whittaker Smoother ####

    whittaker_smoother_lambda_default = 2000
    whittaker_smoother_lambda_range = range(1000,21000,1000)

    whittaker_smoother_degree_default = 3
    whittaker_smoother_degree_range = range(3,7)


    allele_whittaker = [whittaker_smoother_lambda_default,whittaker_smoother_degree_default,
    [whittaker_smoother_lambda_range,whittaker_smoother_degree_range]]


    #### Savitzky Golay Filter ####

    SG_filter_order_default = 2
    SG_filer_order_range = range(2,8)

    SG_filter_window_default = 11
    SG_filter_window_range  = range(9,31,2)

    allele_SG = [SG_filter_order_default,SG_filter_window_default,
    [SG_filer_order_range,SG_filter_window_range]]

    ##

    Alleles_smoothing = [no_smooth_allele,allele_whittaker,allele_SG]

    return Alleles_smoothing


def get_default_baseline():
    """Get all the parameters and the values'range of those for the baseline correction methods implemented.

    Parameters
    ----------
    None
    
    Returns
    -------
    Allele_smoothing : array like 
        list of all the baseline correction methods that are a list of parameters / range of parameter
    """


    #### No baseline ####

    no_baseParam_default = 0
    no_baseParam_range = range(0,1)

    no_base_allele = [no_baseParam_default,[no_baseParam_range]]

    #### Asymetric Whittaker####

    whittaker_base_lambda_default = 580000
    whittaker_base_lambda_range = range(60000,600000,20000)

    penalty_whittaker_default = 0.075
    penalty_whittaker_range = np.arange(0.005,0.080,0.005)
      
    allele_asymetric_whittaker = [whittaker_base_lambda_default,penalty_whittaker_default,
    [whittaker_base_lambda_range,penalty_whittaker_range]]

    #### AirPLS ####

    lambda_air_default = 900000
    lambda_air_range  =range(50000,1050000,50000)

    allele_airpls = [lambda_air_default,[lambda_air_range]]

    #### AsLs ####

    lambda_as_default = 850000
    lambda_as_range  =range(50000,1050000,50000)
    
    penalty_as_default= 0.005
    penalty_as_range = np.arange(0.005,0.105,0.005)
  
    allele_asls = [lambda_as_default,penalty_as_default,[lambda_as_range,penalty_as_range]]

    #### ArPLs ####

    lambda_ar_default= 800000
    lambda_ar_range = range(50000,1050000,50000)

    allele_arpls = [lambda_ar_default,[lambda_ar_range]]

    #### AsPLs ####

    lambda_aspls_default = 65000
    lambda_aspls_range  =range(5000,105000,5000)


    allele_aspls = [lambda_aspls_default,[lambda_aspls_range]]

    #### drPLS ####

    lambda_dr_default =  100000
    lambda_dr_range = range(5000,105000,5000)

    eta_dr_default = 0.3
    eta_dr_range = np.arange(0.3,0.9,0.1)

    allele_drpls = [lambda_dr_default,eta_dr_default,[lambda_dr_range,eta_dr_range]] 

    ####  iarpls ####

    lamda_iar_default = 20000
    lambda_iar_range = range(5000,105000,5000)


    allele_iarpls = [lamda_iar_default,[lambda_iar_range]]

    #### iasls ####
    
    lambda_ias_default = 650000
    lambda_ias_range = range(50000,1050000,50000)

    penalty_ias_default = 4e-2 
    penalty_ias_range = np.arange(1e-3,4.1e-2,1e-3)

    lambdaDer1_ias_default = 1e-3
    lambdaDer1_ias_range = np.arange(1e-4,2.1e-3,1e-4)

    allele_iasls = [lambda_ias_default,penalty_ias_default,lambdaDer1_ias_default,
                    [lambda_ias_range,penalty_ias_range,lambdaDer1_ias_range]]

    ## Normal Polynomial ##

    normpoly_order_default = 3
    normpoly_order_range = range(3,7)

    allele_poly_normal = [normpoly_order_default,[normpoly_order_range]]

    ## Modified Polynomial ##

    modpoly_order_default = 3
    modpoly_order_range = range(3,7)

    allele_poly_modified = [modpoly_order_default,[modpoly_order_range]]

    ## Improve Modified Polynomial ##

    imodpoly_order_default = 3
    imodpoly_order_range = range(3,7)

    allele_improve_poly_modified= [imodpoly_order_default,[imodpoly_order_range]]

    ## Penalized Polynomial ##

    penpoly_order_default = 3
    penpoly_order_range  = range(3,7)

    allele_poly_penalized = [penpoly_order_default,[penpoly_order_range]]

    ## Qunatile Poly ##

    qpoly_order_default = 4
    qpoly_order_range = range(4,8)

    poly_quantile_default = 0.06
    poly_quantile_range = np.arange(0.005,0.105,0.015)
    
    allele_quantile_poly = [qpoly_order_default,poly_quantile_default,[qpoly_order_range,poly_quantile_range]]

    ## Quantile Spline ##

    lambda_irsqr_default = 3850000
    lambda_irsqr_range  =range(50000,1050000,50000)

    quantile_irsqr_default = 0.0025
    quantile_irsqr_range = np.arange(0.005,0.105,0.005)

    knots_irsqr_default = 5
    knots_irsqr_range = range(5,50,5)

    spline_deg_irsqr_default = 2
    spline_deg_irsqr_range = range(1,4)

    allele_quantile_spline=[lambda_irsqr_default,quantile_irsqr_default,knots_irsqr_default,spline_deg_irsqr_default,
                    [lambda_irsqr_range,quantile_irsqr_range,knots_irsqr_range,spline_deg_irsqr_range]]

    ## Spline AirPLS ##
    
    lambda_spline_air_default = 800
    lambda_spline_air_range = range(500,1550,50)

    knots_air_default = 40
    knots_air_range = range(10,110,10)
    
    spline_deg_air_default = 3
    spline_deg_air_range = range(1,4)

    allele_spline_airpls=[lambda_spline_air_default,knots_air_default,spline_deg_air_default,
                        [lambda_spline_air_range,knots_air_range,spline_deg_air_range]]

    ## Spline AsLS ##

    lambda_spline_asls_default = 800
    lambda_spline_asls_range =range(500,1550,50)

    penaly_spline_asls_default = 0.025
    penaly_spline_asls_range = np.arange(0.005,0.205,0.005)

    knots_asls_default = 30
    knots_asls_range = range(10,110,10)

    spline_deg_asls_default = 3
    spline_deg_asls_range = range(1,4)

    allele_spline_asls=[lambda_spline_asls_default,penaly_spline_asls_default,knots_asls_default,spline_deg_asls_default,
                    [lambda_spline_asls_range,penaly_spline_asls_range,knots_asls_range,spline_deg_asls_range]]

    ## Spline ArPLS ##

    lambda_spline_arpls_default = 1400
    lambda_spline_arpls_range = range(500,1550,50)

    knots_arpls_default = 10
    knots_arpls_range = range(5,110,10)

    spline_deg_arpls_default = 1
    spline_deg_arpls_range = range(1,4)

    allele_spline_arpls=[lambda_spline_arpls_default,knots_arpls_default,spline_deg_arpls_default,
                    [lambda_spline_arpls_range,knots_arpls_range,spline_deg_arpls_range]]

    ## Spline improve ArPLS ##

    lambda_spline_iar_default = 1350
    lambda_spline_iar_range  = range(500,1550,50)

    knots_iar_default = 10
    knots_iar_range = range(5,110,5)

    spline_deg_iar_default = 3
    spline_deg_iar_range = range(1,4)

    allele_improve_spline_arpls=[lambda_spline_iar_default,knots_iar_default,spline_deg_iar_default,
                            [lambda_spline_iar_range,knots_iar_range,spline_deg_iar_range]]


    ## Spline improve AsLS ##
    
    lambda_spline_ias_default = 700
    lambda_spline_ias_range  =range(500,1550,50)

    penalty_spline_ias_default = 0.095
    penalty_spline_ias_range = np.arange(0.005,0.205,0.005)

    lambdaDer1_spline_ias_default = 0.01
    lambdaDer1_spline_ias_range = np.arange(0.0005,0.0105,0.0005)

    knots_ias_default = 30
    knots_ias_range = range(10,110,10)

    spline_deg_ias_default = 1
    spline_deg_ias_range = range(1,4)

    allele_improve_spline_asls=[lambda_spline_ias_default,penalty_spline_ias_default,lambdaDer1_spline_ias_default,
                                knots_ias_default,spline_deg_ias_default,[lambda_spline_ias_range,penalty_spline_ias_range,
                                lambdaDer1_spline_ias_range,knots_ias_range,spline_deg_ias_range]]

    ## AmorMol ##

    half_window_amormol_default = 29
    half_window_amormol_range = range(17,51,2)

    allele_amormol = [half_window_amormol_default,[half_window_amormol_range]]

    ## SNIP ##

    max_half_window_snip_default = 29
    max_half_window_snip_range = range(17,51,2)


    allele_snip= [max_half_window_snip_default,[max_half_window_snip_range]]

    Alleles_baseline = [no_base_allele,allele_asymetric_whittaker,allele_asls,allele_airpls,allele_arpls, 
                    allele_aspls, allele_drpls,allele_iarpls, allele_iasls,allele_poly_normal,
                    allele_poly_modified, allele_improve_poly_modified,allele_poly_penalized, 
                    allele_quantile_poly, allele_quantile_spline, allele_spline_airpls, allele_spline_arpls, allele_spline_asls,
                    allele_improve_spline_arpls, allele_improve_spline_asls,allele_amormol, allele_snip]


    return Alleles_baseline

def get_default_normalization():
    """Get all the parameters and the values'range of those for the normalization methods implemented.

    Parameters
    ----------
    None
    
    Returns
    -------
    Allele_smoothing : array like 
        list of all the nromalization methods that are a list of parameters / range of parameter
    """


    #### No normalization ####

    no_normParam_default = 0
    no_normParam_range = range(0,1)

    no_norm_allele = [no_normParam_default,[no_normParam_range]]

    #### MSC ####

    ### SNV ### 
    #No parameter for the moment 

    Alleles_normalization = [no_norm_allele,no_norm_allele,no_norm_allele]

    return Alleles_normalization


    
