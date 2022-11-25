### Smoothhing ###

'''
allele_snip= [window,[range_window]]
allele_amormol = [window,[range_window]]
allele_improve_spline_asls=[lam,p,lam1,knots,spline_deg,m_order,[range_lam,range_p,range_lam1,range_knots,range_spline_deg,range_m_order]]
allele_improve_spline_arpls=[lam,knots,spline_deg,m_order,max_iter,[range_lam,range_knots,range_spline_deg,range_m_order,range_max_iter]]
allele_spline_arpls=[lam,knots,spline_deg,m_order,[range_lam,range_knots,range_spline_deg,range_m_order]]
allele_spline_asls=[lam,p,knots,spline_deg,m_order,[range_lam,range_p,range_knots,range_spline_deg,range_m_order]]
allele_spline_airpls=[lam,knots,spline_deg,m_order,[range_lam,range_knots,range_spline_deg,range_m_order]]
allele_quantile_spline=[lam,q,knots,spline_deg,m_order,[range_lam,range_q,range_knots,range_spline_deg,range_m_order]]
allele_quantile_poly = [p_order,q,[range_p_order,range_q]]
allele_poly_penalized = [p_order,[range_p_order]]
allele_improve_poly_modified= [p_order,[range_p_order]]
allele_poly_modified = [p_order,[range_p_order]]
allele_poly_normal = [p_order,[range_p_order]]
allele_improve_perf_base = [lam,lam1,p,max_iter,[range_lam,range_lam1,range_p,range_max_iter]]
allele_iarpls = [lam,m_order,[range_lam,range_m_order]]
allele_drpls = [lam,eta,m_order,[range_lam,range_eta,range_m_order]] 
allele_aspls = [lam,m_order,[range_lam,range_m_order]] 
allele_arpls = [lam,m_order,[range_lam,range_m_order]]
allele_asls = [lam,p,m_order,[range_lam,range_p,range_m_order]]
allele_airpls = [lam,m_order,[range_lam,range_m_order]]
allele_improve_perf_base = [lam,lam1,p,max_iter,[range_lam,range_lam1,range_p,range_max_iter]]
allele_perf_base = [d0,lambda0,p0,[range_d0,range_lamda0,range_p0]]
'''

def get_default_smoothing():

    #### No smoothing ####

    no_smoothParam_default = 0
    no_smoothParam_range = range(0,1)

    no_smooth_allele = [no_smoothParam_default,[no_smoothParam_range]]

    #### Whittaker Smoother ####

    whittaker_smoother_lambda_default = 2000
    whittaker_smoother_lambda_range = range(1000,11000,1000)

    whittaker_smoother_degree_default = 3
    whittaker_smoother_degree_range = range(3,7)


    allele_whittaker = whittaker_smoother_lambda_default,whittaker_smoother_degree_default,
    [whittaker_smoother_lambda_range,whittaker_smoother_degree_range]

    #### Savitzky Golay Filter ####

    SG_filter_order_default = 2
    SG_filer_order_range = range(2,8)

    SG_filter_window_default = 11
    SG_filter_window_range  = range(9,31,2)

    allele_SG = SG_filter_order_default,SG_filter_window_default,
    [SG_filer_order_range,SG_filter_window_range]

    ##

    Alleles_smoothing = [no_smooth_allele,allele_whittaker,allele_SG]

    return Alleles_smoothing



    
