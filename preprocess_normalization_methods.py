import numpy as np
import pandas as pd

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


