
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA



def calc_best_fitness(fitness,verbose):
    """Get the chromosome with the best fitness value and its index inside the population.

    Parameters
    ----------
    fitness : array like (list of float)
        List of fitness of each chromosome inside the population
        (ratio of the variances inter/intra sample (b/w) related to the defined preprocessing method 
        defined by each chromosome inside the population).
    
    Returns
    -------
    best_fit : float
        Best chromosome's fitness value in the population
    
    iter_stock : int
        Index_of the chromosome with the best fitness inside the population

    verbose : Bool
        if True : Print the best fitness of the population
    """
    best_fit = 0
    iter_stock = 0
    for i in range(len(fitness)):
        if fitness[i] > best_fit:
            iter_stock = i
            best_fit = fitness[i]
    if verbose:
        print(best_fit)
    return best_fit,iter_stock



def evaluate(files_names,files,same_sample,replicants):
    """Calculate the ratio b/w (variance between each sample (Variance inter Sample) over
        the variance within each samples (Variance intra sample)).

    Parameters
    ----------

    file_names : array of string
        List of all the names corresponding to the raman spectra files 
 
    files : array like of pandas dataframe
        List of the spectra of the raman experiement (pandas dataframe)
    
    same_sample : array of string
        List of all the sample number (string) to ignore during the calcualtion
        i.e. Differents sample with the same parameter of the experiment (DOE middle triplicants)

    replicants : int
        Number of replicants for each sample.
    Returns
    -------
    ratio : float
        ratio of the variances inter/intra sample (b/w)
    """
    w = calc_intra_sample_variance(files_names,files)
    b = calc_inter_variance_sample(files_names,files,same_sample,replicants)
    
    ratio = b/w
    return ratio


def calc_intra_sample_variance(files_names,files):
    """Calculate the variance within each replication (Variance intra Sample).

    Parameters
    ----------

    file_names : array of string
        List of all the names corresponding to the raman spectra files 
 
    files : array like of pandas dataframe
        List of the spectra of the raman experiement (pandas dataframe)
    
    Returns
    -------
    w : float
        Variance Intra Sample
        i.e. Square of the (Sum of standard deviation of the 3 Principal Components over the replicants of each sample / 
        number of sample in total)
    """

    name_trip = []
    for f in files_names: 
        sh = f.find('_')
        val = f[0:sh]
        name_trip.append(val)
    name_trip = list(set(name_trip))
    std = 0

    for name in name_trip:
        data = []
        for i in range(len(files_names)):
            sh = files_names[i].find('_')
            val = files_names[i][0:sh]
            if val == name:
                data.append(files[i])
        for k in range(len(data)):
            if k==0:
                df = data[k].transpose()
                df.columns = df.iloc[0] 
                df = df[1:]
            else:
                new_df = data[k].transpose()
                new_df.columns = new_df.iloc[0] 
                new_df = new_df[1:]
                df = pd.concat([df, new_df], ignore_index = True)
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(df)
        principalDf = pd.DataFrame(data = principalComponents
                    , columns = ['principal component 1', 'principal component 2','principal component 3'])
        std = std + principalDf.std()['principal component 1'] + principalDf.std()['principal component 2'] + principalDf.std()['principal component 3']

    return (std/len(name_trip))**2
      
def calc_inter_variance_sample(files_names,files,same_sample,replicants):

    """Calculate the variance within each sample (Variance inter Sample).

    Parameters
    ----------

    file_names : array of string
        List of all the names corresponding to the raman spectra files 
 
    files : array like of pandas dataframe
        List of the spectra of the raman experiement (pandas dataframe)
    
    same_sample : array of string
        List of all the sample number (string) to ignore during the calcualtion
        i.e. Differents sample with the same parameter of the experiment (DOE middle triplicants)

    replicants : int
        Number of replicants for each sample.
    Returns
    -------
    w : float
        Variance Inter Sample
        i.e. Square of the (Sum of standard deviation of the 3 Principal Components over the samples for each replicants/ 
        number of replicants)
    """
    name_trip = []
    for f in files_names: 
        sh = f.find('_')
        val = f[0:sh]
        name_trip.append(val)
    name_trip = list(set(name_trip))
    name_removed = []
    for name in (name_trip):
        if name in same_sample:
            name_removed.append(name)
    for name in name_removed:
        name_trip.remove(name)
    
    data_rep = [[] for l in range(replicants)]
    for name in name_trip:
        num = 0
        for i in range(len(files_names)):
            sh = files_names[i].find('_')
            val = files_names[i][0:sh]
            if val == name:
                data_rep[num].append(files[i])
                num = num + 1
    std = 0
    for data in data_rep:
        for k in range(len(data)):
            if k==0:
                df = data[k].transpose()
                df.columns = df.iloc[0] 
                df = df[1:]
            else:
                new_df = data[k].transpose()
                new_df.columns = new_df.iloc[0] 
                new_df = new_df[1:]
                df = pd.concat([df, new_df], ignore_index = True)

        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(df)
        principalDf = pd.DataFrame(data = principalComponents
                    , columns = ['principal component 1', 'principal component 2','principal component 3'])
        std = std + principalDf.std()['principal component 1'] + principalDf.std()['principal component 2'] + principalDf.std()['principal component 3']
  
    return (std/3)**2
      