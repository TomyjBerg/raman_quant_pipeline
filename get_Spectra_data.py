import os
import pandas as pd
import numpy as np
from os import walk

def get_spectra_files(mpath,delete_):
    """ Get all the spectra data data inside a folder without the unwanted sample

    Parameters
    ----------
    mpath : string
        path of a folder containaing the spectra files

    delete_ : array like (list of string)
        list of the unwanted sample to do not put in the dataset

    Returns
    -------
    spectra_files : array like (list of pandas data frame)
        list of the spectra data of the wanted samples.
    files_names : array like (list of string)
        list of name of wanted spectra 2D files inside a path    
    """
    files_names =list_files_2D(mpath,delete_)
    spectra_files = [pd.read_csv(mpath + '/' + f, sep="\t", header=None, decimal=",") for f in files_names]
    return spectra_files,files_names

def get_spectra_specific_file(mpath,spec):
    """ Get the wanted spectra data inside a folder 

    Parameters
    ----------
    mpath : string
        path of a folder containaing the spectra files

    spec : array like (list of string)
        list of the specific sample to be loaded

    Returns
    -------
    spectra_files : array like (list of pandas data frame)
        list of the spectra data of the specific samples.
    files_names : array like (list of string)
        list of name of wanted spectra 2D files inside a path    
    """
    files_names =list_files_specific(mpath,spec)
    spectra_files = [pd.read_csv(mpath + '/' + f, sep="\t", header=None, decimal=",") for f in files_names]
    return spectra_files,files_names

def list_files(mpath):
    """ List of all the files inside a folder using "walk" function

    Parameters
    ----------
    mpath : string
        path of a folder
  
    Returns
    -------
    f : array like (list of string)
        list of name of file inside a path    
    """
    f = []
    for (_, _, filenames) in walk(mpath):
        f.extend(filenames)
        break
    return(f)


def list_files_2D(mpath,delete_):
    """ List of all the files containing the 2 dimensional sample file inside a folder 
    without the unwanted sample

    Parameters
    ----------
    mpath : string
        path of a folder containaing the spectra files

    delete_ : array like (list of string)
        list of the unwanted sample to be not listed

    Returns
    -------
    f : array like (list of string)
        list of name of wanted spectra 2D files inside a path    
    """
    files_2d = []
    files = list_files(mpath)
  
    for f in files :
        if f.endswith('.csv'):
            sh = f.find('_')
            val = f[0:sh]
            if f.endswith('2D.csv') and val not in delete_:
                files_2d.append(f)
    return(files_2d)

def list_files_specific(mpath,spec):
    """ List of all the files containing the wanted sample file inside a folder
    Parameters
    ----------
    mpath : string
        path of a folder containaing the spectra files

    spec : array like (list of string)
        list of the specific sample to be listed

    Returns
    -------
    f : array like (list of string)
        list of name of teh specific wanted spectra 2D files inside a path    
    """
    files_2d = []
    files = list_files(mpath)
    for f in files :
        if f.endswith('.csv'):
            sh = f.find('_')
            val = f[0:sh]
        if f.endswith('2D.csv') and val in spec:
            files_2d.append(f)
    return(files_2d)

    