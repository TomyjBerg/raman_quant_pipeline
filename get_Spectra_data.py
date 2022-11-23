import os
import pandas as pd
import numpy as np
from os import walk


def list_files(mpath):
  f = []
  for (dirpath, dirnames, filenames) in walk(mpath):
      f.extend(filenames)
      break
  return(f)


def list_files_2D(mpath,delete_):
  files_2d = []
  files = list_files(mpath)
  for f in files :
    sh = f.find('_')
    val = f[0:sh]
    if f[-6] == '2' and val not in delete_:
      files_2d.append(f)
  return(files_2d)

def get_spectra_files(mpath,delete):
    files_names =list_files_2D(mpath,delete)
    spectra_files = [pd.read_csv(mpath + '/' + f, sep="\t", header=None, decimal=",") for f in files_names]
    return spectra_files
