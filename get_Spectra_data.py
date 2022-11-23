import os
import pandas as pd
import numpy as np
import walk

folder_path = 'C:\Users\thoma\Desktop\Master Thesis\Master\Mile 3\221107_Bertho01-CDW3'
#sample_overview = pd.read_excel(folder_path + "\221107_Bertho01-CDW3", sheet_name="Samples")
delete_experiment = []



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

print('hey')
print(list_files_2D(folder_path,delete_experiment))