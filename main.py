import get_Spectra_data as get_data
import preprocess_method as preprocess
import GA_preprocess_optimization as gapro
import copy 
import numpy as np

folder_path = 'C:\\Users\\thoma\\Desktop\\Master Thesis\\Master\\Mile 3\\221107_Bertho01-CDW3'
delete_experiment = ['0']

spectra_files,file_names = get_data.get_spectra_files(folder_path,delete_experiment)

shift_lim = [450, 1550]

cropped_files = [preprocess.crop_file(f, shift_lim) for f in spectra_files]
smooth_files = [preprocess.crop_file(f, shift_lim) for f in spectra_files]

#print(cropped_files)


same = ['5','6','11','12','13']
rep = 10

ev_test = gapro.evaluate(file_names,smooth_files,same,rep)

print(ev_test)

