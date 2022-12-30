import get_spectra_data as get_data
import preprocess_baseline_methods as basecorrecter
import preprocess_smoothing_methods as smoother
import preprocess_normalization_methods as normalizer
import preprocess_cropping_methods as cropper
import ga_preprocess_optimization as gapro
import preprocess_DefaultParameter as preprocessparam
import copy 
import numpy as np

#folder_path = 'C:\\Users\\thoma\\Desktop\\Master Thesis\\Master\\Mile 3\\221107_Bertho01-CDW3'
#folder_path = 'C:\\Users\\thoma\\Desktop\\Master Thesis\\Master\\ACEGlc_data'
folder_path = 'C:\\Users\\thoma\\Desktop\\Master Thesis\\Master\\Mile2\\glucitamodow'
#delete_experiment = ['01']
delete_experiment= ['0']
#delete_experiment= []
spectra_files,file_names = get_data.get_spectra_files(folder_path,delete_experiment)

#shift_lim = [500, 1800]
shift_lim = [450, 1500]

cropped_files = [cropper.crop_file(f, shift_lim) for f in spectra_files]
ref_spectra,ref_name = get_data.get_spectra_specific_file(folder_path,['0'])
cropped_ref = [cropper.crop_file(f, shift_lim) for f in ref_spectra]
#cropped_ref = []

#same = ['5','6','11','12','13']
#same = ['6','7']
same = ['10','17','18']
rep = 15

ev_test = gapro.evaluate(file_names,spectra_files,same,rep)

print(ev_test)

