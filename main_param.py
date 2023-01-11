import get_spectra_data as get_data
import preprocess_baseline_methods as basecorrecter
import preprocess_smoothing_methods as smoother
import preprocess_normalization_methods as normalizer
import preprocess_cropping_methods as cropper
import ga_preprocess_optimization as gapro
import preprocess_DefaultParameter as preprocessparam
import ga_baseline_parameter_optimization as gaparam
import copy 
import numpy as np

#folder_path = 'C:\\Users\\thoma\\Desktop\\Master Thesis\\Master\\ACEGlc_data'
#folder_path = 'C:\\Users\\thoma\\Desktop\\Master Thesis\\Master\\Mile2\\glucitamodow'
folder_path = 'C:\\Users\\thoma\\Desktop\\Master Thesis\\Master\\Mile 3\\221107_Bertho01-CDW3'

delete_experiment = []
#delete_experiment = []

spectra_files,file_names = get_data.get_spectra_files(folder_path,delete_experiment)
'''
for i in range(len(spectra_files)):
    print(spectra_files[i][0])
    print(file_names[i])
'''
shift_lim = [500, 1800]

cropped_files = [cropper.crop_file(f, shift_lim) for f in spectra_files]
smooth_files = [smoother.sg_filter(f,2, 49) for f in cropped_files]


#same = ['10','17','18']
same = ['5','6','11','12','13']
#same = ['6','7']
rep = 10


alleles_smoothing = preprocessparam.get_default_smoothing()
alleles_baseline = preprocessparam.get_default_baseline()
alleles_normaliztaion = preprocessparam.get_default_normalization()

pop_size = 30
mut_prob = 0.5
patience_cond = 5
smoothed_data = smooth_files
baseline_parameters = alleles_baseline
same_files = same
replicants = rep
#file_path = 'C:/Users/thoma/Desktop/Master Thesis/Master/ResultParam'
file_path = 'C:/Users/thoma/Desktop/Master Thesis/Master/CDW_PreGA'
 
yep = gaparam.mini_ga_param_loop(pop_size,mut_prob,patience_cond,smoothed_data,file_names,baseline_parameters,same_files,replicants,file_path,verbose=True)
