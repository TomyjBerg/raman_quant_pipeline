import get_Spectra_data as get_data
import preprocess_method as preprocess
import GA_preprocess_optimization as gapro
import preprocess_DefaultParameter as preprocessparam
import copy 
import numpy as np

folder_path = 'C:\\Users\\thoma\\Desktop\\Master Thesis\\Master\\Mile 3\\221107_Bertho01-CDW3'
delete_experiment = ['0']

spectra_files,file_names = get_data.get_spectra_files(folder_path,delete_experiment)

shift_lim = [450, 1550]

cropped_files = [preprocess.crop_file(f, shift_lim) for f in spectra_files]
#smooth_files = [preprocess.crop_file(f, shift_lim) for f in spectra_files]

#print(cropped_files)

ref_spectra,ref_name = get_data.get_spectra_specific_file(folder_path,['0'])
cropped_ref = [preprocess.crop_file(f, shift_lim) for f in ref_spectra]


same = ['5','6','11','12','13']
rep = 10

ev_test = gapro.evaluate(file_names,cropped_files,same,rep)

#print(ev_test)

alleles_smoothing = preprocessparam.get_default_smoothing()
alleles_baseline = preprocessparam.get_default_baseline()
alleles_normaliztaion = preprocessparam.get_default_normalization()

pop_size = 30
mut_gene = 0.2
mut_all = 0.5
patience = 10

test_GA = gapro.perform_GA_optimization(pop_size,mut_gene,mut_all,patience,cropped_files,file_names,rep,
                           cropped_ref,alleles_smoothing,alleles_baseline,alleles_normaliztaion,same)


