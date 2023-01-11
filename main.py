import get_spectra_data as get_data
import preprocess_baseline_methods as basecorrecter
import preprocess_smoothing_methods as smoother
import preprocess_normalization_methods as normalizer
import preprocess_cropping_methods as cropper
import ga_preprocess_optimization as gapro
import preprocess_DefaultParameter as preprocessparam
import copy 
import numpy as np

folder_path = 'C:\\Users\\thoma\\Desktop\\Master Thesis\\Master\\Mile 3\\221107_Bertho01-CDW3'
#folder_path = 'C:\\Users\\thoma\\Desktop\\Master Thesis\\Master\\ACEGlc_data'
#folder_path = 'C:\\Users\\thoma\\Desktop\\Master Thesis\\Master\\Mile2\\glucitamodow'
#delete_experiment = ['01']
delete_experiment= []
spectra_files,file_names = get_data.get_spectra_files(folder_path,delete_experiment)

shift_lim = [500, 1800]

cropped_files = [cropper.crop_file(f, shift_lim) for f in spectra_files]
#smooth_files = [preprocess.crop_file(f, shift_lim) for f in spectra_files]

#print(cropped_files)

ref_spectra,ref_name = get_data.get_spectra_specific_file(folder_path,['0'])
cropped_ref = [cropper.crop_file(f, shift_lim) for f in ref_spectra]
#cropped_ref = []

same = ['5','6','11','12','13']
#same = ['10','17','18','19B','19C','20B','20C','21B','21C','22B','22C','23B','23C','24B','24C']
#same = ['6','7']
rep = 10

ev_test = gapro.evaluate(file_names,cropped_files,same,rep)

#print(ev_test)

alleles_smoothing = preprocessparam.get_default_smoothing()
alleles_baseline = preprocessparam.get_default_baseline()
alleles_normaliztaion = preprocessparam.get_default_normalization()

pop_size = 30
mut_gene = 0.5
mut_all = 0.6
patience = 10

fit,prepross = gapro.perform_GA_optimization(pop_size,mut_gene,mut_all,patience,cropped_files,file_names,rep,
                           cropped_ref,alleles_smoothing,alleles_baseline,alleles_normaliztaion,same,verbose=True)

#files_text_path = 'C:\\Users\\thoma\\Desktop\\Master Thesis\\Master\\ACEGlc_data'

with open(folder_path+'_'+'result_3'+'.txt','w') as f:
    f.write(str(fit))
    f.write('\n')
    f.write('\n')
    for list in prepross[0]:
        f.write(str(list))
        f.write('\n')
    f.write(str(prepross))