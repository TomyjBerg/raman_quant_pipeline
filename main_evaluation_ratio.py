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
#delete_experiment= [0]
delete_experiment= []
spectra_files,file_names = get_data.get_spectra_files(folder_path,delete_experiment)

shift_lim_2 = [500, 1800]
shift_lim = [450, 1550]

cropped_files = [cropper.crop_file(f, shift_lim) for f in spectra_files]
cropped_files_2 = [cropper.crop_file(f, shift_lim_2) for f in spectra_files]
ref_spectra,ref_name = get_data.get_spectra_specific_file(folder_path,['0'])
cropped_ref = [cropper.crop_file(f, shift_lim) for f in ref_spectra]
smooth_files = [smoother.whittaker_smoother(f,1000, 3) for f in cropped_files]
#smooth_files = [smoother.sg_filter(f,2, 33) for f in cropped_files]
smooth_files_2 = [smoother.sg_filter(f,2, 63) for f in cropped_files_2]
bc_files = [basecorrecter.improved_asls(f,10,0.0001,0.002)for f in smooth_files]
#bc_files = [basecorrecter.spline_arpls(f,500,35,3)for f in smooth_files]
bc_files_2 = [basecorrecter.spline_arpls(f,550,5,2)for f in smooth_files_2]
#bc_files_2 = [basecorrecter.amormol(f,23)for f in smooth_files_2]
#bc_files_2 = [basecorrecter.penalized_poly(f,3)for f in smooth_files_2]

bc_files_2=normalizer.snv(bc_files_2)


#cropped_ref = []

same = ['5','6','11','12','13']
#same = ['6','7']
#same = ['10','17','18','19B','19C','20B','20C','21B','21C','22B','22C','23B','23C','24B','24C']
rep = 10

ev_test = gapro.evaluate(file_names,spectra_files,same,rep)
ev_test_up = gapro.evaluate(file_names,bc_files,same,rep)
ev_test_pro = gapro.evaluate(file_names,bc_files_2,same,rep)

print(ev_test)
print(ev_test_up)
print(ev_test_pro)

