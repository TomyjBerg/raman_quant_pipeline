import get_Spectra_data as get_data
import preprocess_method as preprocess

folder_path = 'C:\\Users\\thoma\\Desktop\\Master Thesis\\Master\\Mile 3\\221107_Bertho01-CDW3'
delete_experiment = []

spectra_files = get_data.get_spectra_files(folder_path,delete_experiment)

shift_lim = [300, 1550]

cropped_files_mini_trip = [preprocess.crop_file(f, shift_lim) for f in spectra_files]

print(cropped_files_mini_trip)