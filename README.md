# process_spectra
The folder contains the code used for the Master Thesis "Advanced Spectroscopic Analysis and Data
Exploitation Concepts in Bioprocessing" by Thomas Berger. In this Master Thesis, the TG Raman spectrometry were used in order to predict the concentration of molecules of interest in the sample.

This folder contains the code for the "Raman Quantification Pipeline" built to create PLS and Convolutional Neuron Network model to predict the concentration of target of interest in the raw raman spectra. Included in the Pipeline, a Genetic Algorithm was built in order to optimize the preprocessing methods and parametrs used. The goal of the algorithm is to reduce the variance between the replications of the sample and increase the variance between the samples

The data can be access using the following files :
* get_spectra_data.py : Contains all the functions used to load the data from the folder (raw raman data) 


The repository contains the Genetic algorithm optimization Main files:

* ga_preprocess_optimization.py : Contains the function related to determine the best preprocessing methods and parameters to preprocess the data.

* ga_baseline_parameter_optimization.py : Contains the functions of the "mini-ga" (mini Genetic ALgorithm) algorithm used to find the best suited parameters of the baseline corrections methods for the experiments-files

* preprocess_fitness.py ; Contains the functions used to evaluate the "fitness"  (ratio variance between sample/variance intra-replicantso of a chromosome or a raman spectra

Preprocessing methods files :

* preprocess_cropping_methods.py : functions used to perform the cropping on the data.
* preprocess_smoothing_methods.py : functions used to perform the smoothing on the data
* preprocess_baseline_methods.py : functions used to perform the baseline correction on the data 
* preprocess_normalization_methods.py : functions used to perform the normalization on the data
* preprocess_Default_parameter.py : Contains the default parameters for each methods of preprocessings and the parameter's value range


In order to train the gentic algorithm and the fitenss functions 3 examples files were created :

* example_evaluation_ratio.py : Allow to preprocess the data by selecting specific preprocessing methods and then evaluated the fitness (ratio variance intersample / variance between replicants) 

* example_mainga_preprocess_opt.py : Allow to optimized the preprocessing by getting the best suited methods and their parameters using the main genetic algorithm from a dataset to select the best ratio variance intersample / variance between replicants. 

* example_miniga_baseline_param.py : Allow to optimized the parameters of each baseline corrections methods from a dataset using a the "mini-ga"

Finally regarding the creation of the models to predict the concentration of the target of the interest in the samples using the raman spectra 3 notebook were added.

* The "GlucoseAcetateWater.ipynb" is the notebook used for the "Glucose, Acetate in Water experiment" to vizualize the dataset and to build the PLS and CNN model used to predict the target of interest's concentration in the Design of experiment on from the bioprocess sample.

* The "GlucoseCitricAcidAmmoniumChlorideYAL_models.ipynb" is the notebook used for the "Glucose, Citric Acid and Ammonium Chloride in YAL Media experiment" to vizualize the dataset and to build the PLS and CNN model used to predict the target of interest's concentration in the Design of experiment on from the bioprocess sample.

* The "CDWFatYAL.ipynb" is the notebook used for the "Cell dry weight and fat in YAL Media experiment" to vizualize the dataset and to build the PLS and CNN model used to predict the target of interest's concentration in the Design of experiment on from the bioprocess sample.


To used the functions and notebook the following files will hep to test the algorithms :

* Water_Ace_Glu.xlsx is the measurement files with the real cocnentration of the target components samples and the measurements_names for the "Glucose and Acetate in Water Experiment".

* Medium_Glu_Cit_AC_doe3.xlsx is the measurement files with the real cocnentration of the target components samples and the measurements_names for the "Glucose, Citric Acid and Ammonium Chloride in YAL Experiment".

* Medium_Cdw_fat.xlsx is the measurement files with the real cocnentration of the target components samples and the measurements_names for the "CDW and Fat in YAL Experiment".


* Real_Measure.xlsx is the measurement files with the bioprocess real concentration of the target components samples and the measurements_names

* Real_Measure_fat.xlsx is the measurement files with the bioprocess real concentration of the fat component and the measurements_names


The Following folder contained the raw raman spectra used for those previous notebooks

* Export_2D_GluAce is the folder that contains all the replicates in 2D of the raman spectra data related to the "Glucose and Acetate in Water experiment".

* Export_2D_GluCitAmo is the folder that contains all the replicates in 2D of the raman spectra data related to the "Glucose, Citric and Ammmonium chloride in YAL experiment".

* Export_2D_CDWFat is the folder that contains all the replicates in 2D of the raman spectra data related to the "CDW and Fat in YAL experiment".

* Bioprocess_2D is the folder that contains all the replicates in 2D of the raman spectra data related to the Bioprocess samples.

* Oil_Measurment_Bioprocess_2D is the folder that contains all the replicates in 2D of the raman spectra data for the bioprocess sample with a known Fat concentration



