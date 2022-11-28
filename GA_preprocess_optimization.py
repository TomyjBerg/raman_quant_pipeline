import get_Spectra_data as get_data
import preprocess_method as preprocess
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import preprocess_method as preprocess
from sklearn.decomposition import PCA
import copy
import random

def get_w(files_names,files):
    """Calculate the variance within each replication (Variance intra Sample).

    Parameters
    ----------

    file_names : array of string
        List of all the names corresponding to the raman spectra files 
 
    files : array like of pandas dataframe
        List of the spectra of the raman experiement (pandas dataframe)
    
    Returns
    -------
    w : float
        Variance Intra Sample
        i.e. Square of the (Sum of standard deviation of the 3 Principal Components over the replicants of each sample / 
        number of sample in total)
    """

    name_trip = []
    for f in files_names: 
        sh = f.find('_')
        val = f[0:sh]
        name_trip.append(val)
    name_trip = list(set(name_trip))
    std = 0

    for name in name_trip:
        data = []
        for i in range(len(files_names)):
            sh = files_names[i].find('_')
            val = files_names[i][0:sh]
            if val == name:
                data.append(files[i])
        for k in range(len(data)):
            if k==0:
                df = data[k].transpose()
                df.columns = df.iloc[0] 
                df = df[1:]
            else:
                new_df = data[k].transpose()
                new_df.columns = new_df.iloc[0] 
                new_df = new_df[1:]
                df = pd.concat([df, new_df], ignore_index = True)
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(df)
        principalDf = pd.DataFrame(data = principalComponents
                    , columns = ['principal component 1', 'principal component 2','principal component 3'])
        std = std + principalDf.std()['principal component 1'] + principalDf.std()['principal component 2'] + principalDf.std()['principal component 3']

    return (std/len(name_trip))**2
      
def get_b(files_names,files,same_sample,replicants):
    """Calculate the variance within each sample (Variance inter Sample).

    Parameters
    ----------

    file_names : array of string
        List of all the names corresponding to the raman spectra files 
 
    files : array like of pandas dataframe
        List of the spectra of the raman experiement (pandas dataframe)
    
    same_sample : array of string
        List of all the sample number (string) to ignore during the calcualtion
        i.e. Differents sample with the same parameter of the experiment (DOE middle triplicants)

    replicants : int
        Number of replicants for each sample.
    Returns
    -------
    w : float
        Variance Inter Sample
        i.e. Square of the (Sum of standard deviation of the 3 Principal Components over the samples for each replicants/ 
        number of replicants)
    """
    name_trip = []
    for f in files_names: 
        sh = f.find('_')
        val = f[0:sh]
        name_trip.append(val)
    name_trip = list(set(name_trip))
    name_removed = []
    for name in (name_trip):
        if name in same_sample:
            name_removed.append(name)
    for name in name_removed:
        name_trip.remove(name)
    
    data_rep = [[] for l in range(replicants)]
    for name in name_trip:
        num = 0
        for i in range(len(files_names)):
            sh = files_names[i].find('_')
            val = files_names[i][0:sh]
            if val == name:
                data_rep[num].append(files[i])
                num = num + 1

    std = 0
    for data in data_rep:
        for k in range(len(data)):
            if k==0:
                df = data[k].transpose()
                df.columns = df.iloc[0] 
                df = df[1:]
            else:
                new_df = data[k].transpose()
                new_df.columns = new_df.iloc[0] 
                new_df = new_df[1:]
                df = pd.concat([df, new_df], ignore_index = True)
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(df)
        principalDf = pd.DataFrame(data = principalComponents
                    , columns = ['principal component 1', 'principal component 2','principal component 3'])
        std = std + principalDf.std()['principal component 1'] + principalDf.std()['principal component 2'] + principalDf.std()['principal component 3']
  
    return (std/3)**2
      
    

def evaluate(files_names,files,same_sample,replicants):
    """Calculate the ratio b/w (variance between each sample (Variance inter Sample) over
        the variance within each samples (Variance intra sample)).

    Parameters
    ----------

    file_names : array of string
        List of all the names corresponding to the raman spectra files 
 
    files : array like of pandas dataframe
        List of the spectra of the raman experiement (pandas dataframe)
    
    same_sample : array of string
        List of all the sample number (string) to ignore during the calcualtion
        i.e. Differents sample with the same parameter of the experiment (DOE middle triplicants)

    replicants : int
        Number of replicants for each sample.
    Returns
    -------
    ratio : float
        ratio of the variances inter/intra sample (b/w)
    """
    b = get_b(files_names,files,same_sample,replicants)
    print(b)
    w = get_w(files_names,files)
    print(w)
    ratio = b/w
    return ratio

####  PERFORM_PREPROCESS GA TEST #####

def perform_smoothing(cropped_files,index_smoothing_method,alleles_smoothing_parameters):
    """Perform the smoothing of the cropped raman spectra using a specific method and parameters).

    Parameters
    ---------- 
 
    cropped_files : array like of pandas dataframe
        List of the spectra of the raman experiement after cropping (pandas dataframe)
    
    index_smoothing_method : int
        index of the smoothing methods (1rt gene of the chromosome during GA optimization)

    alleles_smoothing_parameters : array like
        list of all the smoothing methods that are a list of parameters / range of parameter

    Returns
    -------
    smoothed_data : array like of pandas dataframe
        List of the spectra of the raman experiement after smoothing (pandas dataframe)
    """
    param = alleles_smoothing_parameters[:-1]
    print(param)
    if index_smoothing_method  == 1:
        smoothed_data =  [preprocess.whittaker_smoother(d, param[0], param[1]) for d in cropped_files]
    elif index_smoothing_method == 2:
        smoothed_data =  [preprocess.sg_filter(d,param[0],param[1]) for d in cropped_files]
    else: 
        smoothed_data =  [d for d in cropped_files]
    return smoothed_data

def perform_baseline(smoothed_data,index_baseline_method,alleles_baseline_parameters):
    """Perform the baseline of the raman spectra using a specific method and parameters).

    Parameters
    ---------- 
 
    smoothed_data : array like of pandas dataframe
        List of the spectra of the raman experiement (pandas dataframe) after cropping 
        and possibly the smoothing 
    
    index_baseline_method : int
        index of the baseline correction methods (2nd gene of the chromosome during GA optimization)

    alleles_baseline_parameters : array like
        list of all the smoothing methods that are a list of parameters / range of parameter

    Returns
    -------
    base_data : array like of pandas dataframe
        List of the spectra of the raman experiement after baseline correction (pandas dataframe)
    """
    param = alleles_baseline_parameters[:-1]
    print(index_baseline_method)
    if index_baseline_method== 0:
        base_data = [d for d in smoothed_data]
    elif index_baseline_method == 1:
        base_data =  [preprocess.perform_baseline_correction(d, param[0], param[1]) for d in smoothed_data]
    elif index_baseline_method == 2:
        base_data =  [preprocess.asls(d, param[0], param[1]) for d in smoothed_data]
    elif index_baseline_method == 3:
        base_data =  [preprocess.airpls(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 4:
        base_data =  [preprocess.arpls(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 5:
        base_data =  [preprocess.aspls(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 6:
        base_data =  [preprocess.drpls(d, param[0], param[1]) for d in smoothed_data]
    elif index_baseline_method == 7:
        base_data =  [preprocess.improve_arpls(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 8:
        base_data =  [preprocess.improve_asls(d,param[0], param[1],param[2]) for d in smoothed_data]
    elif index_baseline_method == 9:
        base_data =  [preprocess.normal_poly(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 10:
        base_data = [preprocess.mod_poly(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 11:
        base_data =  [preprocess.improve_mod_poly(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 12:
        base_data =  [preprocess.penalized_poly(d,param[0]) for d in smoothed_data]
    elif index_baseline_method == 13:
        base_data =  [preprocess.quantile_poly(d,param[0],param[1]) for d in smoothed_data]
    elif index_baseline_method == 14:
        base_data =  [preprocess.quantile_spline(d,param[0],param[1],param[2],param[3]) for d in smoothed_data]
    elif index_baseline_method == 15:
        base_data =  [preprocess.spline_airpls(d,param[0],param[1],param[2]) for d in smoothed_data]
    elif index_baseline_method == 16:
        base_data =  [preprocess.spline_arpls(d,param[0],param[1],param[2]) for d in smoothed_data]
    elif index_baseline_method == 17:
        base_data =  [preprocess.spline_asls(d,param[0],param[1],param[2],param[3]) for d in smoothed_data]
    elif index_baseline_method == 18:
        base_data =  [preprocess.improve_spline_arpls(d,param[0],param[1],param[2]) for d in smoothed_data]
    elif index_baseline_method == 19:
        base_data =  [preprocess.improve_spline_asls(d,param[0],param[1],param[2],param[3],param[4]) for d in smoothed_data]
    elif index_baseline_method == 20:
        base_data =  [preprocess.amormol(d,param[0]) for d in smoothed_data]
    else:
        base_data =  [preprocess.snip(d,param[0]) for d in smoothed_data]

    return base_data

def perform_normalization(base_data,index_normalization_method,alleles_normalization_parameters,replicants,references):
    """Perform the normalization of the raman spectra using a specific method and parameters).

    Parameters
    ---------- 
 
    base_data : array like of pandas dataframe
        List of the spectra of the raman experiement (pandas dataframe) after cropping 
        and possibly the smoothing and baseline correction
    
    index_normalization_method : int
        index of the baseline correction methods (2nd gene of the chromosome during GA optimization)

    alleles_nomalization_parameters : array like
        list of all the normalization methods that are a list of parameters / range of parameter

    replicants : int
        Number of replication of each sample

    references : pandas dataframe
        [0] = raman_shift 
        [1] = intensity cropped the same way than the other raman spectra
        Reference raman spectra (BlanK)


    Returns
    -------
    norm_data : array like of pandas dataframe
        List of the spectra of the raman experiement after normalization (pandas dataframe)
    """
    if index_normalization_method == 0:
        norm_data = [d for d in base_data]
    elif index_normalization_method  == 1:
        norm_data = preprocess.msc(base_data,replicants, reference=references)
    else:
        norm_data = preprocess.snv(base_data)
    return norm_data

### Population ####

def generate_chromosome(alleles_smoothing_parameters,alleles_baseline_parameters,alleles_normalization_parameters):
    """Generate a chromosome with randomly choosen indexes for each preprocessing method
        (Smoothing / Baseline Correction / Normalization) and pass by the parameters for each preprocessing
        methods

    Parameters
    ---------- 

    alleles_smoothing_parameters : array like
        list of all the smoothing methods that are a list of parameters / range of parameter
    
    alleles_baseline_parameters : array like
        list of all the baseline correction methods that are a list of parameters / range of parameter
 
    alleles_normalization_parameters : array like
        list of all the normalization methods that are a list of parameters / range of parameter
    

    Returns
    -------
    chro : array like
        2 Elements in the list "chro" :
            List of randomly selected index of each preprocessing methods (Smoothing/Baseline/Normalization)
            List of default parameters / range for each preprocessing method (Smoothing/Baseline/Normalization)
    """
    a1 = random.randint(0,2)
    smooth_a1 = alleles_smoothing_parameters[a1]
    a1_param = copy.deepcopy(smooth_a1)
    a2 = random.randint(0,21)
    base_a2 = alleles_baseline_parameters[a2]
    a2_param = copy.deepcopy(base_a2)
    a3 = random.randint(0,2)
    norm_a3 = alleles_normalization_parameters[a3]
    a3_param = copy.deepcopy(norm_a3)
    chromosome = [a1,a2,a3]
    chromo_param = [a1_param,a2_param,a3_param]
    chro = [chromosome,chromo_param]
    return chro

def generate_pop_ini_random(pop_nb,al_smoothing_parameters,al_baseline_parameters,al_norm_parameters):
    """ Generate a population with a selected number of randomly generated chromosome

    Parameters
    ---------- 

    pop_nb : int
        Number of chromosomes wanted in the initial population

    al_smoothing_parameters : array like
        list of all the normalization methods that are a list of parameters / range of parameter
    
    al_baseline_parameters : array like
        list of all the normalization methods that are a list of parameters / range of parameter
 
    al_norm_parameters : array like
        list of all the normalization methods that are a list of parameters / range of parameter
    

    Returns
    -------
    population : array like
        List of all the randomly generated chromosome for the initial population
        
    """
    population = []
    for i in range(pop_nb):
        population.append(generate_chromosome(al_smoothing_parameters,al_baseline_parameters,al_norm_parameters))
    return population

### FITNESS ###

def get_chro_fitness(chrom,cropped_data,file_names,references,replicants,same_samp):
    """Calculate the fitness (ratio b/w (variance between each sample (Variance inter Sample) over
        the variance within each samples (Variance intra sample))) of a chromosome
        by performing the preprocessing defined by this chormosome over the cropped raman sample.

    Parameters
    ----------

    chrom : array like
        2 Elements in the list "chrom" :
            List of randomly selected index of each preprocessing methods (Smoothing/Baseline/Normalization)
            List of default parameters / range for each preprocessing method (Smoothing/Baseline/Normalization

    cropped_data : array like of pandas dataframe
        List of the spectra of the raman experiement with cropping (pandas dataframe)

    file_names : array of string
        List of all the names corresponding to the raman spectra files 
 
    
    references : pandas dataframe
        [0] = raman_shift 
        [1] = intensity cropped the same way than the other raman spectra
        Reference raman spectra (BlanK)

    replicants : int
        Number of replicants for each sample.

    same_sample : array of string
        List of all the sample number (string) to ignore during the calcualtion
        i.e. Differents sample with the same parameter of the experiment (DOE middle triplicants)
    
    Returns
    -------
    indiv_fitness : float
        Fitness of the chromosome
        ratio of the variances inter/intra sample (b/w) related to the defined preprocessinbg method 
        defined by the chromosome..
    """
    smooth_data = perform_smoothing(cropped_data,chrom[0][0],chrom[1][0])
    smooth_ref = perform_smoothing(references,chrom[0][0],chrom[1][0])
    baseline_corrected_data = perform_baseline(smooth_data,chrom[0][1],chrom[1][1])
    baseline_corrected_ref = perform_baseline(smooth_ref,chrom[0][1],chrom[1][1])
    normal_data = perform_normalization(baseline_corrected_data,chrom[0][2],chrom[1][2],replicants,baseline_corrected_ref)
    indiv_fitness = evaluate(file_names,normal_data,same_samp,replicants)
    return indiv_fitness

def get_pop_fitness(pop,cropped_data,file_names,references,replicants,same_samp):
    """Calculate all the fitness of the chromosome inside the population.

    Parameters
    ----------

    population : array like
        List of all the chromosome inside population

    cropped_data : array like of pandas dataframe
        List of the spectra of the raman experiement with cropping (pandas dataframe)

    file_names : array of string
        List of all the names corresponding to the raman spectra files 
    
    references : pandas dataframe
        [0] = raman_shift 
        [1] = intensity cropped the same way than the other raman spectra
        Reference raman spectra (BlanK)

    replicants : int
        Number of replicants for each sample.

    same_sample : array of string
        List of all the sample number (string) to ignore during the calcualtion
        i.e. Differents sample with the same parameter of the experiment (DOE middle triplicants)
    
    Returns
    -------
    pop_fitness : array like (list of float)
        List of fitness of each chromosome inside the population
        (ratio of the variances inter/intra sample (b/w) related to the defined preprocessing method 
        defined by each chromosome inside the population).
    """
    pop_fitness = []
    for chrom in pop:
        chrom_fit = get_chro_fitness(chrom,cropped_data,file_names,references,replicants,same_samp)
        pop_fitness.append(chrom_fit)
    return pop_fitness

def get_best_fitness(fitness):
    """Get the chromosome with the best fitness value and its index inside the population.

    Parameters
    ----------
    fitness : array like (list of float)
        List of fitness of each chromosome inside the population
        (ratio of the variances inter/intra sample (b/w) related to the defined preprocessing method 
        defined by each chromosome inside the population).
    
    Returns
    -------
    best_fit : float
        Best chromosome's fitness value in the population
    
    iter_stock : int
        Index_of the chromosome with the best fitness inside the population 
    """
    best_fit = 0
    iter_stock = 0
    for i in range(len(fitness)):
        if fitness[i] > best_fit:
            iter_stock = i
            best_fit = fitness[i]
    print(best_fit)
    return best_fit,iter_stock


### NEXT GENERATION ###

def elitism_selection(pop,fitness):
    """Get the chromosomes with respectively the 1th to 5th, 6th to 10th best fitness and the 5 worst fitness
    from the population

    Parameters
    ----------

    population : array like 
        List of all the chromosome inside population

    fitness : array like (list of float)
        List of fitness of each chromosome inside the population
        (ratio of the variances inter/intra sample (b/w) related to the defined preprocessing method 
        defined by each chromosome inside the population).
    
    Returns
    -------
    elitism_pop : array like
        List of the 5 chromosomes with the 1th to 5th best fitness value inside population
    
    sec_elitism_pop : array like
        List of the 5 chromosomes with the 6th to 10th best fitness value inside population

    noob_pop : array like
        List of the 5 chromosomes with the worst fitness value inside population
    """
    fit = copy.deepcopy(fitness)
    popu = copy.deepcopy(pop)
    zip_pop = zip(fit,popu)
    zip_pop_2 = copy.deepcopy(zip_pop)
    order_pop = [x for _, x in sorted(zip_pop, key=lambda pair: pair[0],reverse =True)]
    worst_pop = [x for _, x in sorted(zip_pop_2, key=lambda pair: pair[0],reverse =False)]

    elitism_pop = order_pop[:5]
    sec_elitism_pop = order_pop[5:10]
    noob_pop = worst_pop[:5]
    return elitism_pop,sec_elitism_pop,noob_pop

def crossover_param(father,mother):
    """ Perform crossover reproduction over 2 chromosomes (fatehr/mother) by selecting ransdomly
         an index point and create one new chromosome with the element in the father list before
         the index point and the element in the mother list after the index point.
         Then create another new chromosome with the element in the mother list before the the 
         index point and the element in the father list after the index point.

    Parameters
    ----------

    father : array like 
        Father-Chromosome for crossover reproduction
        2 Elements in the list "father" :
            List of randomly selected index of each preprocessing methods (Smoothing/Baseline/Normalization)
            List of default parameters / range for each preprocessing method (Smoothing/Baseline/Normalization

    mother : array like
        Mother-Chromosome for crossover reproduction
        same element than in the "father" list
    
    Returns
    -------
    childs : array like
        List of 2 chromosomes (daughter and son) from the parents chromosome (crossover reproduction)
    
    """
    boy = []
    boy_param = []
    girl = []
    girl_param = []
    choice = random.randrange(0, len(father[0]))
    for allele in range(len(father[0])):
        if allele < choice :
            boy.append(mother[0][allele])
            boy_param.append(mother[1][allele])
            girl.append(father[0][allele])
            girl_param.append(father[1][allele])
        else:
            boy.append(father[0][allele])
            boy_param.append(father[1][allele])
            girl.append(mother[0][allele])
            girl_param.append(mother[1][allele])
    boy_full = [boy,boy_param]
    girl_full = [girl,girl_param]
    childs = [boy_full,girl_full]
    return childs

def directed_reproduction(sec_elitism_pop,gen_noob):
    """ Directed reproduction by performing a crossover reproduction with a non_randomly selection of 
        parents. The chromosomes in the sec_elitsism_pop list reproduce with the chromosome in the 
        gen_noob list element_wise

    Parameters
    ----------

    sec_elitism_pop : array like
        List of the 5 chromosomes with the 6th to 10th best fitness value inside population
        A chromosome is a list of 2 elements :
            List of randomly selected index of each preprocessing methods (Smoothing/Baseline/Normalization)
            List of default parameters / range for each preprocessing method (Smoothing/Baseline/Normalization)


    noob_pop : array like
        List of the 5 chromosomes with the worst fitness value inside population
    
    Returns
    -------
    dir_childs : array like
        List of a new chromosomes (daughters and sons) from the selected parents chromosome 
        (by a crossover reproduction)
    
    """
    dir_childs = []
    for i in range(int(len(sec_elitism_pop))):
        father = sec_elitism_pop[i]
        mother = gen_noob[i]
        childs = crossover_param(father,mother)
        childs_ = copy.deepcopy(childs)
        dir_childs = dir_childs + childs_
    return dir_childs

def param_mutation(allele_param,proba_allele_mut):
    """ Define new parameter or not (chosen by probability) for a preprocessing method using by selected
        randomly a parameter inside the range previously defined with the value that could be taken for 
        each parameter

    Parameters
    ----------

    allele_param : array like
        List of parameters / range of a preprocessing method (Smoothing/Baseline/Normalization)

    proba_allele_mut : array like
        Probability than the parameters for a defined preprocessing method can mutate
    
    Returns
    -------
    allele_param : array like
        List of parameters / range for the preprocessing (Smoothing/Baseline/Normalization)
        with the new parameters modified or not by the function
    
    """
    param_range = allele_param[-1]
    for i in range(len(param_range)):
        rand = random.random()
        if rand < proba_allele_mut:
            allele_param[i] = random.choice(param_range[i])
    return allele_param

def allele_mutation(chro,proba_allele_mut):
    """ Create new chromosome with different parameter or not (chosen by probability) for each preprocessing method
        by selected randomly a parameter inside the range previously defined with the value that 
        could be taken for each parameter

    Parameters
    ----------

    chro : array like
        2 Elements in the list "chro" :
            List of index of each preprocessing methods (Smoothing/Baseline/Normalization)
            List of parameters / range for each preprocessing method (Smoothing/Baseline/Normalization)

    proba_allele_mut : array like
        Probability than the parameters for a defined preprocessing method can mutate
    
    Returns
    -------
    new_chro : array like
        2 Elements in the list "new_chro" :
            List of index of each preprocessing methods (Smoothing/Baseline/Normalization)
            List of parameters (modified or not)/ range for each preprocessing method (Smoothing/Baseline/Normalization)
    
    """
    gene = chro[0]
    allele = chro[1]
    new_allele = []
    for i in range(len(allele)):
        param = param_mutation(allele[i],proba_allele_mut)
        new_allele.append(param)
    new_chro = [gene,new_allele]
    return new_chro

def affine(elitism_pop,proba_mut_allele):
    """ Create a 3 times a list of new chromosomes with different parameter or not (chosen by probability) for each 
        preprocessing method for the 5 chromosomes with the best fitness inside the current population
        and then combine the list into one list

    Parameters
    ----------

    elitism_pop : array like
        List of the 5 chromosomes with the 1th to 5th best fitness value inside population

    proba_mut_allele : array like
        Probability than the parameters for a defined preprocessing method can mutate
    
    Returns
    -------
    pop_elitism_param_mutation : array like
        List of the 15 new chromosomes (same preprocessing method than the input but probably with different
        parameters)
    """
    pop_elitism_param_mutation = []
    for i in range(len(elitism_pop)):
        for j in range(3):
            elite_chro = copy.deepcopy(elitism_pop[i])
            new_chrom = allele_mutation(elite_chro,proba_mut_allele)
            pop_elitism_param_mutation.append(new_chrom)
    return pop_elitism_param_mutation

def mutation(childs,proba_gen_mut,proba_allele_mut,alleles_smoothing_parameters,alleles_baseline_parameters,
    alleles_norm_parameters):
    """ Probably modify the preprocessing methods (smoothing methods / baseline methods / normalization method)
        and probably select the default parameters or randomly selected new onnes for those methods

    Parameters
    ----------

    childs  : array like 
        new population of chromosomes that have to be mutated

    proba_gen_mut : float
        Probability that one or more than one preprocessing method of the chromosome will be change
    
    proba_allele_mut : float
        Probability that one or more than one preprocessing method of the chromosome will be change

    alleles_smoothing_parameters : array like
        list of all the smoothing methods that are a list of parameters / range of parameter
    
    alleles_baseline_parameters : array like
        list of all the baseline correction methods that are a list of parameters / range of parameter
 
    alleles_normalization_parameters : array like
        list of all the normalization methods that are a list of parameters / range of parameter
    
    Returns
    -------
    pop_elitism_param_mutation : array like
        List of the 15 new chromosomes (same preprocessing method than the input but probably with different
        parameters)
    """
    for chromosome in childs:
        for allele in range(len(chromosome[0])):
          if allele == 0:
            rand = random.random()
            if rand < probability:
                a1 = random.randint(0,2)
                chromosome[0][0] = a1
                a1_smooth = copy.deepcopy(alleles_smoothing[a1])
                a1_mute = param_mutation(a1_smooth,allele_proba)
                chromosome[1][0] = a1_mute
          elif allele == 1:
            rand = random.random()
            if rand < probability:
                a2 = random.randint(0,21)
                chromosome[0][1] = a2
                a2_base = copy.deepcopy(alleles_baseline[a2])
                a2_mute = param_mutation(a2_base,allele_proba)
                chromosome[1][1] = a2_mute
          else:
            rand = random.random()
            if rand < probability:
                a3 = random.randint(0,2)
                chromosome[0][2] = a3
                chromosome[1][2] = alleles_norm[a3]
    return childs


def next_generation(pop,fit_pop,mut_proba,all_mut_proba,al_smooth,al_base,al_norm):
    next_gen_elitism,next_gen_sec,next_gen_noob = elitism_selection(pop,fit_pop)
    next_gen_cross = copy.deepcopy(next_gen_sec)
    next_gen_worst = copy.deepcopy(next_gen_noob)
    next_gen_elite = copy.deepcopy(next_gen_elitism)
    next_gen_chro = directed_reproduction(next_gen_cross,next_gen_worst)
    next_gen_chro = mutation(next_gen_chro,mut_proba,all_mut_proba,al_smooth,al_base,al_norm)
    next_gen_affine = affine(next_gen_elite,all_mut_proba)
    next_gen_affine = mutation(next_gen_affine,mut_proba,all_mut_proba,al_smooth,al_base,al_norm)
    next_gen = next_gen_elitism + next_gen_chro + next_gen_affine
    return next_gen

def perform_GA_optimization(pop_size,mut_prob,allele_mut_prob,patience_cond,cropped_data,file_names,
triplicants,references,alleles_smoothing,alleles_baseline,alleles_norm,same_samp):
    pop1 = generate_pop_ini_random(pop_size,alleles_smoothing,alleles_baseline,alleles_norm)
    pop = pop1
    best_fitness = 0
    patience = 0
    iter = 0
    x_ = []
    y_ = []
    best_scenario = pd.DataFrame
    while patience < patience_cond:
        iter = iter + 1
        x_.append(iter)
        fit_pop = get_pop_fitness(pop,cropped_data,file_names,references,triplicants,same_samp)
        best_fit = get_best_fitness(fit_pop)
        y_.append(best_fit[0])
        print(pop[best_fit[1]])
        if best_fit[0] > best_fitness:
            patience = 0
            best_preproc = pop[best_fit[1]]
            best_fitness = best_fit[0]
        else:
            patience = patience + 1
        new_pop = next_generation(pop,fit_pop,mut_prob,allele_mut_prob,alleles_smoothing,alleles_baseline,alleles_norm)
        pop = copy.deepcopy(new_pop)

    return best_fitness,best_preproc

