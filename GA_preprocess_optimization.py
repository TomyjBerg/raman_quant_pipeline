import get_spectra_data as get_data
import preprocess_baseline_methods as basecorrecter
import preprocess_smoothing_methods as smoother
import preprocess_normalization_methods as normalizer
import preprocess_cropping_methods as cropper
import pandas as pd
import numpy as np
from preprocess_fitness import evaluate, calc_best_fitness

import copy
import random


def perform_GA_optimization(pop_size,probability_meth_mut,probability_param_mut,patience_cond,cropped_data,
file_names,replicants,references,alleles_smoothing_parameters,alleles_baseline_parameters,alleles_norm_parameters,
same_samp,verbose=False):
    """ Perform optimization in order to select the best preprocessing methods and parameters by the fitness
        along multiple generation using a genetic algorithm with elitism selection and directed reproduction

    Parameters
    ----------
    
    pop_nb : int
        Number of chromosomes wanted in the initial population
    
    probability_meth_mut : float
        Probability that one or more than one preprocessing method of the chromosome will be change
    
    probability_param_mut : float
        Probability than the parameters for a defined preprocessing method can mutate

    patience_cond : int
        number of iteration wanted before the algorithm stop without any modification of the best fitness

    cropped_data : array like of pandas dataframe
        List of the spectra of the raman experiement with cropping (pandas dataframe)

    file_names : array of string
        List of all the names corresponding to the raman spectra files 
    
    replicants : int
        Number of replicants for each sample.
 
    references : pandas dataframe
        [0] = raman_shift 
        [1] = intensity cropped the same way than the other raman spectra
        Reference raman spectra (BlanK)

    alleles_smoothing_parameters : array like
        list of all the smoothing methods that are a list of parameters / range of parameter
    
    alleles_baseline_parameters : array like
        list of all the baseline correction methods that are a list of parameters / range of parameter
 
    alleles_normalization_parameters : array like
        list of all the normalization methods that are a list of parameters / range of parameter
    
    same_sample : array of string
        List of all the sample number (string) to ignore during the calcualtion
        i.e. Differents sample with the same parameter of the experiment (DOE middle triplicants)
    
    verbose : Bool
        if True : Print baseline correction methods indexes inside each chromosome and the fitness of the chromosome
    
    Returns
    -------
    best_fitness : float
        Fitness of the best chromosomes / the best preprocessing method and parameter determined 
        by the GA algorithm
        The fitness is ratio of the variances inter/intra sample (b/w) related to the defined preprocessinbg method 
        defined by the chromosome..

    best_preprocessing : array like
        Chromosome selected by the Optimization by the Genetic Algorithm, having the best fitness value
        The chromosome is composed of :
            List of index of the best of each preprocessing methods determined by the GA algorithm
            List of best parameters / range for each best preprocessing methods

    """
    pop1 = generate_pop_ini_random(pop_size,alleles_smoothing_parameters,alleles_baseline_parameters,alleles_norm_parameters)
    pop = pop1
    best_fitness = 0
    patience = 0
    iter = 0
    x_ = []
    y_ = []
    while patience < patience_cond:
        iter = iter + 1
        x_.append(iter)
        fit_pop = get_pop_fitness(pop,cropped_data,file_names,references,replicants,same_samp,verbose)
        best_fit = calc_best_fitness(fit_pop,verbose)
        y_.append(best_fit[0])
        if verbose:
            print(pop[best_fit[1]])
        if best_fit[0] > best_fitness:
            patience = 0
            best_preprocessing = copy.deepcopy(pop[best_fit[1]])
            best_fitness = copy.deepcopy(best_fit[0])
        else:
            patience = patience + 1
        new_pop = create_next_generation(pop,fit_pop,probability_meth_mut,probability_param_mut,alleles_smoothing_parameters,
                                alleles_baseline_parameters,alleles_norm_parameters)
        pop = copy.deepcopy(new_pop)

    return best_fitness,best_preprocessing


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



def get_pop_fitness(pop,cropped_data,file_names,references,replicants,same_samp,verbose):
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

    verbose : Bool
        True : Print the baseline correction method index of the chromosomes
    Returns
    -------
    pop_fitness : array like (list of float)
        List of fitness of each chromosome inside the population
        (ratio of the variances inter/intra sample (b/w) related to the defined preprocessing method 
        defined by each chromosome inside the population).
    """
    pop_fitness = []
    for chrom in pop:
        chrom_fit = get_chro_fitness(chrom,cropped_data,file_names,references,replicants,same_samp,verbose)
        pop_fitness.append(chrom_fit)
    return pop_fitness



def get_chro_fitness(chrom,cropped_data,file_names,references,replicants,same_samp,verbose):
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
    
    verbose : Bool
        True : Print the baseline correction method index of the chromosome
    
    Returns
    -------
    indiv_fitness : float
        Fitness of the chromosome
        ratio of the variances inter/intra sample (b/w) related to the defined preprocessinbg method 
        defined by the chromosome..
    """
    smooth_data = smoother.perform_smoothing(cropped_data,chrom[0][0],chrom[1][0])
    smooth_ref = smoother.perform_smoothing(references,chrom[0][0],chrom[1][0])
    baseline_corrected_data = basecorrecter.perform_baseline(smooth_data,chrom[0][1],chrom[1][1],verbose)
    baseline_corrected_ref = basecorrecter.perform_baseline(smooth_ref,chrom[0][1],chrom[1][1],verbose=False)
    normal_data = normalizer.perform_normalization(baseline_corrected_data,chrom[0][2],chrom[1][2],replicants,baseline_corrected_ref)
    indiv_fitness = evaluate(file_names,normal_data,same_samp,replicants)
    return indiv_fitness



####  PERFORM_PREPROCESS GA TEST #####



### Population ####

### FITNESS ###



def create_next_generation(pop,fit_pop,proba_gen_mut,proba_allele_mut,al_smooth_parameters,al_base_parameters,al_norm_parameters,pop_elite_size=5):
    """ Create the new generation using an elistim selection / directed crossover reproduction / 
        perform_mutation of the preprocessing methods / perform_mutation of the parameters of the preprocessing methods

    Parameters
    ----------

    pop : array like 
        List of all the chromosome inside population

    fit_pop : array like (list of float)
        List of fitness of each chromosome inside the population
        (ratio of the variances inter/intra sample (b/w) related to the defined preprocessing method 
        defined by each chromosome inside the population).
    
    proba_gen_mut : float
        Probability that one or more than one preprocessing method of the chromosome will be change
    
    proba_allele_mut : float
        Probability than the parameters for a defined preprocessing method can mutate
   

    al_smoothing_parameters : array like
        list of all the smoothing methods that are a list of parameters / range of parameter
    
    al_baseline_parameters : array like
        list of all the baseline correction methods that are a list of parameters / range of parameter
 
    al_normalization_parameters : array like
        list of all the normalization methods that are a list of parameters / range of parameter
    
     pop_elite_size : int
        number of chromosome wanted in the population of elitism
        by default = 5
    
    Returns
    -------
    next_gen  : array like 
        new population of chromosomes :
            pop_elite_size chromosomes with the best fitness value (elitism selection)
            3 copy of the pop_elite_size  best chromosome's fitness with probabily parameter of preprocessing perform_mutation
            pop_elite_size directed reproduction chromosomes with the best fitness after the elitism_pop selection value 
            inside population with the pop_elite_size worst chromosome's fitness
            (probabily preprocessing method / parameter perform_mutation)
    """
    next_gen_elitism,next_gen_sec,next_gen_noob = perform_elitism_selection(pop,fit_pop,pop_elite_size)
    next_gen_cross = copy.deepcopy(next_gen_sec)
    next_gen_worst = copy.deepcopy(next_gen_noob)
    next_gen_elite = copy.deepcopy(next_gen_elitism)
    next_gen_chro = perform_directed_reproduction(next_gen_cross,next_gen_worst)
    next_gen_chro = perform_mutation(next_gen_chro,proba_gen_mut,proba_allele_mut,al_smooth_parameters,al_base_parameters,al_norm_parameters)
    next_gen_affine = affine(next_gen_elite,proba_gen_mut)
    next_gen_affine = perform_mutation(next_gen_affine,proba_gen_mut,proba_allele_mut,al_smooth_parameters,al_base_parameters,al_norm_parameters)
    next_gen = next_gen_elitism + next_gen_chro + next_gen_affine
    return next_gen




### NEXT GENERATION ###

def perform_elitism_selection(pop,fitness,pop_elite_size):
    """Get the chromosomes with respectively the 1 to pop_elite_size, pop_elite_size to 2*pop_elite_size 
    best fitness and the pop_elite_size worst fitness from the population

    Parameters
    ----------

    population : array like 
        List of all the chromosome inside population

    fitness : array like (list of float)
        List of fitness of each chromosome inside the population
        (ratio of the variances inter/intra sample (b/w) related to the defined preprocessing method 
        defined by each chromosome inside the population).

    pop_elite_size : int
        number of chromosome wanted in the population of elitism
    
    Returns
    -------
    elitism_pop : array like
        List of the pop_elite_size chromosomes with the best fitness value inside population
    
    sec_elitism_pop : array like
        List of the pop_elite_size chromosomes with the best fitness after the elitism_pop selection value 
        inside population 

    noob_pop : array like
        List of the pop_elite_size chromosomes with the worst fitness value inside population
    """
    fit = copy.deepcopy(fitness)
    popu = copy.deepcopy(pop)
    zip_pop = zip(fit,popu)
    zip_pop_2 = copy.deepcopy(zip_pop)
    order_pop = [x for _, x in sorted(zip_pop, key=lambda pair: pair[0],reverse =True)]
    worst_pop = [x for _, x in sorted(zip_pop_2, key=lambda pair: pair[0],reverse =False)]

    elitism_pop = order_pop[:pop_elite_size]
    sec_elitism_pop = order_pop[pop_elite_size:pop_elite_size*2]
    noob_pop = worst_pop[:pop_elite_size]
    return elitism_pop,sec_elitism_pop,noob_pop


def perform_directed_reproduction(sec_elitism_pop,gen_noob):
    """ Directed reproduction by performing a crossover reproduction with a non_randomly selection of 
        parents. The chromosomes in the sec_elitsism_pop list reproduce with the chromosome in the 
        gen_noob list element_wise

    Parameters
    ----------

    sec_elitism_pop : array like
        List of the chromosomes with the best fitness after the elitism selection value inside population
        A chromosome is a list of 2 elements :
            List of randomly selected index of each preprocessing methods (Smoothing/Baseline/Normalization)
            List of default parameters / range for each preprocessing method (Smoothing/Baseline/Normalization)


    noob_pop : array like
        List of the chromosomes with the worst fitness value inside population
    
    Returns
    -------
    dir_children : array like
        List of a new chromosomes (daughters and sons) from the selected parents chromosome 
        (by a crossover reproduction)
    
    """
    dir_children = []
    for i in range(int(len(sec_elitism_pop))):
        father = sec_elitism_pop[i]
        mother = gen_noob[i]
        childs = perform_crossover_param_reproduction(father,mother)
        children = copy.deepcopy(childs)
        dir_children = dir_children + children
    return dir_children


def perform_crossover_param_reproduction(father,mother):
    """ Perform crossover reproduction over 2 chromosomes (father/mother) by selecting ransdomly
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
    children : array like
        List of 2 chromosomes (daughter and son) from the parents chromosome (crossover reproduction)
    
    """
    son = []
    son_param = []
    daughter = []
    daughter_param = []
    choice = random.randrange(0, len(father[0]))
    for allele in range(len(father[0])):
        if allele < choice :
            son.append(mother[0][allele])
            son_param.append(mother[1][allele])
            daughter.append(father[0][allele])
            daughter_param.append(father[1][allele])
        else:
            son.append(father[0][allele])
            son_param.append(father[1][allele])
            daughter.append(mother[0][allele])
            daughter_param.append(mother[1][allele])
    son_full = [son,son_param]
    daughter_full = [daughter,daughter_param]
    childs = [son_full,daughter_full]
    return childs

def perform_mutation(children,proba_gen_mut,proba_allele_mut,alleles_smoothing_parameters,alleles_baseline_parameters,
    alleles_norm_parameters):
    """ Probably modify the preprocessing methods (smoothing methods / baseline methods / normalization method)
        and probably select the default parameters or randomly selected new onnes for those methods

    Parameters
    ----------

    children  : array like 
        new population of chromosomes that have to be mutated

    proba_gen_mut : float
        Probability that one or more than one preprocessing method of the chromosome will be change
    
    proba_allele_mut : float
        Probability than the parameters for a defined preprocessing method can mutate

    alleles_smoothing_parameters : array like
        list of all the smoothing methods that are a list of parameters / range of parameter
    
    alleles_baseline_parameters : array like
        list of all the baseline correction methods that are a list of parameters / range of parameter
 
    alleles_normalization_parameters : array like
        list of all the normalization methods that are a list of parameters / range of parameter
    
    Returns
    -------
    children  : array like 
        new population of chromosomes that have been mutated or not
    """
    for chromosome in children:
        for allele in range(len(chromosome[0])):
          if allele == 0:
            rand = random.random()
            if rand < proba_gen_mut:
                a1 = random.randint(0,2)
                chromosome[0][0] = a1
                a1_smooth = copy.deepcopy(alleles_smoothing_parameters[a1])
                a1_mute = perform_param_mutation(a1_smooth,proba_allele_mut)
                chromosome[1][0] = a1_mute
          elif allele == 1:
            rand = random.random()
            if rand < proba_gen_mut:
                a2 = random.randint(0,21)
                chromosome[0][1] = a2
                a2_base = copy.deepcopy(alleles_baseline_parameters[a2])
                a2_mute = perform_param_mutation(a2_base,proba_allele_mut)
                chromosome[1][1] = a2_mute
          else:
            rand = random.random()
            if rand < proba_gen_mut:
                a3 = random.randint(0,2)
                chromosome[0][2] = a3
                chromosome[1][2] = alleles_norm_parameters[a3]
    return children

def perform_param_mutation(allele_param,proba_allele_mut):
    """ Define new parameter or not (chosen by probability) for a preprocessing method using by selected
        randomly a parameter inside the range previously defined with the value that could be taken for 
        each parameter

    Parameters
    ----------

    allele_param : array like
        List of parameters / range of a preprocessing method (Smoothing/Baseline/Normalization)

    proba_allele_mut : float
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


def affine(elitism_pop,proba_mut_allele):
    """ Create a 3 times a list of new chromosomes with different parameter or not (chosen by probability) for each 
        preprocessing method for the chromosomes with the best fitness inside the current population
        and then combine the list into one list

    Parameters
    ----------

    elitism_pop : array like
        List of the chromosomes comin from the elitism selction

    proba_mut_allele : array like
        Probability than the parameters for a defined preprocessing method can mutate
    
    Returns
    -------
    pop_elitism_perform_param_mutation : array like
        List of the new chromosomes (same preprocessing method than the input but probably with different
        parameters)
    """
    pop_elitism_perform_param_mutation = []
    for i in range(len(elitism_pop)):
        for j in range(3):
            elite_chro = copy.deepcopy(elitism_pop[i])
            new_chrom = perform_allele_mutation(elite_chro,proba_mut_allele)
            pop_elitism_perform_param_mutation.append(new_chrom)
    return pop_elitism_perform_param_mutation


def perform_allele_mutation(chro,proba_allele_mut):
    """ Create new chromosome with different parameter or not (chosen by probability) for each preprocessing method
        by selected randomly a parameter inside the range previously defined with the value that 
        could be taken for each parameter

    Parameters
    ----------

    chro : array like
        2 Elements in the list "chro" :
            List of index of each preprocessing methods (Smoothing/Baseline/Normalization)
            List of parameters / range for each preprocessing method (Smoothing/Baseline/Normalization)

    proba_allele_mut : float
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
        param = perform_param_mutation(allele[i],proba_allele_mut)
        new_allele.append(param)
    new_chro = [gene,new_allele]
    return new_chro



