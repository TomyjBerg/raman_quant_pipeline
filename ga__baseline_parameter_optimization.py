import random
import copy
import ga_preprocess_optimization as ga
import preprocess_baseline_methods as basecorrecter
from preprocess_fitness import evaluate, calc_best_fitness
import pandas as pd


def mini_ga_param_loop(pop_size,mut_prob,patience_cond,smoothed_data,file_names,baseline_parameters,same_files,replicants,verbose=False):
    """ Perform optimization in order to select the best parameters for each baseline correction methods
        along multiple generation using a genetic algorithm with elitism selection and directed reproduction

    Parameters
    ----------
    
    pop_size : int
        Number of chromosomes wanted in the initial population
    
    probability_param_mut : float
        Probability that one or more than one preprocessing method of the chromosome will be change

    patience_cond : int
        number of iteration wanted before the algorithm stop without any modification of the best fitness

    smoothed_data : array like of pandas dataframe
        List of the smoothed spectra of the raman experiement (pandas dataframe)

    file_names : array of string
        List of all the names corresponding to the raman spectra files 
    
    baseline_parameters : array like
        list of all the baseline correction methods that are a list of parameters / range of parameter
    

    replicants : int
        Number of replicants for each sample.
    
    same_sample : array of string
        List of all the sample number (string) to ignore during the calcualtion
        i.e. Differents sample with the same parameter of the experiment (DOE middle triplicants)
    
    verbose : Bool
        if True : Print baseline correction methods indexes inside each chromosome and the fitness of the chromosome
    
    Returns
    -------

    new_baseline_meth : array like
        List of the best parameters for each baseline correction methods

    """
    new_baseline_meth = []
    for baseline_idx in range(len(baseline_parameters)):
        if baseline_idx == 0 :
            new_baseline_meth.append(baseline_parameters[baseline_idx])
        else:
            fit,best_base_param = mini_GA_param(pop_size,mut_prob,patience_cond,smoothed_data,file_names,baseline_parameters,baseline_idx,same_files,replicants,verbose)
            new_baseline_meth.append(best_base_param)
            if verbose == True:
                print("baseline")
                print(f'Baseline : {baseline_idx:.3f}, Best fitness : {fit:.3f}, New_parameters : {best_base_param:.3f}')
                print(best_base_param)
    return new_baseline_meth


def mini_GA_param(pop_size,probability_param_mut,patience_cond,smoothed_data,file_names,baseline_methods_param,baseline_param_idx,same_sample,replicants,verbose):
    """ Perform optimization in order to select the best parameters for a baseline correction method
        along multiple generation using a genetic algorithm with elitism selection and directed reproduction

    Parameters
    ----------
    
    pop_size : int
        Number of chromosomes wanted in the initial population
    
    probability_param_mut : float
        Probability that one or more than one preprocessing method of the chromosome will be change

    patience_cond : int
        number of iteration wanted before the algorithm stop without any modification of the best fitness

    smoothed_data : array like of pandas dataframe
        List of the smoothed spectra of the raman experiement (pandas dataframe)

    file_names : array of string
        List of all the names corresponding to the raman spectra files 
    
    baseline_methods_param : array like :
        list of parameters / range of parameter of the baseline correction methods
    
    baseline_param_index : array like
        index of the baseline method used

    replicants : int
        Number of replicants for each sample.
    
    same_sample : array of string
        List of all the sample number (string) to ignore during the calcualtion
        i.e. Differents sample with the same parameter of the experiment (DOE middle triplicants)
    
    verbose : Bool
        if True : Print the fitness of the chromosome
    
    Returns
    -------
    best_fitness : float
        Fitness of the best chromosomes / the best preprocessing method and parameter determined 
        by the GA algorithm
        The fitness is ratio of the variances inter/intra sample (b/w) related to the defined baseline method parameters
        defined by the chromosome.

    best_baseline_param : array like
        Chromosome selected by the Optimization by the Genetic Algorithm, having the best fitness value
        The chromosome is composed of :
            List of best parameters / range for the related baseline method
 

    """
    pop1 = generate_baseline_population(pop_size,baseline_methods_param,baseline_param_idx,)
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
        fit_mini_pop = get_baseline_pop_fit(pop,smoothed_data,file_names,baseline_methods_param,baseline_param_idx,same_sample,replicants)
        best_mini_fit = calc_best_fitness(fit_mini_pop,verbose)
        y_.append(best_mini_fit[0])
        if verbose:
            print(pop[best_mini_fit[1]])
        if best_mini_fit[0] > best_fitness:
            patience = 0
            best_baseline_param= pop[best_mini_fit[1]]
            best_fitness = best_mini_fit[0]
        else:
            patience = patience + 1
        new_pop = next_generation_mini(pop,fit_mini_pop,probability_param_mut)
        pop = copy.deepcopy(new_pop)

    return best_fitness,best_baseline_param

#population

def generate_baseline_population(pop_size,baseline_methods_param,baseline_method_idx):
    """Generate a population with a defined number randomly choosen parameter as chromosome for the selected baseline method

    Parameters
    ---------- 
    pop_size: int
        number of wanted chromosomes inside the population
    
    baseline_methods_param : array like
        list of all the baseline correction methods that are a list of parameters / range of parameter

    baseline_method_idx : int
        index of baseline correction methods
    
    Returns
    -------
    chro : array like
        List of default parameters / range for the defined baseline methods
    """
    population = []
    for i in range(pop_size):
        population.append(generate_baseline_chromosome(baseline_methods_param,baseline_method_idx))
    return population

def generate_baseline_chromosome(baseline_methods_param,baseline_method_idx):
    """Generate a chromosome with randomly choosen parameter for the selected baseline method

    Parameters
    ----------
    baseline_methods_param : array like
        list of all the baseline correction methods that are a list of parameters / range of parameter

    baseline_method_idx : int
        index of baseline correction methods
    
    Returns
    -------
    chro : array like
        List of default parameters / range for the defined baseline methods
    """
    base_a2 = baseline_methods_param[baseline_method_idx]
    a2_param = copy.deepcopy(base_a2)
    chro = generate_random_baseline_parameter(a2_param)
    return chro

def generate_random_baseline_parameter(baseline_selected_method_param):
    """Generate a chromosome with randomly choosen parameter for the selected baseline method

    Parameters
    ----------
    baseline_selected_methods_param : array like
        list of old parameters / range of parameter for the selected baseline method
    
    Returns
    -------
    chro : array like
        List of new parameters / range for the defined baseline methods
    """
    param_range = baseline_selected_method_param[-1]
    for i in range(len(param_range)):
        baseline_selected_method_param[i] = random.choice(param_range[i])
    return baseline_selected_method_param

#fitness

def get_baseline_chro_fit(chrom,smooth_data,file_names_,base_met,same_sample,replicants):
    baseline_corrected_data = basecorrecter.perform_baseline(smooth_data,base_met,chrom)
    indiv_fitness = evaluate(file_names_, baseline_corrected_data,same_sample,replicants)
    return indiv_fitness

def get_baseline_pop_fit(pop,smoothed_data,file_names,base_met,same_sample,replicants):
    pop_fitness = []
    for chrom in pop:
        chrom_fit = get_baseline_chro_fit(chrom,smoothed_data,file_names,base_met,same_sample,replicants)
        pop_fitness.append(chrom_fit)
    return pop_fitness


def elitism_mini_selection(pop,fitness):
    fit = copy.deepcopy(fitness)
    popu = copy.deepcopy(pop)
    zip_pop = zip(fit,popu)
    zip_pop_2 = copy.deepcopy(zip_pop)
    order_pop = [x for _, x in sorted(zip_pop, key=lambda pair: pair[0],reverse =True)]
    noob_pop = [x for _, x in sorted(zip_pop_2, key=lambda pair: pair[0],reverse =False)]  
    elitism_pop = order_pop[:5]
    sec_elitism_pop = order_pop[5:10]
    noo_pop = noob_pop[:5]
    return elitism_pop,sec_elitism_pop,noo_pop


def crossover_param_mini(father,mother):
    param_range = father[-1]
    boy = []
    girl = []
    choice = random.randrange(0, len(param_range))
    for allele in range(len(param_range)):
        if allele < choice :
            boy.append(mother[allele])
            girl.append(father[allele])
        else:
            boy.append(father[allele])
            girl.append(mother[allele])
    boy.append(param_range)
    girl.append(param_range)
    childs = [boy,girl]
    return childs

def directed_reproduction_mini(elitism_pop,gen_noob):
    dir_childs = []
    for i in range(int(len(elitism_pop))):
        father = elitism_pop[i]
        mother = gen_noob[i]
        childs = crossover_param_mini(father,mother)
        childs_ = copy.deepcopy(childs)
        dir_childs = dir_childs + childs_
        print(i)
    return dir_childs

def mini_mutation(childs,probability):
    param = childs[0][-1]
    for chromosome in childs:
        for i in range(len(param)):
            rand = random.random()
            if rand < probability:
                new_param = random.choice(param[i])
                param_copy = copy.deepcopy(new_param)
                chromosome[i] = param_copy
    return childs

import copy

def next_generation_mini(pop,fit_pop,mut_proba):
    next_gen_elitism,next_gen_sec,next_gen_noob = elitism_mini_selection(pop,fit_pop)
    next_gen_cross = copy.deepcopy(next_gen_sec)
    next_gen_noobi = copy.deepcopy(next_gen_noob)
    next_gen_elite_1 = copy.deepcopy(next_gen_elitism)
    next_gen_elite_2 = copy.deepcopy(next_gen_elitism)
    next_gen_chro = directed_reproduction_mini(next_gen_cross,next_gen_noobi)
    next_gen_chro = mini_mutation(next_gen_chro,mut_proba)
    next_gen_mut1 = mini_mutation(next_gen_elite_1,0.8)
    next_gen_mut2 = mini_mutation(next_gen_elite_2,0.8)

  
    next_gen = next_gen_elitism + next_gen_chro +next_gen_mut1 + next_gen_mut2
    print(next_gen)
    return next_gen

def mini_GA_param(pop_size,mut_prob,patience_cond,smoothed_data,file_names,num_base_method):
    pop1 = generate_pop_mini(pop_size,num_base_method)
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
        fit_mini_pop = get_mini_fit(pop,smoothed_data,file_names,num_base_method)
        best_mini_fit = calc_best_fitness(fit_mini_pop)
        y_.append(best_mini_fit[0])
        print(pop[best_mini_fit[1]])
        if best_mini_fit[0] > best_fitness:
            patience = 0
            best_preproc = pop[best_mini_fit[1]]
            best_fitness = best_mini_fit[0]
        else:
            patience = patience + 1
        new_pop = next_generation_mini(pop,fit_mini_pop,mut_prob)
        pop = copy.deepcopy(new_pop)

    return best_fitness,best_preproc
  