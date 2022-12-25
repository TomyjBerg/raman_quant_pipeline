import random
import copy
import ga_preprocess_optimization as ga
import preprocess_baseline_methods as basecorrecter
from preprocess_fitness import evaluate, calc_best_fitness
import pandas as pd


def mini_ga_param_loop(pop_size,mut_prob,patience_cond,smoothed_data,file_names,baseline_parameters,same_files,replicants,files_text_path,verbose=False,):
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
                print(f'Baseline : {baseline_idx}, Best fitness : {fit}, New_parameters : {best_base_param}')
                print(best_base_param)
                with open(files_text_path+'_'+str(baseline_idx)+'.txt','w') as f:
                    f.write(str(fit))
                    f.write('\n')
                    f.write('\n')
                    for list in best_base_param[:-1]:
                        f.write(str(list))
                        f.write('\n')

                        
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
        fit_mini_pop = calc_baseline_pop_fit(pop,smoothed_data,file_names,baseline_param_idx,same_sample,replicants,verbose)
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
        new_pop = create_next_baseline_generation(pop,fit_mini_pop,probability_param_mut)
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

def calc_baseline_pop_fit(pop,smoothed_data,file_names,baseline_method_idx,same_sample,replicants,verbose):
    """Calculate all the fitness of the chromosome inside the population.

    Parameters
    ----------

    pop: array like
        List of all the chromosome inside population

    smoothed_data : array like of pandas dataframe
        List of the smoothed spectra of the raman experiement with cropping (pandas dataframe)

    file_names : array of string
        List of all the names corresponding to the raman spectra files 
    
    baseline_method_idx : int
        Index of the selected baseline_method

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
        (ratio of the variances inter/intra sample (b/w) related to the defined preprocessing baseline step
         for each chromosome inside the population).
    """
    pop_fitness = []
    for chrom in pop:
        chrom_fit = calc_baseline_chro_fit(chrom,smoothed_data,file_names,baseline_method_idx,same_sample,replicants,verbose)
        pop_fitness.append(chrom_fit)
    return pop_fitness

def calc_baseline_chro_fit(chrom,smoothed_data,file_names_,baseline_method_idx,same_sample,replicants,verbose):
    """Calculate the fitness (ratio b/w (variance between each sample (Variance inter Sample) over
        the variance within each samples (Variance intra sample))) of a chromosome
        by performing the baseline correction method defined by the index and this chromosome over the smoothed raman sample.

    Parameters
    ----------

    chrom : array like
        List of the new parameter / default range for the selected baseline method

    smoothed_data : array like of pandas dataframe
        List of the spectra of the raman experiement with cropping (pandas dataframe)

    file_names : array of string
        List of all the names corresponding to the raman spectra files 
    
    baseline_method_idx : int
        Index of the selected baseline_method

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
        ratio of the variances inter/intra sample (b/w) related to the defined baseline correction method 
        with the parameters defined by the chromosome.
    """
    
    baseline_corrected_data = basecorrecter.perform_baseline(smoothed_data,baseline_method_idx,chrom,verbose)
    indiv_fitness = evaluate(file_names_, baseline_corrected_data,same_sample,replicants)
    return indiv_fitness


def create_next_baseline_generation(pop,fit_pop,baseline_param_mutation,pop_elite_size=5):
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
    
    baseline_param_mutation: float
        Probability than the parameters for a defined baseline correction method can mutate
    
    baseline_parameters : array like
        list of all the baseline correction methods that are a list of parameters / range of parameter
 
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
    next_gen_noobi = copy.deepcopy(next_gen_noob)
    next_gen_elite_1 = copy.deepcopy(next_gen_elitism)
    next_gen_elite_2 = copy.deepcopy(next_gen_elitism)
    next_gen_chro = peform_directed_reproduction_baseline_param(next_gen_cross,next_gen_noobi)
    next_gen_chro = perform_baseline_param_mutation(next_gen_chro,baseline_param_mutation)
    next_gen_mut1 = perform_baseline_param_mutation(next_gen_elite_1)
    next_gen_mut2 = perform_baseline_param_mutation(next_gen_elite_2)

  
    next_gen = next_gen_elitism + next_gen_chro +next_gen_mut1 + next_gen_mut2
    print(next_gen)
    return next_gen


def perform_elitism_selection(pop,fitness,pop_elite_size):
    """Get the chromosomes with respectively the 1 to pop_elite_size, pop_elite_size to 2*pop_elite_size 
    best fitness and the pop_elite_size worst fitness from the population

    Parameters
    ----------

    population : array like 
        List of all the chromosome inside population

    fitness : array like (list of float)
        List of fitness of each chromosome inside the population
        (ratio of the variances inter/intra sample (b/w) related to the defined  baseline correction method 
        with paramater defined each chromosome inside the population).

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
    noob_pop = [x for _, x in sorted(zip_pop_2, key=lambda pair: pair[0],reverse =False)]  
    elitism_pop = order_pop[:pop_elite_size]
    sec_elitism_pop = order_pop[pop_elite_size:pop_elite_size*2]
    noo_pop = noob_pop[:pop_elite_size]
    return elitism_pop,sec_elitism_pop,noo_pop

def peform_directed_reproduction_baseline_param(elitism_pop,gen_noob):
    """ Directed reproduction by performing a crossover reproduction with a non_randomly selection of 
        parents. The chromosomes in the sec_elitsism_pop list reproduce with the chromosome in the 
        gen_noob list element_wise

    Parameters
    ----------

    sec_elitism_pop : array like
        List of the chromosomes with the best fitness after the elitism selection value inside population
        A chromosome is a list of elements :
            List of tested parameters / default range for the defined baseline correction method

    noob_pop : array like
        List of the chromosomes with the worst fitness value inside population
    
    Returns
    -------
    dir_children : array like
        List of a new chromosomes (daughters and sons) from the selected parents chromosome 
        (by a crossover reproduction)
    
    """
    dir_children = []
    for i in range(int(len(elitism_pop))):
        father = elitism_pop[i]
        mother = gen_noob[i]
        childs = perform_crossover_baseline_param_reproduction(father,mother)
        children = copy.deepcopy(childs)
        dir_children = dir_children + children
    return dir_children

def perform_crossover_baseline_param_reproduction(father,mother):
    """ Perform crossover reproduction over 2 chromosomes (father/mother) by selecting randomly
         an index point and create one new chromosome with the element in the father list before
         the index point and the element in the mother list after the index point.
         Then create another new chromosome with the element in the mother list before the the 
         index point and the element in the father list after the index point.

    Parameters
    ----------

    father : array like 
        Father-Chromosome for crossover reproduction
        Element in the list "father" :
            List of selected parameters / default range for the baseline correction method

    mother : array like
        Mother-Chromosome for crossover reproduction
        same element than in the "father" list
    
    Returns
    -------
    children : array like
        List of 2 chromosomes (daughter and son) from the parents chromosome (crossover reproduction)
    
    """
    param_range = father[-1]
    son = []
    daughter = []
    choice = random.randrange(0, len(param_range))
    for allele in range(len(param_range)):
        if allele < choice :
            son.append(mother[allele])
            daughter.append(father[allele])
        else:
            son.append(father[allele])
            daughter.append(mother[allele])
    son.append(param_range)
    daughter.append(param_range)
    children = [son,daughter]
    return children


def perform_baseline_param_mutation(children,mutation_param_probability=0.8):
    """ Define new parameter or not (chosen by probability) for the selected baseline correction method
        acording to the mutation_param_probability by selected randomly a parameter inside the range previously defined with the value that could be taken for 
        each parameter

    Parameters
    ----------

    children: array like
        List of chromosome 
            List of selected parameters / default range for the baseline correction method

    mutation_param_probability : float
        Probability than a parameters forthe defined baseline correction method can mutate
    
    Returns
    -------
    children: array like
        List of chromosome 
            List of selected parameters (mutated or not) / default range for the baseline correction method 
    """
    param = children[0][-1]
    for chromosome in children:
        for i in range(len(param)):
            rand = random.random()
            if rand < mutation_param_probability:
                new_param = random.choice(param[i])
                param_copy = copy.deepcopy(new_param)
                chromosome[i] = param_copy
    return children