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
    b = get_b(files_names,files,same_sample,replicants)
    print(b)
    w = get_w(files_names,files)
    print(w)
    ratio = b/w
    return ratio

####  PERFORM_PREPROCESS GA TEST #####

def perform_smoothing(cropped_files,a1,allele):
    param = allele[:-1]
    print(param)
    if a1  == 1:
        smoothed_data =  [preprocess.whittaker_smoother(d, param[0], param[1]) for d in cropped_files]
    elif a1 == 2:
        smoothed_data =  [preprocess.sg_filter(d,param[0],param[1]) for d in cropped_files]
    else: 
        smoothed_data =  [d for d in cropped_files]
    return smoothed_data

def perform_baseline(smoothed_data,a2,allele):
    param = allele[:-1]
    print(a2)
    if a2 == 0:
        base_data = [d for d in smoothed_data]
    elif a2 == 1:
        base_data =  [preprocess.perform_baseline_correction(d, param[0], param[1]) for d in smoothed_data]
    elif a2 == 2:
        base_data =  [preprocess.asls(d, param[0], param[1]) for d in smoothed_data]
    elif a2 == 3:
        base_data =  [preprocess.airpls(d,param[0]) for d in smoothed_data]
    elif a2 == 4:
        base_data =  [preprocess.arpls(d,param[0]) for d in smoothed_data]
    elif a2 == 5:
        base_data =  [preprocess.aspls(d,param[0]) for d in smoothed_data]
    elif a2 == 6:
        base_data =  [preprocess.drpls(d, param[0], param[1]) for d in smoothed_data]
    elif a2 == 7:
        base_data =  [preprocess.improve_arpls(d,param[0]) for d in smoothed_data]
    elif a2 == 8:
        base_data =  [preprocess.improve_asls(d,param[0], param[1],param[2]) for d in smoothed_data]
    elif a2 == 9:
        base_data =  [preprocess.normal_poly(d,param[0]) for d in smoothed_data]
    elif a2 == 10:
        base_data = [preprocess.mod_poly(d,param[0]) for d in smoothed_data]
    elif a2 == 11:
        base_data =  [preprocess.improve_mod_poly(d,param[0]) for d in smoothed_data]
    elif a2 == 12:
        base_data =  [preprocess.penalized_poly(d,param[0]) for d in smoothed_data]
    elif a2 == 13:
        base_data =  [preprocess.quantile_poly(d,param[0],param[1]) for d in smoothed_data]
    elif a2 == 14:
        base_data =  [preprocess.quantile_spline(d,param[0],param[1],param[2],param[3]) for d in smoothed_data]
    elif a2 == 15:
        base_data =  [preprocess.spline_airpls(d,param[0],param[1],param[2]) for d in smoothed_data]
    elif a2 == 16:
        base_data =  [preprocess.spline_arpls(d,param[0],param[1],param[2]) for d in smoothed_data]
    elif a2 == 17:
        base_data =  [preprocess.spline_asls(d,param[0],param[1],param[2],param[3]) for d in smoothed_data]
    elif a2 == 18:
        base_data =  [preprocess.improve_spline_arpls(d,param[0],param[1],param[2]) for d in smoothed_data]
    elif a2 == 19:
        base_data =  [preprocess.improve_spline_asls(d,param[0],param[1],param[2],param[3],param[4]) for d in smoothed_data]
    elif a2 == 20:
        base_data =  [preprocess.amormol(d,param[0]) for d in smoothed_data]
    else:
        base_data =  [preprocess.snip(d,param[0]) for d in smoothed_data]

    return base_data

def perform_normalization(base_data,a3,allele,replicants,references):
    if a3 == 0:
        norm_data = [d for d in base_data]
    elif a3 == 1:
        norm_data = preprocess.msc(base_data,replicants, reference=references)
    else:
        norm_data = preprocess.snv(base_data)
    return norm_data

### Population ####

def generate_chromosome(alleles_smoothing,alleles_baseline,alleles_norm):
    a1 = random.randint(0,2)
    smooth_a1 = alleles_smoothing[a1]
    a1_param = copy.deepcopy(smooth_a1)
    a2 = random.randint(0,21)
    base_a2 = alleles_baseline[a2]
    a2_param = copy.deepcopy(base_a2)
    a3 = random.randint(0,2)
    norm_a3 = alleles_norm[a3]
    a3_param = copy.deepcopy(norm_a3)
    chromosome = [a1,a2,a3]
    chromo_param = [a1_param,a2_param,a3_param]
    chro = [chromosome,chromo_param]
    return chro

def generate_pop_ini_random(pop_nb,al_smoothing,al_baseline,al_norm):
    population = []
    for i in range(pop_nb):
        population.append(generate_chromosome(al_smoothing,al_baseline,al_norm))
    return population

### FITNESS ###

def get_chro_fitness(chrom,cropped_data,file_names,references,triplicants,same_samp):
    smooth_data = perform_smoothing(cropped_data,chrom[0][0],chrom[1][0])
    smooth_ref = perform_smoothing(references,chrom[0][0],chrom[1][0])
    baseline_corrected_data = perform_baseline(smooth_data,chrom[0][1],chrom[1][1])
    baseline_corrected_ref = perform_baseline(smooth_ref,chrom[0][1],chrom[1][1])
    normal_data = perform_normalization(baseline_corrected_data,chrom[0][2],chrom[1][2],triplicants,baseline_corrected_ref)
    indiv_fitness = evaluate(file_names,normal_data,same_samp,triplicants)
    return indiv_fitness

def get_pop_fitness(pop,cropped_data,file_names,references,triplicants,same_samp):
    pop_fitness = []
    for chrom in pop:
        chrom_fit = get_chro_fitness(chrom,cropped_data,file_names,references,triplicants,same_samp)
        pop_fitness.append(chrom_fit)
    return pop_fitness

def get_best_fitness(fitness):
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

def directed_reproduction(elitism_pop,gen_noob):
    dir_childs = []
    for i in range(int(len(elitism_pop))):
        father = elitism_pop[i]
        mother = gen_noob[i]
        childs = crossover_param(father,mother)
        childs_ = copy.deepcopy(childs)
        dir_childs = dir_childs + childs_
    return dir_childs

def param_mutation(allele,proba):
    param_range = allele[-1]
    for i in range(len(param_range)):
        rand = random.random()
        if rand < proba:
            allele[i] = random.choice(param_range[i])
    return allele

def allele_mutation(chro,proba):
    gene = chro[0]
    allele = chro[1]
    new_allele = []
    for i in range(len(allele)):
        param = param_mutation(allele[i],proba)
        new_allele.append(param)
    new_chro = [gene,new_allele]
    return new_chro

def affine(elitism_pop,proba):
    pop_elitism_param_mutation = []
    for i in range(len(elitism_pop)):
        if i < 5:
            for j in range(3):
                elite_chro = copy.deepcopy(elitism_pop[i])
                new_chrom = allele_mutation(elite_chro,proba)
                pop_elitism_param_mutation.append(new_chrom)
        else:
            elite_chro = copy.deepcopy(elitism_pop[i])
            new_chrom = allele_mutation(elite_chro,proba)
            pop_elitism_param_mutation.append(new_chrom)
    return pop_elitism_param_mutation

def mutation(childs,probability,allele_proba,alleles_smoothing,alleles_baseline,alleles_norm):
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

