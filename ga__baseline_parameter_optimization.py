import random
import copy
import ga_preprocess_optimization as ga
import preprocess_baseline_methods as basecorrecter
from preprocess_fitness import evaluate, calc_best_fitness
import pandas as pd


#population

def miniga_param_mutation(allele):
  param_range = allele[-1]
  for i in range(len(param_range)):
    allele[i] = random.choice(param_range[i])
  return allele

def miniga_allele_param_mutation(allele):
  param = miniga_param_mutation(allele)
  return param

def generate_mini_chromosome(b_alelle,alleles_baseline):
    base_a2 = alleles_baseline[b_alelle]
    a2_param = copy.deepcopy(base_a2)
    a2 = miniga_allele_param_mutation(a2_param)
    return a2
  
def generate_pop_mini(size,base_met):
  population = []
  for i in range(size):
      population.append(generate_mini_chromosome(base_met))
  return population

#fitness

def get_chro_mini_fit(chrom,smooth_data,file_names_,base_met,same_sample,replicants):
  baseline_corrected_data = basecorrecter.perform_baseline(smooth_data,base_met,chrom)
  indiv_fitness = evaluate(file_names_, baseline_corrected_data,same_sample,replicants)
  return indiv_fitness

def get_mini_fit(pop,smoothed_data,file_names,base_met,same_sample,replicants):
  pop_fitness = []
  for chrom in pop:
    chrom_fit = get_chro_mini_fit(chrom,smoothed_data,file_names,base_met)
    pop_fitness.append(chrom_fit)
  return pop_fitness

import copy
#5 - 2eme avec les dernier
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
  
def mini_ga_param_loop(pop_size,mut_prob,patience_cond,smoothed_data,file_names,baseline_met):
  new_baseline_meth = []
  for i in range(len(baseline_met)):
    if i in range(0,16):
      new_baseline_meth.append(baseline_met[i])
    else:
      fit,best_base_param = mini_GA_param(pop_size,mut_prob,patience_cond,smoothed_data,file_names,i)
      new_baseline_meth.append(best_base_param)
      print("baseline")
      print(i)
      print(best_base_param)
    
  return new_baseline_meth


