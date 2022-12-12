

'''
pop_size = 25
mut_prob = 0.5
patience_cond = 5

print(np.array(smoothed_full_data).shape)
print(len(file_full_names_trip))

fit,find_preprocess = mini_GA_param(pop_size,mut_prob,patience_cond,smoothed_mini_data,file_mini_names_trip,14)
fit,find_preprocess = mini_GA_param(pop_size,mut_prob,patience_cond,smoothed_full_data,file_full_names_trip,14)
print(fit,find_preprocess)
'''