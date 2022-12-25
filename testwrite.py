hey = 'test'
des = 2
list_t = [['hey','hi','j']]

path = 'C:/Users/thoma/Desktop/Master Thesis/Master/'

with open(path + 'ResultParam.txt','w') as f:
    f.write('hey')
    f.write('\n')
    f.write(str(des))
    