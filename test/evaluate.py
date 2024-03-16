import pickle as p
import numpy as np
from utils import calculate_IGD_value

result = p.load(open('./results/MO-NAS201-1/MO-NAS201-1_NSGA-II_20_False_0_d06_m01_H14_M44_S42/0/#Evals_and_Elitist_Archive_search.p', 'rb'))
approximation_front = np.array(result[1][-1][-1])
# print(approximation_front)
# print()
approximation_front = np.unique(approximation_front, axis=0)
# print(approximation_front)
# print()
pof = p.load(open('data/RobustnessBench/pof_pgd@Linf-acc-0.5_flops.p', 'rb'))

min_max_flops = p.load(open('./data/NASBench201/[CIFAR-10]_min_max.p', 'rb'))

min_flops, max_flops = min_max_flops['FLOPs']['min'], min_max_flops['FLOPs']['max']
approximation_front[:, 0] = (approximation_front[:, 0] - min_flops) / (max_flops - min_flops) 
pof[:, 0] = (pof[:, 0] - min_flops) / (max_flops - min_flops) 

IGD_value = calculate_IGD_value(pareto_front=pof, non_dominated_front=approximation_front)
print(IGD_value)