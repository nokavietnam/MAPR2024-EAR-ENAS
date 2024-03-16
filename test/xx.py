import pickle as p
from utils import get_front_0
import numpy as np
import matplotlib.pyplot as plt

# pof = p.load(open('/Users/duongdiep/workspace/Research/github/robustness-ev-mo/data/NASBench201/[CIFAR-10]_pareto_front(testing).p', 'rb'))
# print(pof)

# _pof = pof.copy()
# _pof[:, 1] = -(1 - _pof[:, 1])

# p.dump(_pof, open('/Users/duongdiep/workspace/Research/github/robustness-ev-mo/data/NASBench201/[CIFAR-10]_pareto_front(testing)_new.p', 'wb'))


# f_min_max = open('data/NASBench201/[CIFAR-10]_min_max.p', 'rb');
# min_max = p.load(f_min_max)


# # nFLOPs = round((self.data['200'][key]['FLOPs'] - self.min_max['FLOPs']['min']) /
# #              (self.min_max['FLOPs']['max'] - self.min_max['FLOPs']['min']), 6)



# plt.scatter(_pof[:, 0], _pof[:, 1])
# plt.show()

search_res = p.load(open('results/MO-NAS201-1/MO-NAS201-1_NSGA-II_20_False_0_d27_m01_H17_M19_S23/0/#Evals_and_Elitist_Archive_search.p', 'rb'))
final_front = search_res[1][-1]
all_arch = final_front[1]
print(all_arch)

data = p.load(open('data/NASBench201/[CIFAR-10]_data.p', 'rb'))
# pgd_data = p.load(open('/Users/duongdiep/Master/github/benmark-test/result/data_pgd_acc.p', 'rb'))
pgd_data = p.load(open('/Users/duongdiep/Master/github/benmark-test/result/data_aa_apgd-ce_acc.p', 'rb'))
F = []
for arch in all_arch:
    F.append([data['200'][arch]['FLOPs'], -pgd_data[arch][2]])
F = np.unique(F, axis=0)
F = F[get_front_0(F)]

pof = p.load(open('/Users/duongdiep/Master/github/benmark-test/paretofront/pareto-front-0.5.p', 'rb'))
pof[:, 1] = -(1 - pof[:, 1])

plt.scatter(pof[:, 0], pof[:, 1], facecolors='none', edgecolors='b', s=40, label=f'Pareto-optimal Front (noise 0.5)')
plt.scatter(F[:, 0], F[:, 1], c='red', s=15, label=f'Our Front (noise 0.5)')

plt.legend()
plt.title('Noise Epsilon 0.5')

plt.savefig('figs/Pareto-optimal_apgd-ce_0.5.jpg')
plt.show()
