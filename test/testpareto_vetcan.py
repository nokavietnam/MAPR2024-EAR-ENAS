import matplotlib.pyplot as plt
import pickle as p
import numpy as np
from utils import get_front_0

data = p.load(open('data/NASBench201/[CIFAR-10]_data.p', 'rb'))
pgd_data = p.load(open('data/RobustnessBench/data_pgd@Linf-0.5.p', 'rb'))

PS = []
F = []
for i, arch in enumerate(data['200']):
    F.append([data['200'][arch]['FLOPs'], -data['200'][arch]['test_acc']])
    PS.append(i)
F = np.array(F)
PS = np.array(PS)
F, idx = np.unique(F, axis=0, return_index=True)
PS = PS[idx]
# print(PS)
plt.scatter(F[:, 0], F[:, 1], facecolors='none', edgecolors='tab:blue', s=80)

idx_pof = get_front_0(F)
# print(F)
pof = F[idx_pof]
PS = PS[idx_pof]  
plt.scatter(pof[:, 0], pof[:, 1], c='red', s=60, label='TestAcc-FLOPs')
p.dump(pof, open('/Users/duongdiep/workspace/Research/github/robustness-ev-mo/data/NASBench201/[CIFAR-10]_pareto_front(testing)_1.p', 'wb'))

plt.legend()
plt.savefig('PGD@Acc0.5-FLOPs.jpg')
plt.show()
