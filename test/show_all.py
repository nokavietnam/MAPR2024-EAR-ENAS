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
#print(PS)
# print(idx)
# print(idx_pof)
plt.scatter(pof[:, 0], pof[:, 1], c='red', s=60, label='TestAcc-FLOPs')

pof_pgd05 = p.load(open('data/RobustnessBench/pof_pgd@Linf-acc-0.5_flops.p', 'rb'))
# plt.scatter(pof_pgd05[:, 0], pof_pgd05[:, 1], c='green', s=40, label='PGD@Acc0.5-FLOPs')

common_rows = np.isin(pof_pgd05[:, 0], F[:, 0])
                      
#result = np.column_stack((pof_pgd05[common_rows], F[common_rows, 1]))
#result = np.column_stack((pof_pgd05[common_rows], F[np.isin(F[:, 0], pof_pgd05[:, 0]), 1]))

#print(result)

a = pof_pgd05
b = F
result = []

for a_row in a:
    # Tìm giá trị cần trong b dựa trên giá trị cột đầu tiên của a
    for b_row in b:
        if a_row[0] == b_row[0]:
            result.append([a_row[0], b_row[1]])
            break  # Đã tìm thấy giá trị, không cần tìm tiếp

#print(np.array(result))
result = np.array(result)
print(pof_pgd05)
#plt.scatter(result[:, 0], result[:, 1], c='green', s=40, label='PGD@Acc0.5-FLOPs')

dataNSGA = p.load(open('results/MO-NAS201-1/MO-NAS201-1_NSGA-II_20_False_0_d06_m01_H10_M32_S08/20/#Evals_and_Elitist_Archive_evaluate.p', 'rb'))

dataNSGA = np.array(dataNSGA[1][-1][2])
plt.scatter(dataNSGA[:, 1], dataNSGA[:, 0], c='tab:green', s=40, label='TestAcc-FLOPs-Noise')

plt.legend()
plt.savefig('PGD@Acc0.5-FLOPs.jpg')
plt.show()
