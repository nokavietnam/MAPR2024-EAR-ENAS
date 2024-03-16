import numpy as np
import pickle as p 
import os
import matplotlib.pyplot as plt

def main():
    error = '3'
    data = p.load(open('data/NASBench201/[CIFAR-10]_data.p', 'rb'))
    pgd_data = p.load(open(f'data/RobustnessBench/data_pgd@Linf-{error}.p', 'rb'))
    pof_pdg = p.load(open(f'data/RobustnessBench/pof_pgd@Linf-acc-{error}_flops.p', 'rb'))
    pof_aa = p.load(open(f'data/RobustnessBench/pof_aa_pgd@Linf-acc-{error}_flops.p', 'rb'))

    PS = []
    F = []
    F_AA = []
    list_arch_pof_aa = list(pof_aa[:, 0])
    for i, arch in enumerate(data['200']):
        F.append([data['200'][arch]['FLOPs'], -data['200'][arch]['test_acc']])
        PS.append(i)
        if arch in list_arch_pof_aa:
            F_AA.append([data['200'][arch]['FLOPs'], -data['200'][arch]['test_acc']])
    F = np.array(F)
    PS = np.array(PS)
    F, idx = np.unique(F, axis=0, return_index=True)
    PS = PS[idx]
    F_AA = np.array(F_AA)
    plt.scatter(F[:, 0], F[:, 1], facecolors='none', edgecolors='tab:blue', s=80)
    plt.scatter(F_AA[:, 0], F_AA[:, 1], c='red', s=40, label='TestAcc-FLOPs')

    plt.legend()
    plt.savefig(f'bench-flop-pdg-{error}.jpg')
    plt.show()


if __name__ == '__main__':
    main()