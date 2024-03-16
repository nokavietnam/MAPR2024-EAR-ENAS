import numpy as np
import pickle as p 
import os
import matplotlib.pyplot as plt

from utils import get_front_0

def main():
    error = '0.1'
    data = p.load(open('data/NASBench201/[CIFAR-10]_data.p', 'rb'))
    pgd_data = p.load(open(f'data/RobustnessBench/data_pgd@Linf-{error}.p', 'rb'))
    pof_pdg = p.load(open(f'data/RobustnessBench/pof_pgd@Linf-acc-{error}_flops.p', 'rb'))
    pof_aa = p.load(open(f'data/RobustnessBench/pof_aa_pgd@Linf-acc-{error}_flops.p', 'rb'))

    F = []
    PS = []
    F_E = []
    F_ARCH = []
    ACC_LIST = []
    list_arch_pof_aa = list(pof_aa[:, 0])
    for i, arch in enumerate(data['200']):
        F.append([data['200'][arch]['FLOPs'], -data['200'][arch]['test_acc']])
        F_E.append([data['200'][arch]['FLOPs'], -pgd_data[arch]])
        ACC_LIST.append([-data['200'][arch]['test_acc'], -pgd_data[arch]])
        PS.append(i)
        F_ARCH.append(arch)
    F_E = np.array(F_E)
    PS = np.array(PS)
    ACC_LIST = np.array(ACC_LIST)
    F_E_test = F_E
    F_E, idx = np.unique(F_E, axis=0, return_index=True)
    PS = PS[idx]
    # F_ARCH = F_ARCH[idx]
    # print(F_ARCH)
    # for id in idx:
    #    print(id)

    idx_pof = get_front_0(F)
    idx_acc = get_front_0(ACC_LIST)
    #pof = F[idx_pof]
    #PS = PS[idx_pof]
    result_test = []
    for id, value in enumerate(idx_pof):
        if value:
            result_test.append(F_E_test[id])
            print(f'{F_E_test[id]}-{F[id]}')
    result_test = np.array(result_test)


    result_test_acc = []
    #result_test_acc_temp = []
    for id, value in enumerate(idx_acc):
        if value:
            result_test_acc.append(F_E_test[id])

    result_test_acc = np.array(result_test_acc)

    # F_AA = np.array(F_AA)
    plt.scatter(F_E[:, 0], F_E[:, 1], facecolors='none', edgecolors='tab:purple', s=80)
    plt.scatter(result_test[:, 0], result_test[:, 1], c='tab:orange', s=80, label='TestAcc-FLOPs-clear')

    F = np.array(F)
    #plt.scatter(F[:, 0], F[:, 1], facecolors='none', edgecolors='tab:green', s=80)
    #plt.scatter(result_test_acc[:, 0], result_test_acc[:, 1], c='yellow', s=20, label='TestAcc-FLOPs')

    # plt.scatter(pof_pdg[:, 0], pof_pdg[:, 1], c='yellow', s=20, label='TestAcc-FLOPs')
    plt.scatter(pof_pdg[:, 0], pof_pdg[:, 1], c='tab:green', s=40, label='TestAcc-FLOPs-Noise')

    

    plt.legend()
    plt.savefig(f'bench-flop-test-pgd-{error}.jpg', dpi=300)
    plt.show()

    # print(pof_pdg)




if __name__ == '__main__':
    main()