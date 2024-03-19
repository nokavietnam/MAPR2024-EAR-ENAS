import pickle as p
import json
import os
import numpy as np


list_ops = np.array(['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3'])

# convert 
# {'nb201-string': '|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|', 'isomorph': '0'} 
# to 
# 421211
def decode_arch(arch):
    return ''.join(
        str(np.nonzero(list_ops == item.split('~')[0])[0][0]) for item in filter(None, arch.split('|')) if item != '+')

def main(noise, dataset):
    data_pdg = json.load(open(f'data/robustness-data/{dataset}/{noise}@Linf_accuracy.json', 'r'))
    data_pdg_acc = data_pdg[f'{dataset}'][f'{noise}@Linf']['accuracy']
    meta_data = json.load(open('data/robustness-data/meta.json', 'r'))

    data_result = dict()
    clean_data = json.load(open(f'data/robustness-data/{dataset}/clean_accuracy.json', 'r'))
    clean_data_acc = clean_data[f'{dataset}']['clean']['accuracy']
    for item in clean_data_acc:
        data_pdg_acc[item].insert(0, clean_data_acc[item])

    for item in meta_data['ids']:
        try:
            data_result.update({decode_arch(meta_data['ids'][item]['nb201-string']) : data_pdg_acc[item]})
        except:
            data_result.update({decode_arch(meta_data['ids'][item]['nb201-string']) : data_pdg_acc[meta_data['ids'][item]['isomorph']]})

    for item in data_result:
        print(f'{item} - {data_result[item]}')
        break

    #p.dump(data_result, open(f'data/robustness/result/data_{noise}_acc.p', 'wb'))


if __name__ == '__main__':
    main(noise='aa_apgd-ce', dataset='cifar10')