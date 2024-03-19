import matplotlib.pyplot as plt
import pickle as p
import numpy as np
import json

from robustness.robustness_dataset import RobustnessDataset

#from utils import get_front_0

# data = p.load(open('data\\NASBench201\\[CIFAR-10]_data.p', 'rb'))

# if __name__ == "__main__":
#     for i, arch in enumerate(data['200']):
#         print(data['200'][arch])
#         break


dataMeta = json.load(open('../data/robustness-data/meta.json', 'r'))

if __name__ == "__main__":
    # for item in dataMeta['ids']:
    #     print(dataMeta['ids'][item])
    #     break
    data = RobustnessDataset(path="../data/robustness-data")
    data.draw_arch(s='|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|')