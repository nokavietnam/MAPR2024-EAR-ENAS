import matplotlib.pyplot as plt
import pickle as p
import numpy as np
#from utils import get_front_0

data = p.load(open('data\\NASBench201\\[CIFAR-10]_data.p', 'rb'))

if __name__ == "__main__":
    for i, arch in enumerate(data['200']):
        print(data['200'][arch])
        break