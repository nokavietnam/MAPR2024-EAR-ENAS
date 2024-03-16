import pickle as p
import json as j

pgd_data = p.load(open('/Users/duongdiep/Master/github/benmark-test/result/data_pgd_acc.p', 'rb'))

# for arch in pgd_data:
#     print(pgd_data[arch])
#     break

meta_data = j.load(open('/Users/duongdiep/Master/github/benmark-test/data/robustness-data/meta.json', 'r'))

for item in meta_data['epsilons']:
    print(meta_data['epsilons'][item])