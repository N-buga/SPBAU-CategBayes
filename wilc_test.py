import os
import json
import numpy as np
from scipy.stats import wilcoxon

dir_ = 'appet'
file_name = os.path.join(dir_, 'boot.txt')
with open(file_name, 'r') as file_to:
    auc = json.load(file_to)

for cl1 in auc.keys():
    for cl2 in auc.keys():
        if cl1 != cl2:
            stat, p_value = wilcoxon(auc[cl1], auc[cl2])
            print('for {} & {}: stat: {}, p_value: {}, mean_diff: {}'.format(cl1, cl2, stat, p_value, (np.array(auc[cl1]) - np.array(auc[cl2])).mean()))
