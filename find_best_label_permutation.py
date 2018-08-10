import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.utils import gen_even_slices
from tqdm import tqdm

with open('results/AlexNetSobel_lr0.01_unsup_v2_nc10000_lfc7_rec1_/labels_holder.json', 'r') as f:
    h = json.load(f)

labels = np.asarray(h['labels'])
labels_prev = np.asarray(h['labels_prev_step'])
num_classes = labels.max() + 1

print 'Frac labels unchanged:', (labels == labels_prev).sum() / float(len(labels))
#
a = labels
b = labels_prev

sim = np.zeros((num_classes, num_classes), dtype=np.int)
#
for i in tqdm(xrange(len(a))):
    sim[a[i], b[i]] += 1

new_label = np.argmax(sim, axis=1)

mapping = {i: x for i, x in enumerate(new_label)}

a = np.array([mapping[x] for x in a])

print 'Frac labels unchanged after GREEDY:', (a == b).sum() / float(len(labels))


# for s in tqdm(gen_even_slices(num_classes, 3)):
#     print '\n--'
#     cur_classes = np.arange(num_classes)[s]
#     min_class = cur_classes.min()
#
#     idxs = np.arange(len(labels))[np.isin(labels, cur_classes)]
#     a = labels[idxs]
#     b = labels_prev[idxs]
#
#     cur_classes -= min_class
#     a -= min_class
#     b -= min_class
#
#     print 'Frac labels unchanged:', (a == b).sum() / float(len(a))
#
#     sim = np.zeros((len(cur_classes), len(cur_classes)), dtype=np.int)
#
#     for i in tqdm(xrange(len(a))):
#         if 0 <= b[i] < len(cur_classes):
#             sim[a[i], b[i]] += 1
#
#     print sim.reshape(-1).sum()
#
#     print 'Running Hungarian algo'
#     row_ind, col_ind = linear_sum_assignment(-sim)
#     print 'Frac labels unchanged after matching:', sim[row_ind, col_ind].sum() / float(len(a))




