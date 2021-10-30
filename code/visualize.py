from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

dataset = 'StackOverflow'

f = open('../data/{}/{}.txt'.format(dataset, dataset), 'r')
lines = f.readlines()
f.close()

f = open('../data/{}/{}_PTE_doc_embedding.txt'.format(dataset, dataset), 'r')
embedding_lines = f.readlines()
f.close()

target_names = set()
labels = []
docs = []
for i in range(len(lines)):
    line = lines[i].strip()
    temp = line.split('\t')
    if temp[1].find('test') != -1:
        labels.append(temp[2])
        emb_str = embedding_lines[i].strip().split()
        values_str_list = emb_str
        values = [float(x) for x in values_str_list]
        docs.append(values)
        target_names.add(temp[2])

target_names = list(target_names)

label = np.array(labels)

fea = TSNE(n_components=2).fit_transform(docs)
cls = np.unique(label)

fea_num = [fea[label == i] for i in cls]
for i, f in enumerate(fea_num):
    if cls[i] in range(10):
        plt.scatter(f[:, 0], f[:, 1], label=cls[i], marker='+')
    else:
        plt.scatter(f[:, 0], f[:, 1], label=cls[i])

plt.tight_layout()
plt.show()
