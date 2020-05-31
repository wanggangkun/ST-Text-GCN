#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from random import shuffle

text = []
labels = []

label2id = {}
with open("class.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        label2id[lines[i].strip()] = str(i)
with open("SearchSnippets.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        text.append(line.strip())
with open("SearchSnippets_gnd.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        labels.append(str(label2id[line.strip()]))
index = []
for i in range(len(text)):
    index.append(i)
shuffle(index)
dataset_number = len(index)
train_size = int(dataset_number * 0.1)
print(train_size)
print(dataset_number)
with open("../Snippets.txt", 'w', encoding='utf-8') as f:
    for i in range(train_size):
        f.writelines(str(i) + "\ttrain\t" + labels[index[i]] + "\n")
    for i in range(train_size, dataset_number):
        f.writelines(str(i) + "\ttest\t" + labels[index[i]] + "\n")
with open("../Snippets_clean.txt", 'w', encoding='utf-8') as f:
    for i in range(dataset_number):
        f.writelines(text[index[i]] + "\n")
