#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from random import shuffle

text = []
labels = []

label2id = {}
with open("classes.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        label2id[lines[i].strip()] = str(i)
with open("rt-polarity.neg", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        text.append(line.strip())
        labels.append(str(0))
with open("rt-polarity.pos", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        text.append(line.strip())
        labels.append(str(1))
index = []
for i in range(len(text)):
    index.append(i)
shuffle(index)
dataset_number = len(index)
train_size = int(dataset_number * 0.67)
print(train_size)
print(dataset_number)
with open("../mr.txt", 'w', encoding='utf-8') as f:
    for i in range(train_size):
        f.writelines(str(i) + "\ttrain\t" + labels[index[i]] + "\n")
    for i in range(train_size, dataset_number):
        f.writelines(str(i) + "\ttest\t" + labels[index[i]] + "\n")
with open("../mr_clean.txt", 'w', encoding='utf-8') as f:
    for i in range(dataset_number):
        f.writelines(text[index[i]] + "\n")
