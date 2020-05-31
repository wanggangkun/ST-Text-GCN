#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from random import shuffle
import jieba

text = []
labels = []

label2id = {}
with open("class.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        label2id[lines[i].strip()] = str(i)
with open("train.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        text.append(" ".join(jieba.cut(line.strip().split("\t")[0])))
        labels.append(str(label2id[line.strip().split("\t")[-1]]))
with open("dev.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        text.append(" ".join(jieba.cut(line.strip().split("\t")[0])))
        labels.append(str(label2id[line.strip().split("\t")[-1]]))
with open("test.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        text.append(" ".join(jieba.cut(line.strip().split("\t")[0])))
        labels.append(str(label2id[line.strip().split("\t")[-1]]))
index = []
for i in range(len(text)):
    index.append(i)
shuffle(index)
dataset_number = 70000
train_size = int(dataset_number * 0.1)
print(train_size)
print(dataset_number)
with open("../THUCNews.txt", 'w', encoding='utf-8') as f:
    for i in range(train_size):
        f.writelines(str(i) + "\ttrain\t" + labels[index[i]] + "\n")
    for i in range(train_size, dataset_number):
        f.writelines(str(i) + "\ttest\t" + labels[index[i]] + "\n")
with open("../THUCNews_clean.txt", 'w', encoding='utf-8') as f:
    for i in range(dataset_number):
        f.writelines(text[index[i]] + "\n")
