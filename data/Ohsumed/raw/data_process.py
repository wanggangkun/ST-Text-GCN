#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import os
from random import shuffle

text = []
labels = []
for i in range(9):
    for _, _, doc in os.walk("training/C0{}".format(i+1)):
        for d in doc:
            with open("training/C0{}/{}".format(i+1, d), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                sen = ""
                for line in lines:
                    sen += line.strip()
                    break
                text.append(sen)
                labels.append(str(i))
for i in range(9, 23):
    for _, _, doc in os.walk("training/C{}".format(i+1)):
        for d in doc:
            with open("training/C{}/{}".format(i+1, d), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                sen = ""
                for line in lines:
                    sen += line.strip()
                    break
                text.append(sen)
                labels.append(str(i))
for i in range(9):
    for _, _, doc in os.walk("test/C0{}".format(i+1)):
        for d in doc:
            with open("test/C0{}/{}".format(i+1, d), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                sen = ""
                for line in lines:
                    sen += line.strip()
                    break
                text.append(sen)
                labels.append(str(i))
for i in range(9, 23):
    for _, _, doc in os.walk("test/C{}".format(i+1)):
        for d in doc:
            with open("test/C{}/{}".format(i+1, d), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                sen = ""
                for line in lines:
                    sen += line.strip()
                    break
                text.append(sen)
                labels.append(str(i))
index = []
for i in range(len(text)):
    index.append(i)
shuffle(index)
train_size = int(len(index) * 0.1)
print(train_size)
with open("../Ohsumed.txt", 'w', encoding='utf-8') as f:
    for i in range(train_size):
        f.writelines(str(i) + "\ttrain\t" + labels[index[i]] + "\n")
    for i in range(train_size, len(index)):
        f.writelines(str(i) + "\ttest\t" + labels[index[i]] + "\n")
with open("../Ohsumed_clean.txt", 'w', encoding='utf-8') as f:
    for i in range(len(index)):
        f.writelines(text[index[i]] + "\n")
