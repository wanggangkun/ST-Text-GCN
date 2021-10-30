#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import numpy as np
import heapq
import matplotlib
import matplotlib.pyplot as plt
import gensim
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import seaborn as sns

sns.set(style="darkgrid")
sns.set(font='SimHei')

font = {
    'family': 'SimHei',
    'weight': 'bold',
    'size': 12
}
matplotlib.rc("font", **font)


def analyse_dataset_avg_length(dataset):
    total_length = 0
    count = 0
    with open("../data/{}/{}_clean.txt".format(dataset, dataset), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            total_length += len(line.split(" "))
            count += 1
    print(total_length / count)


def find_keywords(dataset):
    vocab_label = {}
    with open("../data/{}/{}_vocab.txt".format(dataset, dataset), 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()
        with open("../data/{}/{}_all_vocab_labels.txt".format(dataset, dataset), 'r', encoding='utf-8') as f2:
            lines2 = f2.readlines()
            for i in range(len(lines1)):
                vocab_label[lines1[i].strip()] = lines2[i].strip()
    text_label = {}
    with open("../data/{}/{}_clean.txt".format(dataset, dataset), 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()
        with open("../data/{}/{}_all_labels.txt".format(dataset, dataset), 'r', encoding='utf-8') as f2:
            lines2 = f2.readlines()
            for i in range(len(lines1)):
                text_label[lines1[i].strip()] = lines2[i].strip()
    word_conf = {}
    with open("../data/{}/{}_word_conf.txt".format(dataset, dataset), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            word_conf[line.split("\t")[0]] = float(line.split("\t")[-1].strip())
    with open("../data/{}/{}_keywords.txt".format(dataset, dataset), 'w', encoding='utf-8') as f:
        for text in text_label:
            label = text_label[text]
            for word in text.split(" "):
                if word in vocab_label and vocab_label[word] == label and word_conf[word] > 0.9:
                    f.writelines(word + " ")
            f.writelines("\n")


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


def most_similar_sen(dataset, index):
    text = []
    with open("../data/{}/{}_clean.txt".format(dataset, dataset), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            text.append(line.strip())
    embedding = []
    with open("../data/{}/{}_w2v_raw_doc_embedding.txt".format(dataset, dataset), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            embedding.append([float(x) for x in line.strip().split(" ")])
    index_text = text[index]
    index_embedding = embedding[index]
    similar_score = {}
    for i in range(len(text)):
        if i == index:
            continue
        similar_score[i] = cosine_similarity(index_embedding, embedding[i])
    most_similar = heapq.nlargest(3, similar_score.items(), key=lambda x: x[1])
    most_similar_list = []
    for i in most_similar:
        most_similar_list.append(text[i[0]])
    print(index_text)
    print("=====================")
    print("\n".join(most_similar_list))


def plot_round():
    # plt.errorbar(
    #     [1, 2, 3, 4],  # X
    #     [0.4227, 0.4349, 0.4356, 0.4281],  # Y
    #     yerr=[0.0016, 0.0014, 0.0041, 0.0028],  # Y-errors
    #     label="Ohsumed",
    #     fmt="r<--",  # format line like for plot()
    #     linewidth=2  # width of plot line
    # )

    plt.errorbar(
        [1, 2, 3, 4],  # X
        [0.8803 * 100, 0.8873 * 100, 0.8880 * 100, 0.8833 * 100],  # Y
        yerr=[0.0062 * 100, 0.0047 * 100, 0.0050 * 100, 0.0016 * 100],  # Y-errors
        label="AGNews",
        fmt="bo-",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [1, 2, 3, 4],  # X
        [0.7106 * 100, 0.7665 * 100, 0.7692 * 100, 0.7753 * 100],  # Y
        yerr=[0.0035 * 100, 0.0044 * 100, 0.0041 * 100, 0.0018 * 100],  # Y-errors
        label="StackOverflow",
        fmt="g.--",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [1, 2, 3, 4],  # X
        [0.8533 * 100, 0.8615 * 100, 0.8635 * 100, 0.8618 * 100],  # Y
        yerr=[0.0013 * 100, 0.0011 * 100, 0.0030 * 100, 0.0012 * 100],  # Y-errors
        label="THUCNews",
        fmt="yx-.",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [1, 2, 3, 4],  # X
        [0.8857 * 100, 0.8978 * 100, 0.8980 * 100, 0.8979 * 100],  # Y
        yerr=[0.0013 * 100, 0.0042 * 100, 0.0037 * 100, 0.0038 * 100],  # Y-errors
        label="Snippets",
        fmt="c>-.",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [1, 2, 3, 4],  # X
        [0.7094 * 100, 0.7244 * 100, 0.7245 * 100, 0.7239 * 100],  # Y
        yerr=[0.0031 * 100, 0.0035 * 100, 0.0048 * 100, 0.0024 * 100],  # Y-errors
        label="MR",
        fmt="mv:",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.xlabel("训练轮数")
    plt.ylabel("准确率(%)")
    plt.legend()  # Show legend
    plt.show()


def plot_confidence():
    # plt.errorbar(
    #     [0.1, 0.3, 0.5, 0.7, 0.9],  # X
    #     [0.3431, 0.3559, 0.3944, 0.4158, 0.4249],  # Y
    #     yerr=[0.0030, 0.0018, 0.0020, 0.0011, 0.0019],  # Y-errors
    #     label="Ohsumed",
    #     fmt="r<--",  # format line like for plot()
    #     linewidth=2  # width of plot line
    # )

    plt.errorbar(
        [0.1, 0.3, 0.5, 0.7, 0.9],  # X
        [0.8847 * 100, 0.8846 * 100, 0.8854 * 100, 0.8865 * 100, 0.8857 * 100],  # Y
        yerr=[0.0047 * 100, 0.0034 * 100, 0.0024 * 100, 0.0019 * 100, 0.0014 * 100],  # Y-errors
        label="AGNews",
        fmt="bo-",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [0.1, 0.3, 0.5, 0.7, 0.9],  # X
        [0.7467 * 100, 0.7491 * 100, 0.7543 * 100, 0.7643 * 100, 0.7656 * 100],  # Y
        yerr=[0.0033 * 100, 0.0022 * 100, 0.0012 * 100, 0.0039 * 100, 0.0017 * 100],  # Y-errors
        label="StackOverflow",
        fmt="g.--",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [0.1, 0.3, 0.5, 0.7, 0.9],  # X
        [0.8548 * 100, 0.8562 * 100, 0.8576 * 100, 0.8581 * 100, 0.8601 * 100],  # Y
        yerr=[0.0046 * 100, 0.0015 * 100, 0.0039 * 100, 0.0025 * 100, 0.0015 * 100],  # Y-errors
        label="THUCNews",
        fmt="yx-.",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [0.1, 0.3, 0.5, 0.7, 0.9],  # X
        [0.8818 * 100, 0.8834 * 100, 0.8827 * 100, 0.8916 * 100, 0.8946 * 100],  # Y
        yerr=[0.0018 * 100, 0.0018 * 100, 0.0025 * 100, 0.0015 * 100, 0.0021 * 100],  # Y-errors
        label="Snippets",
        fmt="c>-.",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [0.1, 0.3, 0.5, 0.7, 0.9],  # X
        [0.7196 * 100, 0.7195 * 100, 0.7194 * 100, 0.7231 * 100, 0.7160 * 100],  # Y
        yerr=[0.0041 * 100, 0.0013 * 100, 0.0017 * 100, 0.0029 * 100, 0.0012 * 100],  # Y-errors
        label="MR",
        fmt="mv:",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.xlabel("单词置信度")
    plt.ylabel("准确率(%)")
    plt.legend()  # Show legend
    plt.show()


def plot_confidence2():
    # plt.errorbar(
    #     [0.1, 0.3, 0.5, 0.7, 0.9],  # X
    #     [0.3431, 0.3559, 0.3944, 0.4158, 0.4249],  # Y
    #     yerr=[0.0030, 0.0018, 0.0020, 0.0011, 0.0019],  # Y-errors
    #     label="Ohsumed",
    #     fmt="r<--",  # format line like for plot()
    #     linewidth=2  # width of plot line
    # )

    plt.errorbar(
        [0.5, 0.7, 0.9],  # X
        [0.8854 * 100, 0.8865 * 100, 0.8857 * 100],  # Y
        yerr=[0.0024 * 100, 0.0019 * 100, 0.0014 * 100],  # Y-errors
        label="AGNews",
        fmt="bo-",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [0.5, 0.7, 0.9],  # X
        [0.7543 * 100, 0.7643 * 100, 0.7656 * 100],  # Y
        yerr=[0.0012 * 100, 0.0039 * 100, 0.0017 * 100],  # Y-errors
        label="StackOverflow",
        fmt="g.--",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [0.5, 0.7, 0.9],  # X
        [0.8576 * 100, 0.8581 * 100, 0.8601 * 100],  # Y
        yerr=[0.0039 * 100, 0.0025 * 100, 0.0015 * 100],  # Y-errors
        label="THUCNews",
        fmt="yx-.",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [0.5, 0.7, 0.9],  # X
        [0.8827 * 100, 0.8916 * 100, 0.8946 * 100],  # Y
        yerr=[0.0025 * 100, 0.0015 * 100, 0.0021 * 100],  # Y-errors
        label="Snippets",
        fmt="c>-.",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [0.5, 0.7, 0.9],  # X
        [0.7194 * 100, 0.7231 * 100, 0.7160 * 100],  # Y
        yerr=[0.0017 * 100, 0.0029 * 100, 0.0012 * 100],  # Y-errors
        label="MR",
        fmt="mv:",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.xlabel("单词置信度阈值")
    plt.ylabel("准确率(%)")
    plt.legend()  # Show legend
    plt.show()


def plot_vocab_freq():
    # plt.errorbar(
    #     [1, 3, 10, 100, 1000],  # X
    #     [0.4291, 0.4249, 0.4311, 0.4290, 0.4297],  # Y
    #     yerr=[0.0039, 0.0033, 0.0026, 0.0039, 0.0029],  # Y-errors
    #     label="Ohsumed",
    #     fmt="r<--",  # format line like for plot()
    #     linewidth=2  # width of plot line
    # )

    plt.errorbar(
        [1, 3, 10, 100, 1000],  # X
        [0.8823 * 100, 0.8857 * 100, 0.8838 * 100, 0.8817 * 100, 0.8821 * 100],  # Y
        yerr=[0.0030 * 100, 0.0025 * 100, 0.0038 * 100, 0.0038 * 100, 0.0038 * 100],  # Y-errors
        label="AGNews",
        fmt="bo-",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [1, 3, 10, 100, 1000],  # X
        [0.7617 * 100, 0.7665 * 100, 0.7664 * 100, 0.7620 * 100, 0.7614 * 100],  # Y
        yerr=[0.0038 * 100, 0.0034 * 100, 0.0015 * 100, 0.0014 * 100, 0.0041 * 100],  # Y-errors
        label="StackOverflow",
        fmt="g.--",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [1, 3, 10, 100, 1000],  # X
        [0.8586 * 100, 0.8601 * 100, 0.8608 * 100, 0.8607 * 100, 0.8610 * 100],  # Y
        yerr=[0.0012 * 100, 0.0039 * 100, 0.0017 * 100, 0.0037 * 100, 0.0032 * 100],  # Y-errors
        label="THUCNews",
        fmt="yx-.",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [1, 3, 10, 100, 1000],  # X
        [0.8818 * 100, 0.8834 * 100, 0.8827 * 100, 0.8916 * 100, 0.8946 * 100],  # Y
        yerr=[0.0014 * 100, 0.0045 * 100, 0.0046 * 100, 0.0041 * 100, 0.0040 * 100],  # Y-errors
        label="Snippets",
        fmt="c>-.",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [1, 3, 10, 100, 1000],  # X
        [0.7196 * 100, 0.7195 * 100, 0.7194 * 100, 0.7231 * 100, 0.7160 * 100],  # Y
        yerr=[0.0034 * 100, 0.0040 * 100, 0.0032 * 100, 0.0039 * 100, 0.0028 * 100],  # Y-errors
        label="MR",
        fmt="mv:",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.xlabel("词频")
    plt.xscale('log')
    plt.ylabel("准确率(%)")
    plt.legend()  # Show legend
    plt.show()


def plot_proportions():
    plt.errorbar(
        [1, 2, 5, 10, 20],  # X
        [0.1410 * 100, 0.1933 * 100, 0.2657 * 100, 0.3142 * 100, 0.3429 * 100],  # Y
        yerr=[0.0027 * 100, 0.0042 * 100, 0.0044 * 100, 0.0033 * 100, 0.0044 * 100],  # Y-errors
        label="LR+TF-IDF",
        fmt="r<--",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [1, 2, 5, 10, 20],  # X
        [0.1463 * 100, 0.2496 * 100, 0.3262 * 100, 0.3832 * 100, 0.4216 * 100],  # Y
        yerr=[0.0040 * 100, 0.0013 * 100, 0.0019 * 100, 0.0016 * 100, 0.0040 * 100],  # Y-errors
        label="TextCNN",
        fmt="bo-",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [1, 2, 5, 10, 20],  # X
        [0.1304 * 100, 0.1573 * 100, 0.2461 * 100, 0.2997 * 100, 0.3459 * 100],  # Y
        yerr=[0.0021 * 100, 0.0013 * 100, 0.0016 * 100, 0.0020 * 100, 0.0018 * 100],  # Y-errors
        label="FastText",
        fmt="g.--",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [1, 2, 5, 10, 20],  # X
        [0.1387 * 100, 0.1586 * 100, 0.2022 * 100, 0.2471 * 100, 0.3319 * 100],  # Y
        yerr=[0.0044 * 100, 0.0046 * 100, 0.0041 * 100, 0.0035 * 100, 0.0042 * 100],  # Y-errors
        label="TextRNN",
        fmt="yx-.",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [1, 2, 5, 10, 20],  # X
        [0.1904 * 100, 0.2426 * 100, 0.3168 * 100, 0.4015 * 100, 0.4510 * 100],  # Y
        yerr=[0.0043 * 100, 0.0016 * 100, 0.0042 * 100, 0.0039 * 100, 0.0039 * 100],  # Y-errors
        label="Text-GCN",
        fmt="c>-.",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.errorbar(
        [1, 2, 5, 10, 20],  # X
        [0.2531 * 100, 0.2885 * 100, 0.3481 * 100, 0.4242 * 100, 0.4818 * 100],  # Y
        yerr=[0.0045 * 100, 0.0036 * 100, 0.0011 * 100, 0.0024 * 100, 0.0029 * 100],  # Y-errors
        label="ST-Text-GCN",
        fmt="mv:",  # format line like for plot()
        linewidth=2  # width of plot line
    )

    plt.xlabel("比例(%)")
    plt.ylabel("准确率(%)")
    plt.legend()  # Show legend
    plt.show()


def doc2vec(dataset):
    with open("../data/{}/{}_clean.txt".format(dataset, dataset), 'r', encoding='utf-8') as cf:
        docs = cf.readlines()
    TaggededDocument = gensim.models.doc2vec.TaggedDocument
    x_train = []
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
    model_dm = Doc2Vec(x_train, min_count=1, window=10, size=300, sample=1e-3, negative=5, workers=1)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=100)
    with open("../data/{}/{}_doc2vec_doc_embedding.txt.txt".format(dataset, dataset), 'w', encoding='utf-8') as fw:
        with open("../data/{}/{}_clean.txt".format(dataset, dataset), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                text = line.strip().split(" ")
                inferred_vector_dm = model_dm.infer_vector(text)
                fw.writelines(" ".join([str(x) for x in inferred_vector_dm]) + "\n")


def plot_pmi_weight():
    AGNews = []
    MR = []
    Ohsumed = []
    Snippets = []
    StackOverflow = []
    THUCNews = []
    with open("../data/AGNews/pmi_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            AGNews.append(float(line.strip()))
    with open("../data/MR/pmi_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            MR.append(float(line.strip()))
    with open("../data/Ohsumed/pmi_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            Ohsumed.append(float(line.strip()))
    with open("../data/Snippets/pmi_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            Snippets.append(float(line.strip()))
    with open("../data/StackOverflow/pmi_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            StackOverflow.append(float(line.strip()))
    with open("../data/THUCNews/pmi_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            THUCNews.append(float(line.strip()))
    df_AGNews = pd.DataFrame({"AGNews": AGNews})
    df_MR = pd.DataFrame({"MR": MR})
    df_Ohsumed = pd.DataFrame({"Ohsumed": Ohsumed})
    df_Snippets = pd.DataFrame({"Snippets": Snippets})
    df_StackOverflow = pd.DataFrame({"StackOverflow": StackOverflow})
    df_THUCNews = pd.DataFrame({"THUCNews": THUCNews})
    fig, axes = plt.subplots(3, 2)
    sns.kdeplot(df_AGNews['AGNews'], shade=True, ax=axes[0][0])
    sns.kdeplot(df_MR['MR'], shade=True, ax=axes[0][1])
    sns.kdeplot(df_Ohsumed['Ohsumed'], shade=True, ax=axes[1][0])
    sns.kdeplot(df_Snippets['Snippets'], shade=True, ax=axes[1][1])
    sns.kdeplot(df_StackOverflow['StackOverflow'], shade=True, ax=axes[2][0])
    sns.kdeplot(df_THUCNews['THUCNews'], shade=True, ax=axes[2][1])
    plt.tight_layout()
    plt.show()


def plot_fi_weight():
    AGNews = []
    MR = []
    Ohsumed = []
    Snippets = []
    StackOverflow = []
    THUCNews = []
    with open("../data/AGNews/fi_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            AGNews.append(float(line.strip()))
    with open("../data/MR/fi_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            MR.append(float(line.strip()))
    with open("../data/Ohsumed/fi_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            Ohsumed.append(float(line.strip()))
    with open("../data/Snippets/fi_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            Snippets.append(float(line.strip()))
    with open("../data/StackOverflow/fi_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            StackOverflow.append(float(line.strip()))
    with open("../data/THUCNews/fi_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            THUCNews.append(float(line.strip()))
    df_AGNews = pd.DataFrame({"AGNews": AGNews})
    df_MR = pd.DataFrame({"MR": MR})
    df_Ohsumed = pd.DataFrame({"Ohsumed": Ohsumed})
    df_Snippets = pd.DataFrame({"Snippets": Snippets})
    df_StackOverflow = pd.DataFrame({"StackOverflow": StackOverflow})
    df_THUCNews = pd.DataFrame({"THUCNews": THUCNews})
    fig, axes = plt.subplots(3, 2)
    sns.kdeplot(df_AGNews['AGNews'], shade=True, ax=axes[0][0])
    sns.kdeplot(df_MR['MR'], shade=True, ax=axes[0][1])
    sns.kdeplot(df_Ohsumed['Ohsumed'], shade=True, ax=axes[1][0])
    sns.kdeplot(df_Snippets['Snippets'], shade=True, ax=axes[1][1])
    sns.kdeplot(df_StackOverflow['StackOverflow'], shade=True, ax=axes[2][0])
    sns.kdeplot(df_THUCNews['THUCNews'], shade=True, ax=axes[2][1])
    plt.tight_layout()
    plt.show()


def plot_fiv_weight():
    AGNews = []
    MR = []
    Ohsumed = []
    Snippets = []
    StackOverflow = []
    THUCNews = []
    with open("../data/AGNews/fiv_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            AGNews.append(float(line.strip()))
    with open("../data/MR/fiv_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            MR.append(float(line.strip()))
    with open("../data/Ohsumed/fiv_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            Ohsumed.append(float(line.strip()))
    with open("../data/Snippets/fiv_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            Snippets.append(float(line.strip()))
    with open("../data/StackOverflow/fiv_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            StackOverflow.append(float(line.strip()))
    with open("../data/THUCNews/fiv_weight.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            THUCNews.append(float(line.strip()))
    df_AGNews = pd.DataFrame({"AGNews": AGNews})
    df_MR = pd.DataFrame({"MR": MR})
    df_Ohsumed = pd.DataFrame({"Ohsumed": Ohsumed})
    df_Snippets = pd.DataFrame({"Snippets": Snippets})
    df_StackOverflow = pd.DataFrame({"StackOverflow": StackOverflow})
    df_THUCNews = pd.DataFrame({"THUCNews": THUCNews})
    fig, axes = plt.subplots(3, 2)
    sns.kdeplot(df_AGNews['AGNews'], shade=True, ax=axes[0][0])
    sns.kdeplot(df_MR['MR'], shade=True, ax=axes[0][1])
    sns.kdeplot(df_Ohsumed['Ohsumed'], shade=True, ax=axes[1][0])
    sns.kdeplot(df_Snippets['Snippets'], shade=True, ax=axes[1][1])
    sns.kdeplot(df_StackOverflow['StackOverflow'], shade=True, ax=axes[2][0])
    sns.kdeplot(df_THUCNews['THUCNews'], shade=True, ax=axes[2][1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # dataset = "mr"
    # analyse_dataset_avg_length(dataset)
    # plot_round()
    # plot_confidence2()
    # plot_vocab_freq()
    # plot_proportions()
    plot_pmi_weight()
    plot_fi_weight()
    plot_fiv_weight()
