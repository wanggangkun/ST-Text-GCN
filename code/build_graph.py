#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from random import shuffle
import numpy as np
import scipy.sparse as sp
from math import log
import pickle as pkl
import argparse


def remove_words(dataset, remove_word_freq):
    word_freq = {}
    with open("../data/{}/{}_clean.txt".format(dataset, dataset), 'r', encoding='utf-8') as f1:
        content = f1.readlines()
        for c in content:
            words = c.strip().split(" ")
            for w in words:
                if w in word_freq:
                    word_freq[w] += 1
                else:
                    word_freq[w] = 1
    stop_vocab = []
    for k, v in word_freq.items():
        if v < remove_word_freq:
            stop_vocab.append(k)
    print("remove vocab size")
    print(len(stop_vocab))
    with open("../data/{}/{}_stopwords.txt".format(dataset, dataset), 'w', encoding='utf-8') as f:
        f.writelines("\n".join(stop_vocab))


def get_vocab_conf(dataset, build_time, class_number):
    with open("../data/{}/{}_clean.txt".format(dataset, dataset), 'r', encoding='utf-8') as f1:
        content = f1.readlines()
        with open("../data/{}/{}.txt".format(dataset, dataset), 'r', encoding='utf-8') as f2:
            labels = f2.readlines()
            train_count = 0
            for l in labels:
                if "train" in l:
                    train_count += 1
                else:
                    break
            if build_time == 1:
                vocab_conf_content = content[:train_count]
                vocab_conf_labels = labels[:train_count]
            else:
                vocab_conf_content = content
                vocab_conf_labels = labels[:train_count]
                with open("../data/{}/{}_pred.txt".format(dataset, dataset), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        vocab_conf_labels.append(line)
            vocab_labels = {}
            for i in range(len(vocab_conf_labels)):
                for word in vocab_conf_content[i].strip().split():
                    if word in vocab_labels:
                        vocab_labels[word][int(vocab_conf_labels[i].strip().split("\t")[-1])] += 1
                    else:
                        vocab_labels[word] = [0] * class_number
                        vocab_labels[word][int(vocab_conf_labels[i].strip().split("\t")[-1])] += 1
            vocab_conf = {}
            for k, v in vocab_labels.items():
                max_index = 0
                for i in range(1, len(v)):
                    if v[i] > v[max_index]:
                        max_index = i
                sum_count = 0
                for i in v:
                    sum_count += i
                vocab_conf[k] = v[max_index] / sum_count
            with open("../data/{}/{}_word_conf.txt".format(dataset, dataset), 'w', encoding='utf-8') as f:
                for k, v in vocab_conf.items():
                    f.writelines(k + "\t" + str(v) + "\n")


def get_train_vocab(dataset, class_number, vocab_freq, vocab_confidence):
    with open("../data/{}/{}_clean.txt".format(dataset, dataset), 'r', encoding='utf-8') as f1:
        content = f1.readlines()
        with open("../data/{}/{}.txt".format(dataset, dataset), 'r', encoding='utf-8') as f2:
            labels = f2.readlines()
            train_count = 0
            for l in labels:
                if "train" in l:
                    train_count += 1
                else:
                    break
            if build_time == 1:
                # train_count = int(train_count * 0.9)
                train_vocab_content = content[:train_count]
                train_vocab_labels = labels[:train_count]
            else:
                train_vocab_content = content
                train_vocab_labels = labels[:train_count]
                with open("../data/{}/{}_pred.txt".format(dataset, dataset), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        train_vocab_labels.append(line.strip())
            train_dis = [0] * class_number
            for l in train_vocab_labels:
                train_dis[int(l.strip().split("\t")[-1])] += 1
            for i in range(len(train_dis)):
                train_dis[i] = round(train_dis[i] / len(train_vocab_labels), 3)
            print("Training set data distribution")
            print(train_dis)
            vocab_labels = {}
            for i in range(len(train_vocab_labels)):
                for word in train_vocab_content[i].strip().split():
                    if word in vocab_labels:
                        vocab_labels[word][int(train_vocab_labels[i].strip().split("\t")[-1])] += 1
                    else:
                        vocab_labels[word] = [0] * class_number
                        vocab_labels[word][int(train_vocab_labels[i].strip().split("\t")[-1])] += 1
            factor_i = [x * class_number for x in train_dis]
            train_vocab = {}
            for k, v in vocab_labels.items():
                max_index = 0
                for i in range(1, len(v)):
                    if v[i] > v[max_index]:
                        max_index = i
                flag = 1
                sum_count = 0
                for i in v:
                    sum_count += i
                if v[max_index] / sum_count < vocab_confidence:
                    flag = 0
                if v[max_index] < vocab_freq * (class_number + factor_i[max_index]) / class_number:
                    flag = 0
                if flag == 1:
                    train_vocab[k] = max_index
            print("number of train_vocab")
            print(len(train_vocab))
            shuffle_vocab = []
            for k, v in train_vocab.items():
                shuffle_vocab.append(k)
            shuffle(shuffle_vocab)
            shuffle_train_vocab = {}
            for sv in shuffle_vocab:
                shuffle_train_vocab[sv] = train_vocab[sv]
            with open("../data/{}/{}_vocab_train.txt".format(dataset, dataset), 'w', encoding='utf-8') as fw1:
                with open("../data/{}/{}_vocab_label.txt".format(dataset, dataset), 'w', encoding='utf-8') as fw2:
                    for k, v in shuffle_train_vocab.items():
                        fw1.writelines(k + "\n")
                        fw2.writelines(str(v) + "\n")


def build_doc_word_graph(dataset, class_number, window_size, build_time):
    word_embeddings_dim = 300
    word_vector_map = {}

    doc_name_list = []
    doc_train_list = []
    doc_test_list = []
    with open('../data/{}/{}.txt'.format(dataset, dataset), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            doc_name_list.append(line.strip())
            temp = line.split("\t")
            if temp[1].find('test') != -1:
                doc_test_list.append(line.strip())
            elif temp[1].find('train') != -1:
                doc_train_list.append(line.strip())

    stop_words = set()
    with open("../data/{}/{}_stopwords.txt".format(dataset, dataset), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.add(line.strip())
    doc_content_list = []

    with open('../data/{}/{}_clean.txt'.format(dataset, dataset), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip().split(" ")
            words = " ".join([x for x in words if x not in stop_words])
            doc_content_list.append(words)

    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)

    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    test_size = len(test_ids)

    ids = train_ids + test_ids
    print("dataset numbers")
    print(len(ids))

    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_doc_words_list.append(doc_content_list[int(id)])

    # build vocab
    word_freq = {}
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab_train = []
    with open("../data/{}/{}_vocab_train.txt".format(dataset, dataset), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            vocab_train.append(line.strip())
    vocab_train_size = len(vocab_train)
    vocab = vocab_train
    for w in word_set:
        if w not in vocab:
            vocab.append(w)
    vocab_size = len(vocab)
    with open("../data/{}/{}_vocab.txt".format(dataset, dataset), 'w', encoding='utf-8') as f:
        f.writelines("\n".join(vocab))

    word_doc_list = {}
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    label_list = [str(x) for x in range(class_number)]
    # x: feature vectors of training docs, no initial features
    # select 90% training set
    train_size = len(train_ids)
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size

    with open("../data/{}/ind.{}.train.index".format(dataset, dataset), 'w', encoding='utf-8') as f:
        for i in range(train_size + vocab_train_size):
            f.writelines(str(i) + "\n")

    with open("../data/{}/ind.{}.test.index".format(dataset, dataset), 'w', encoding='utf-8') as f:
        for i in range(train_size + vocab_size, train_size + vocab_size + test_size):
            f.writelines(str(i) + "\n")
    print("build x : train")
    row_x = []
    col_x = []
    data_x = []
    # labeled text
    for i in range(real_train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)
        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            data_x.append(doc_vec[j] / doc_len)
    # train vocab
    for i in range(vocab_train_size):
        vocab_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        word = vocab_train[i]
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            vocab_vec = vocab_vec + np.array(word_vector)
        for j in range(word_embeddings_dim):
            row_x.append(real_train_size + i)
            col_x.append(j)
            data_x.append(vocab_vec[j])

    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
        real_train_size + vocab_train_size, word_embeddings_dim))

    y = []
    # labeled text
    for i in range(real_train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    # train vocab
    vocab_label = []
    with open("../data/{}/{}_vocab_label.txt".format(dataset, dataset), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            vocab_label.append(line.strip())
    for i in range(vocab_train_size):
        label = vocab_label[i]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)
    print("build tx: test")
    row_tx = []
    col_tx = []
    data_tx = []
    for i in range(test_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i + train_size]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_tx.append(i)
            col_tx.append(j)
            data_tx.append(doc_vec[j] / doc_len)
    tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                       shape=(test_size, word_embeddings_dim))

    ty = []
    for i in range(test_size):
        doc_meta = shuffle_doc_name_list[i + train_size]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ty.append(one_hot)
    ty = np.array(ty)

    word_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, word_embeddings_dim))
    for i in range(len(vocab)):
        word = vocab[i]
        if word in word_vector_map:
            vector = word_vector_map[word]
            word_vectors[i] = vector
    print("build allx: labeled text + vocab")
    row_allx = []
    col_allx = []
    data_allx = []
    # train
    for i in range(real_train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)
        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            data_allx.append(doc_vec[j] / doc_len)
    # vocab train
    for i in range(vocab_train_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + real_train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))
    # test
    for i in range(real_train_size, train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + vocab_train_size))
            col_allx.append(j)
            data_allx.append(doc_vec[j] / doc_len)
    # vocab
    for i in range(vocab_train_size, vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))
    row_allx = np.array(row_allx)
    col_allx = np.array(col_allx)
    data_allx = np.array(data_allx)
    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

    ally = []
    for i in range(real_train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ally.append(one_hot)
    for i in range(vocab_train_size):
        label = vocab_label[i]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ally.append(one_hot)
    for i in range(real_train_size, train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ally.append(one_hot)
    for i in range(vocab_train_size, vocab_size):
        one_hot = [0 for l in range(len(label_list))]
        ally.append(one_hot)
    ally = np.array(ally)
    print("train x shape, train y shape, test x shape, test y shape, allx shape, ally shape")
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    print("word Co-occurrence with context windows")
    windows = []
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])
    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    row = []
    col = []
    weight = []
    print("pmi as weights")
    # pmi as weights
    PMI_count = []
    num_window = len(windows)
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        if i < vocab_train_size:
            row.append(real_train_size + i)
        else:
            row.append(train_size + i)
        if j < vocab_train_size:
            col.append(real_train_size + j)
        else:
            col.append(train_size + j)
        weight.append(pmi)
        PMI_count.append(pmi)
    with open("../data/{}/pmi_weight.txt".format(dataset), 'w', encoding='utf-8') as pmif:
        for w in PMI_count:
            pmif.writelines(str(w) + "\n")

    print("doc word frequency")
    # doc word frequency
    doc_word_freq = {}

    for doc_id in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    print("vocab confidence")
    vocab_conf = {}
    with open("../data/{}/{}_word_conf.txt".format(dataset, dataset), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            vocab_conf[line.split("\t")[0]] = float(line.strip().split("\t")[-1])

    fi_count = []
    fiv_count = []
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < real_train_size:
                row.append(i)
            elif i < train_size:
                row.append(i + vocab_train_size)
            else:
                row.append(i + vocab_size)
            if j < vocab_train_size:
                col.append(real_train_size + j)
            else:
                col.append(train_size + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                      word_doc_freq[vocab[j]])
            if build_time != 1 and word in vocab_conf:
                weight.append(freq * idf * vocab_conf[word])
                fiv_count.append(freq * idf * vocab_conf[word])
            else:
                weight.append(freq * idf)
                fiv_count.append(freq * idf)
            doc_word_set.add(word)
            fi_count.append(freq * idf)
    with open("../data/{}/fi_weight.txt".format(dataset), 'w', encoding='utf-8') as fif:
        for w in fi_count:
            fif.writelines(str(w) + "\n")
    with open("../data/{}/fiv_weight.txt".format(dataset), 'w', encoding='utf-8') as fivf:
        for w in fiv_count:
            fivf.writelines(str(w) + "\n")

    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))
    print(adj)
    print("图节点数：" + str(node_size))
    print("边数：" + str(len(weight)))

    print("save data")
    f = open("../data/{}/ind.{}.x".format(dataset, dataset), 'wb')
    pkl.dump(x, f)
    f.close()
    f = open("../data/{}/ind.{}.y".format(dataset, dataset), 'wb')
    pkl.dump(y, f)
    f.close()
    f = open("../data/{}/ind.{}.tx".format(dataset, dataset), 'wb')
    pkl.dump(tx, f)
    f.close()
    f = open("../data/{}/ind.{}.ty".format(dataset, dataset), 'wb')
    pkl.dump(ty, f)
    f.close()
    f = open("../data/{}/ind.{}.allx".format(dataset, dataset), 'wb')
    pkl.dump(allx, f)
    f.close()
    f = open("../data/{}/ind.{}.ally".format(dataset, dataset), 'wb')
    pkl.dump(ally, f)
    f.close()
    f = open("../data/{}/ind.{}.adj".format(dataset, dataset), 'wb')
    pkl.dump(adj, f)
    f.close()


if __name__ == "__main__":
    # parameter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--build_time', type=int, default=2,
                        help='the round of the build')
    parser.add_argument('--dataset', type=str, default="THUCNews",
                        help='AGNews Ohsumed StackOverflow THUCNews Snippets mr')
    parser.add_argument('--vocab_confidence', type=float, default=0.9,
                        help='the confidence of train vocab')
    parser.add_argument('--vocab_freq', type=int, default=3, help='the min number of train vocab frequent')
    parser.add_argument('--remove_word_freq', type=int, default=2,
                        help='min frequent the word removed')
    parser.add_argument('--window_size', type=int, default=10, help='the pmi window size')
    args = parser.parse_args()

    build_time = args.build_time
    vocab_freq = args.vocab_freq
    window_size = args.window_size
    remove_word_freq = args.remove_word_freq
    dataset = args.dataset

    vocab_confidence = args.vocab_confidence
    # class_number = 10
    class_number = 15

    if dataset == 'AGNews':
        class_number = 4
    elif dataset == 'mr':
        class_number = 2
    elif dataset == 'Ohsumed':
        class_number = 23
    elif dataset == 'Snippets':
        class_number = 8
    elif dataset == 'StackOverflow':
        class_number = 20
    elif dataset == 'THUCNews':
        class_number = 10

    print("build_time:", build_time)
    print("vocab_confidence:", vocab_confidence)
    print("dataset:", dataset)
    print("class_number:", class_number)

    print("remove_words")
    remove_words(dataset, remove_word_freq)
    print("get_vocab_conf")
    get_vocab_conf(dataset, build_time, class_number)
    print("get_train_vocab")
    get_train_vocab(dataset, class_number, vocab_freq, vocab_confidence)
    print("build graph")
    build_doc_word_graph(dataset, class_number, window_size, build_time)
