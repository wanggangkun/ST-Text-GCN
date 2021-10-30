from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import accuracy, load_corpus, f1
from models import GCN
from sklearn import metrics

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="THUCNews",
                    help='AGNews Ohsumed StackOverflow THUCNews Snippets mr')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=2333, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=300,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--early_stopping', type=int, default=20,
                    help='Tolerance for early stopping (# of epochs).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = args.dataset
adj, features, labels, idx_train, idx_val, idx_test, train_size, val_size, test_size, node_size = load_corpus(dataset)
print(train_size, val_size, test_size, node_size)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    f1_val = f1(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'f1_val: {:.4f}'.format(f1_val),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.item()


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    f1_test = f1(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          'f1= {:.4f}'.format(f1_test))
    preds = output[idx_test].max(1)[1].type_as(labels)
    with open("../data/{}/{}_pred.txt".format(dataset, dataset), 'w', encoding='utf-8') as f:
        f.writelines("\n".join([str(x.item()) for x in preds]))
    labels_all = labels[idx_test].cpu()
    predict_all = preds.cpu()
    report = metrics.classification_report(labels_all, predict_all, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    print(report)
    print(confusion)


# Train model
t_total = time.time()
cost_val = []
for epoch in range(args.epochs):
    cost = train(epoch)
    cost_val.append(cost)
    if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping + 1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

'''
# word embedding
print("write word embedding")
embedding = model.get_embedding().cpu().detach().numpy()
print(embedding)
print(embedding.shape)
word_embedding = []
with open("../data/{}/{}_vocab_label.txt".format(dataset, dataset), 'r', encoding='utf-8') as f:
    lines = f.readlines()
    vocab_train_size = len(lines)
for i in embedding[train_size-vocab_train_size-val_size: train_size-val_size]:
    word_embedding.append(i)
for i in embedding[train_size: node_size-test_size]:
    word_embedding.append(i)
vocab = []
with open("../data/{}/{}_vocab.txt".format(dataset, dataset), 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        vocab.append(line.strip())
with open("../data/{}/{}_gcn_word_embedding.txt".format(dataset, dataset), 'w', encoding='utf-8') as f:
    for i in range(len(word_embedding)):
        f.writelines(vocab[i] + " " + " ".join([str(x) for x in word_embedding[i]]) + '\n')

# doc embedding
print("write doc embedding")
doc_embedding = []
for i in embedding[: train_size-vocab_train_size-val_size]:
    doc_embedding.append(i)
for i in embedding[train_size-val_size: train_size]:
    doc_embedding.append(i)
for i in embedding[node_size-test_size: node_size]:
    doc_embedding.append(i)
with open("../data/{}/{}_gcn_doc_embedding.txt".format(dataset, dataset), 'w', encoding='utf-8') as f:
    for i in range(len(doc_embedding)):
        f.writelines(" ".join([str(x) for x in doc_embedding[i]]) + '\n')
'''
