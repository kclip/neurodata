import bisect
import numpy as np
import torch


def one_hot(alphabet_size, idx):
    assert idx <= alphabet_size
    out = [0]*alphabet_size
    if idx > 0:
        out[idx - 1] = 1
    return out


def expand_targets(targets, T=500, burnin=0):
    y = np.tile(targets.copy(), [T, 1, 1])
    y[:burnin] = 0
    return y

def find_indices_for_labels(hdf5_group, labels):
    res = []
    for label in labels:
        res.append(np.where(hdf5_group.labels[:] == label)[0])
    return np.hstack(res)

def make_outputs_binary(labels, T, classes):
    mapping = {classes[i]: i for i in range(len(classes))}

    if hasattr(labels, '__len__'):
        out = torch.zeros([len(labels), len(classes), T])
        out[[i for i in range(len(labels))], [mapping[lbl] for lbl in labels], :] = 1
    else:
        out = torch.zeros([len(classes), T])
        mapping = {classes[i]: i for i in range(len(classes))}
        out[mapping[labels], :] = 1
    return out


def find_first(a, tgt):
    return bisect.bisect_left(a, tgt)
