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


def make_output_from_labels(labels, T, n_classes, classes, size):
    if len(size) == 2:
        return make_outputs_multivalued(labels, T, classes)
    else:
        return make_outputs_binary(labels, T, n_classes, classes)

def make_outputs_binary(labels, T, n_classes, classes):
    return torch.nn.functional.one_hot(torch.Tensor([labels]).type(torch.long),
                                       num_classes=n_classes).transpose(0, 1).repeat(1, T)

def make_outputs_multivalued(labels, T, classes):
    mapping = {classes[i]: i for i in range(len(classes))}

    if hasattr(labels, 'len'):
        out = torch.zeros([len(labels), len(classes), 2, T])
        out[[i for i in range(len(labels))], [mapping[lbl] for lbl in labels], 0, :] = 1
    else:
        out = torch.zeros([len(classes), 2, T])
        mapping = {classes[i]: i for i in range(len(classes))}
        out[mapping[labels], 0, :] = 1
    return torch.Tensor(out)


def find_first(a, tgt):
    return bisect.bisect_left(a, tgt)
