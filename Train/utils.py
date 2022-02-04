import resource
import numpy as np
import scipy.sparse
import sklearn
from sklearn.metrics import f1_score

np.set_printoptions(linewidth=150, edgeitems=5,
                    formatter=dict(float=lambda x: "% 9.3e" % x))


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def muticlass_f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    return micro


def mutilabel_f1(y_true, y_pred):
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0
    return f1_score(y_true, y_pred, average="micro")


def get_max_memory_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
