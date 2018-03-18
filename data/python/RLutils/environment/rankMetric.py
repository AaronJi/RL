
import numpy as np

def relevance(label, emphase=True):
    if emphase:
        # this form places stronger emphasis on retrieving relevant documents
        return 2**label - 1
    else:
        return label

# cumulative gain at single position
def CG_singlePos(label):
    return relevance(label)

# discounted cumulative gain at single position
def DCG_singlePos(label, k):
    if k == 0:
        score = relevance(label)/1.0
    else:
        import math
        score = relevance(label)/math.log(k+1, 2)
    return score

# DCG @ k
def DCG(labels, k=None):
    if k is None:
        k = len(labels) - 1

    assert 0 <= k < len(labels)

    dcg_k = 0.0
    for i in range(k):
        dcg_k += DCG_singlePos(labels[i], i)

    return dcg_k

# NDCG @ k
def NDCG(labels, k=None):
    # a ideal sort with labels monotonically decreasing
    labels_sorted = sorted(labels, reverse=True)

    dcg_at_k = DCG(labels, k)
    idcg_at_k = DCG(labels_sorted, k)

    if idcg_at_k == 0:
        idcg_at_k = 1

    return dcg_at_k/idcg_at_k
