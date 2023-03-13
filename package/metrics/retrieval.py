"""Defines Retrieval Metrics"""
import numpy as np


def compute_metrics(x, scale=100.):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = np.round(scale * float(np.sum(ind == 0)) / len(ind), 3)
    metrics['R5'] = np.round(scale * float(np.sum(ind < 5)) / len(ind), 3)
    metrics['R10'] = np.round(scale * float(np.sum(ind < 10)) / len(ind), 3)
    metrics['MR'] = np.median(ind) + 1
    return metrics
