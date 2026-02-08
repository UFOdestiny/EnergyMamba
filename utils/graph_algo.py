import numpy as np


def compute_gcn_norm(adj):
    A = adj + np.eye(adj.shape[0])
    d = A.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = np.diag(d_inv_sqrt)
    return (D @ A @ D).astype(np.float32)
