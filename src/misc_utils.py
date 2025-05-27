import numpy as np

#useful when plotting CDFs
def log_rank_in_cumulative(SIZE):
    p = 1. * np.arange(1, SIZE + 1) / SIZE
    log_p = np.log10(p)
    return log_p

#useful when plotting CDFs
def sorting(dist_NN):
    ind = np.argsort(dist_NN[0].numpy().reshape(-1,))
    p_NN_dist = dist_NN[0][ind]
    p_NN_idx = dist_NN[1][ind]
    return p_NN_dist, p_NN_idx