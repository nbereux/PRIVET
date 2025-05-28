import numpy as np

#useful when plotting CDFs
def log_rank_in_cumulative(SIZE):
    p = 1. * np.arange(1, SIZE + 1) / SIZE
    log_p = np.log10(p)
    return log_p

def sorting(dist_NN, idx_NN):
    ind = np.argsort(dist_NN.numpy().reshape(-1,))
    p_NN_dist = dist_NN[ind]
    p_NN_idx = idx_NN[ind]
    return p_NN_dist, p_NN_idx

def get_predictions(pred, groundtruth):
    tp = (pred*groundtruth).sum()
    fn = ((~pred)*groundtruth).sum()
    fp = (pred*(~groundtruth)).sum()
    tn = ((~pred)*(~groundtruth)).sum()

    return tp, fn, fp, tn