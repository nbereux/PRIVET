from typing import Tuple

import numpy as np


# useful when plotting CDFs
def log_rank_in_cumulative(size: int) -> np.ndarray:
    p = 1.0 * np.arange(1, size + 1) / size
    log_p = np.log10(p)
    return log_p


def sorting(dist_NN: np.ndarray, idx_NN: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ind = np.argsort(
        dist_NN.reshape(
            -1,
        )
    )
    p_NN_dist = dist_NN[ind]
    p_NN_idx = idx_NN[ind]
    return p_NN_dist, p_NN_idx


def get_predictions(pred, groundtruth):
    tp = (pred * groundtruth).sum()
    fn = ((~pred) * groundtruth).sum()
    fp = (pred * (~groundtruth)).sum()
    tn = ((~pred) * (~groundtruth)).sum()

    return tp, fn, fp, tn


def roc(groundtruth, score, all_thresholds):
    x_fpr, y_tpr, y_precision, x_recall, threshold_lst = [], [], [], [], []
    tp_lst, fp_lst, tn_lst, fn_lst = [], [], [], []

    eps = 1e-6
    for threshold in all_thresholds:
        pred = score < threshold

        tp, fn, fp, tn = get_predictions(pred, groundtruth)

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = tp / eps

        if tp + fn != 0:
            recall = tp / (tp + fn)
        else:
            recall = tp / eps

        x_fpr.append(fpr)
        y_tpr.append(tpr)

        y_precision.append(precision)
        x_recall.append(recall)

        threshold_lst.append(threshold)
        tp_lst.append(tp)
        fp_lst.append(fp)
        tn_lst.append(tn)
        fn_lst.append(fn)

    auc_roc = np.trapezoid(y_tpr, x=x_fpr)
    auc_pr = np.trapezoid(y_precision, x=x_recall)

    return (
        x_fpr,
        y_tpr,
        auc_roc,
        y_precision,
        x_recall,
        auc_pr,
        threshold_lst,
        tp_lst,
        fp_lst,
        tn_lst,
        fn_lst,
    )
