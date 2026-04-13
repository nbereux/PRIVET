# THIS CODE COME FROM: https://github.com/marcojira/fld/blob/main/fld/metrics/AuthPct.py
# the only modification brought here is: "authen" is also return by compute_metric()
# so that I can take the negation of "authen" to retrieve synthetic samples that are deemed in-authentic ie privacy leaked

# Faster implementation of Authenticity metric defined here https://arxiv.org/abs/2102.08921

# To resolve imports, credits to:
#  - https://github.com/marcojira/fld/
#  - https://github.com/marcojira/fld/blob/main/fld/metrics/Metric.py
#  - https://github.com/marcojira/fld/blob/main/fld/utils.py


import numpy as np
import torch
from torch import Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Of samples to take from the compared train/gen sets (full sets aren't used for memory reasons)
SIZE = 10000


def shuffle(tensor: Tensor, size=None):
    """Gets randomly ordered subset of `tensor` of `size`"""
    if size is not None:
        size = min(size, len(tensor))

    idx = np.random.choice(len(tensor), size if size else len(tensor), replace=False)
    return tensor[idx]


class Metric:
    """Generic Metric class"""

    def __init__(self):
        # To be implemented by each metric
        self.name = None
        pass

    def compute_metric(
        self,
        train_feat,
        test_feat,
        gen_feat,
    ):
        """Computes the metric value for the given sets of features (TO BE IMPLEMENTED BY EACH METRIC)
        - train_feat: Features from set of samples used to train generative model
        - test_feat: Features from test samples
        - gen_feat: Features from generated samples

        returns: Metric value
        """
        pass


class AuthPct(Metric):
    """
    Computes the % of samples where the distance to the sample's nearest neighbor in the train set
    is smaller than the distance between that train sample and its nearest train sample
    """

    def __init__(self):
        super().__init__()
        self.name = "AuthPct"

    def compute_metric(
        self,
        train_feat,
        test_feat,  # Test samples not used by AuthPct
        gen_feat,
    ):
        train_feat, gen_feat = train_feat.to(DEVICE), gen_feat.to(DEVICE)
        train_feat, gen_feat = shuffle(train_feat, SIZE), shuffle(gen_feat, SIZE)
        real_dists = torch.cdist(train_feat, train_feat)

        # Hacky way to get it to ignore distance to self in nearest neighbor calculation
        real_dists.fill_diagonal_(float("inf"))
        gen_dists = torch.cdist(train_feat, gen_feat)

        real_min_dists = real_dists.min(axis=0)
        gen_min_dists = gen_dists.min(dim=0)

        authen = real_min_dists.values[gen_min_dists.indices] < gen_min_dists.values
        return (100 * torch.sum(authen) / len(authen)).item() - 50, authen
