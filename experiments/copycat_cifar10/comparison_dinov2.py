#author ran this script on 1x A100-40GB

import sys
PATH = "/scratch/aszatkow/work/projects"
sys.path.append(f"{PATH}/PRIVET")

import time
from collections import defaultdict
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, ImageFolder
import io
from PIL import Image
import pickle
import os

from src.misc_utils import *
from src.data_utils import *
from src.nn_utils import *
from src.stats_utils import *
from src.plot_utils import *
from src.privet import *

from metrics.AuthPct import *
from metrics.AATS import *

# https://github.com/marcojira/fld/tree/main
from fld.metrics.CTTest import CTTest
from fld.metrics.FLD import FLD
from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
# https://github.com/Ciela-Institute/PQM
from pqm import pqm_pvalue, pqm_chi2

ngpu=1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

###########
## UTILS ##
###########

# Downsample synthetic dataset: select 1000 images per class
def downsample_dataset(dataset, n_per_class=1000):
    class_counts = defaultdict(int)
    selected_indices = []
    # dataset.samples is a list of (filepath, class) pairs for ImageFolder.
    for idx, (_, label) in enumerate(dataset.samples):
        if class_counts[label] < n_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
    return torch.utils.data.Subset(dataset, selected_indices)

def build_mixed_feature_set_determinist(gen_feat, train_feat, beta=0.0):
    """
    Create a combined feature set where the first `alpha` fraction are training features 
    and the remaining are synthetic features (deterministic order).
    
    Parameters:
        gen_feat (Tensor): Features from the synthetic dataset (shape: [N, D]).
        train_feat (Tensor): Features from the training dataset (shape: [M, D]).
        alpha (float): Fraction of the output set to take from training data.
    
    Returns:
        Tensor: Combined feature set of size equal to gen_feat, with training features first.
    """
    N = gen_feat.shape[0]
    n_real = int(beta * N)
    n_syn = N - n_real

    # Take the first n_real training samples and first n_syn synthetic samples
    real_feats = train_feat[:n_real]  # Deterministic: first n_real samples
    syn_feats = gen_feat[:n_syn]      # Deterministic: first n_syn samples
    
    # Concatenate (training features first, synthetic features after)
    combined = torch.cat([real_feats, syn_feats], dim=0)
    return combined  # No shuffling

#####################################
## COMPUTER VISION TRANSFORMATIONS ##
#####################################

class JPEGQuality(object):
    def __init__(self, quality=75):
        """
        Args:
            quality (int): JPEG quality, 1 (worst) to 95 (best). 75 is a good default.
        """
        assert 1 <= quality <= 95, "quality must be between 1 and 95"
        self.quality = quality

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Re-encoded JPEG at self.quality.
        """
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer)

jpg75 = T.Compose([
    JPEGQuality(quality=75),
])

crop28 = T.Compose([
    T.CenterCrop(28),
    T.Pad(2),
])

class Posterize(object):
    def __init__(self, bits=5):
        self.bits = bits
    def __call__(self, img):
        return T.functional.posterize(img, self.bits)

posterize = T.Compose([
    Posterize(bits=5),
])

elastic = T.Compose([
    T.ElasticTransform(),
])

transforms_lst = [posterize, crop28, jpg75, elastic]

###############
## LOAD DATA ##
###############

train_dataset = CIFAR10(root="data", train=True, download=True)
test_dataset = CIFAR10(root="data", train=False, download=True)

#######################
## DINOv2 embeddings ##
#######################

feature_extractor = DINOv2FeatureExtractor()
train_feat = feature_extractor.get_features(train_dataset, name="train", recompute=False)
test_feat = feature_extractor.get_features(test_dataset, name="train", recompute=False)

############################
## COMPUTE 1-NN distances ##
############################
#Train-Train
dist_NN_tr_tr,  dist_NN_tr_tr_idx = gpu_nearest_neighbors(train_feat, k=1, distance='standard_euclidean',chunk_size=128,device=device.type,verbose=False)
p_tr_tr_NN_dist, p_tr_tr_NN_idx = sorting(dist_NN_tr_tr, dist_NN_tr_tr_idx)
p_tr_tr_NN_dist, p_tr_tr_NN_idx = np.array(p_tr_tr_NN_dist), np.array(p_tr_tr_NN_idx)

################################
## EVT FIT CDF on Train-Train ##
################################

N_train = train_feat.shape[0]

partition_start = 0.001
partition_end = 0.02

start = int(np.ceil(partition_start*N_train)) #int(0.01*N) # if N is small start = 0 --> problem with log
end = int(partition_end*N_train)

# Fit parameters (adjust start/end indices to avoid extremes)
intercept, alpha, std_err_intercept, std_err_alpha, sigma_Y_pred = fit_nearest_neighbor_cdf_weibull(p_tr_tr_NN_dist.reshape(-1,), start_idx=start, end_idx=end)

print(f"Estimated intercept = {intercept:.2f} ± {std_err_intercept:.2f}")
print(f"Estimated alpha = {alpha:.2f} ± {std_err_alpha:.2f}")

transforms_lst = [posterize]#[posterize, crop28, jpg75, elastic]
transform_names = ['Posterize']#['Posterize', 'Center crop 28', 'JPG 75', 'Elastic transform']


# metrics storage
beta_values = [0,0.1,0.2,0.5]#[0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
metrics = {
    name: {
        'NPL':   [],
        'FLD':   [],
        'Auth':  [],
        'CT':    [],
        'PQM':   []
    } for name in transform_names
}

time_PRIVET, time_FLD, time_AUTH, time_CT, time_PQM = [], [], [], [], []

threshold = -3

torch.manual_seed(42)
for transform_fn, name in zip(transforms_lst, transform_names):
    print(name)
    # extract features once per transform
    train_dataset_trans = CIFAR10(root="data", train=True, download=True, transform=transform_fn)
    train_feat_trans = feature_extractor.get_features(train_dataset_trans, name=f"{name}_train", recompute=False)

    synthetic_full = ImageFolder(root="/scratch/aszatkow/work/data_home/CIFAR10-PFGMPP", transform=transform_fn)
    synthetic_dataset = downsample_dataset(synthetic_full, n_per_class=1000)
    gen_feat = feature_extractor.get_features(synthetic_dataset, name=f"{name}_syn", recompute=False)

    # shuffle train features to mix in
    shuffled_indices = torch.randperm(train_feat_trans.shape[0], generator=torch.Generator().manual_seed(42))
    train_feat_shuf = train_feat_trans[shuffled_indices]

    shuffled_indices = torch.randperm(gen_feat.shape[0], generator=torch.Generator().manual_seed(42))
    gen_feat = gen_feat[shuffled_indices]

    for beta in beta_values:
        print(beta)
        mixed_feat = build_mixed_feature_set_determinist(
            gen_feat, train_feat_shuf, beta=beta
        )

        start = time.time()
        # PRIVET → NPL
        store, p_tr, p_te = PRIVET(
            train_feat, test_feat, mixed_feat,
            intercept, alpha,
            renormalization=5**(1/25),
            distance='standard_euclidean',
            device=device.type
        )
        end = time.time()
        time_PRIVET.append(end-start)
        NPL = (store[:,0] <= threshold).sum()
        metrics[name]['NPL'].append(NPL)

        start = time.time()
        # FLD generalization gap
        fld_gap = FLD(eval_feat="gap").compute_metric(train_feat, test_feat, mixed_feat)
        metrics[name]['FLD'].append(fld_gap)
        end = time.time()
        time_FLD.append(end-start)

        start = time.time()
        # AuthPct
        auth_mask = AuthPct().compute_metric(train_feat, test_feat, mixed_feat)[1]
        auth_score = (~auth_mask.detach().cpu().numpy()).sum()
        metrics[name]['Auth'].append(auth_score)
        end = time.time()
        time_AUTH.append(end-start)

        start = time.time()
        # C_T
        ct_val = CTTest().compute_metric(train_feat, test_feat, mixed_feat)
        metrics[name]['CT'].append(ct_val)
        end = time.time()
        time_CT.append(end-start)

        #PQMass
        start = time.time()
        chi2_stat_synth_tr = pqm_chi2(mixed_feat.to(device), train_feat.to(device), re_tessellation = 1000)
        chi2_stat_synth_te = pqm_chi2(mixed_feat.to(device), test_feat.to(device), re_tessellation = 1000)
        end = time.time()
        time_PQM.append(end-start)
        chi2_stat_synth_tr = np.array(chi2_stat_synth_tr)
        chi2_stat_synth_te = np.array(chi2_stat_synth_te)
        pqm_chi2_gap = pqm_chi2_gap = np.nanmean(chi2_stat_synth_te) - np.nanmean(chi2_stat_synth_tr)#chi2_stat_synth_te.mean() - chi2_stat_synth_tr.mean()
        metrics[name]['PQM'].append(pqm_chi2_gap)

os.makedirs("./output", exist_ok=True)  # succeeds even if directory exists.

with open('./output/comparison_dinov2_METRICS_DIC.pickle', 'wb') as handle:
    pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

rounding = 4

print(f"PRIVET execution time: average={np.round(np.mean(time_PRIVET),rounding)}s, std={np.round(np.std(time_PRIVET),rounding)}")
print(f"Auth execution time: average={np.round(np.mean(time_AUTH),rounding)}s, std={np.round(np.std(time_AUTH),rounding)}")
print(f"FLD execution time: average={np.round(np.mean(time_FLD),rounding)}s, std={np.round(np.std(time_FLD),rounding)}")
print(f"CT execution time: average={np.round(np.mean(time_CT),rounding)}s, std={np.round(np.std(time_CT),rounding)}")
print(f"PQM execution time: average={np.round(np.mean(time_PQM),rounding)}s, std={np.round(np.std(time_PQM),rounding)}")