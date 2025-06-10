# PRIVET: PRIVacy metric based on Extreme value Theory

**Abstract:** Deep generative models are often trained on sensitive data, such as genetic sequences, health data, or more broadly, any copyrighted, licensed or protected content. This raises critical concerns around privacy-preserving synthetic data, and more specifically around privacy leakage, an issue closely tied to overfitting. Existing methods almost exclusively rely on global criteria to estimate the risk of privacy failure associated to a model, offering only quantitative non interpretable insights. The absence of rigorous evaluation methods for data privacy at the sample-level may hinder the practical deployment of synthetic data in real-world applications. Using extreme value statistics on nearest-neighbor distances, we propose PRIVET, a generic sample-based, modality-agnostic algorithm that assigns an individual privacy leak score to each synthetic sample. We empirically demonstrate that PRIVET reliably detects instances of memorization and overfitting across diverse data modalities, including settings with very high dimensionality and limited sample sizes such as genetic data. We compare our method to existing approaches under controlled settings and show its advantage in providing both dataset level and sample level assessments through qualitative and quantitative outputs. Additionally, our analysis reveals limitations in existing computer vision embeddings to yield perceptually meaningful distances when identifying near-duplicate samples.

## Quick start

### Hyperparameters

- Partition of the 1-NN distances for the fit
- Threshold

### Example

Assuming `Train` and `Synthetic` are loaded as torch tensors of shape `(N, D)` where N is sample size and D is the number of features (ie. flattened vectors) 
Having a `Test` set is not mandatory (`Test` is either `None` or loaded too)   
`Train`, `Test` and `Synthetic` can have different sample sizes.   

The partition of the data (**a hyperparameter of the method**) on which the EVT fit is applied should be decided upon examination of the cumulative of 1-NN distances between train samples. Yet, default (1\% to 20\%) might still be a good choice.

```python
import sys
PATH = "/your/path/to"
sys.path.append(f"{PATH}/PRIVET")
from src.privet import *

distance = "hamming" # else: "standard_euclidean"

privet = PRIVET(train, \
    device, \
    partition_fit_start=0.01, \
    partition_fit_end=0.2, \
    distance=distance)

# if N_Tr = N_Te or Test is None, then renormalization is None
out = privet.compute_scores_syn_to_ref(synth, test, renormalization = None)

delta_pi = out[:,privet.COL_DELTA_PI] #(out_score cf paper Main.2.3 $\Delta \pi_r$)

threshold = -3

NPL = delta_pi <= threshold #boolean for each synthetic samples: privacy leak or not 
```

Here is how to plot $d_{TrTr}^*$ (the distribution of nearest neighbor distances from train to train samples) and its EVD fit.   

```python
import sys
PATH = "/your/path/to"
sys.path.append(f"{PATH}/PRIVET")
from src.privet import *

distance = "hamming" # else: "standard_euclidean"

privet = PRIVET(train, \
    device, \
    partition_fit_start=0.01, \
    partition_fit_end=0.2, \
    distance=distance)

styles = [{'color': 'olive', 'label': r'$d^*_{TrTr}$', 'marker': 'x', 's': 1}]

plot_single_CDF_and_EVD(privet.p_tr_tr_NN_dist, privet.label_best_fit, privet.param1, privet.param2, styles, FONTSIZE = 13)
```

### Output scores

Constants for columns in the “score matrix” output (see compute_scores_syn_to_ref)    

```python
(
    COL_DELTA_PI,                     #(out_score)
    COL_DELTA_P,                      #(out_score_implicit_decimation_different_ranks)
    COL_DELTA_PI_RANK_R,              #(out_score)
    COL_DELTA_P_RANK_R,               #(out_score)
    COL_BAR_DELTA_PI_RANK_R,          #(out_score_underfitting)
    COL_SCORE4,                       #(out_score_alternative)  
    COL_N_OVERFIT_RANK_R,             #(out_score_aggreg_overfit)
    COL_N_PRIVACY_LEAKS_RANK_R,       #(out_score_aggreg_underfit)
    COL_MIN_OVERFIT_LEAKS,            #(out_score_aggreg_??) (??)
    COL_PX_TRAIN,                     #(stats_utils)
    COL_P_TRAIN,                      #(out_score_order_stat_implicit_decimation)
    COL_PI_TRAIN,                     #(out_score)
    COL_PX_TEST,                      #(stats_utils)
    COL_PI_TEST,                      #(utils_order_stats)
    COL_ONE_MINUS_PI_TRAIN,           #(utils_order_stats)
    COL_ONE_MINUS_PI_TEST,            #(utils_order_stats)
    COL_PX_TEST_RANK_R,               #(stats_utils)
    COL_PI_TEST_RANK_R,               #(utils_order_stats)
    COL_ONE_MINUS_PI_TEST_RANK_R,     #(utils_order_stats)
    COL_RANK_S_TR,                    #(utils)
    COL_RANK_S_TE,                    #(utils)
    COL_N_PRIVACY_LEAKS_RANK_R_BIS,   #(out_score_aggreg_underfit) (??)
    COL_DIST_TR_RANK_R,               #(utils)
    COL_DIST_TE_RANK_R_PRIME,         #(utils)
    COL_IDX_TRAIN,                    #(utils)
    COL_IDX_TEST,                     #(utils)
    COL_IDX_SYN,                      #(utils)
    COL_GT                            #(utils)
)
```

Remark: `COL_DELTA_PI` is the score used in the paper

## Reproducibility

**!!Works with the following tag ONLY: v1.1-refactor!!**

- Main.Fig.2 and Appendix.Fig.7 are reproducible through: compile `PRIVET/experiments/genetic_leakage/65K_SNPs_1668_samples.py` to compute the privacy maps ($\sim 3$ hours on $1 \times$ A100-40GB) then plot them via `PRIVET/experiments/genetic_leakage/PLOT_65K_SNPs_1668_samples.ipynb`. Change the grid from $3 \times 3$ to $31 \times 31$

- Main.Fig.3 is reproducible through: compile `PRIVET/experiments/copycat_cifar10/comparison_dinov2.py` for long computations (< 3 hours on $1 \times$ A100-40GB) then plot them via `PRIVET/experiments/copycat_cifar10/PLOT_comparison.ipynb`. Change the transformations and the $\beta$ values by un-commenting. Repeat the same for Wavelet embeddings

- Main.Fig.4 is reproducible through `PRIVET/experiments/membership_attack/65K_controlled_leakage.ipynb` and `PRIVET/experiments/membership_attack/RBM_HGD.ipynb`

- Appendix.Fig.5 and Appendix.Fig.6 are reproducible through`PRIVET/experiments/genetic_leakage/low_sample_size.ipynb`


## TODO

**Priority 1**
- [ ] experiment on what should be the renormalization factor in gumbel case

**Priority 2**
- [ ] change linear regression to maximum likelihood
- [ ] update reproducible code with new class
- [ ] make privet into a package so that you don't have to add path to privet
- [ ] use FAISS to compute 1-NNs

**Done**
- [x] Add Gumbel case (it's still estimated via linear regression though) and choose best fit
- [x] try using same rank on test for the basic score (cf score4)
- [x] change binom.sf and binom.cdf to binom.logsf and binom.logcdf when no under/over-flow alarm & implement them manually with numba when alarm case
- [x] add all the scores for `compute_scores()` function
- [x] do a class for privet (compute 1-NN, EVT fit, decide renormalization based on user or default, decide partition based on user or default, same for threshold ...)
- [x] make a quick start example (subsection in the readme) on how to use PRIVET in a minimal setting, and how to parse/read the results
