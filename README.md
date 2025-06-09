# PRIVET: PRIVacy metric based on Extreme value Theory

**Abstract:** Deep generative models are often trained on sensitive data, such as genetic sequences, health data, or more broadly, any copyrighted, licensed or protected content. This raises critical concerns around privacy-preserving synthetic data, and more specifically around privacy leakage, an issue closely tied to overfitting. Existing methods almost exclusively rely on global criteria to estimate the risk of privacy failure associated to a model, offering only quantitative non interpretable insights. The absence of rigorous evaluation methods for data privacy at the sample-level may hinder the practical deployment of synthetic data in real-world applications. Using extreme value statistics on nearest-neighbor distances, we propose PRIVET, a generic sample-based, modality-agnostic algorithm that assigns an individual privacy leak score to each synthetic sample. We empirically demonstrate that PRIVET reliably detects instances of memorization and overfitting across diverse data modalities, including settings with very high dimensionality and limited sample sizes such as genetic data. We compare our method to existing approaches under controlled settings and show its advantage in providing both dataset level and sample level assessments through qualitative and quantitative outputs. Additionally, our analysis reveals limitations in existing computer vision embeddings to yield perceptually meaningful distances when identifying near-duplicate samples.

## Quick start

Assuming `Train` and `Synthetic` are loaded as torch tensors of shape `(N, D)` where N is sample size and D is the number of features   
Having a `Test` set is not mandatory (`Test` is either `None` or loaded too)   
`Train`, `Test` and `Synthetic` can have different sample sizes.   

The partition of the data (**a hyperparameter of the method**) on which the EVT fit is applied should be decided upon examination of the cumulative of 1-NN distances between train samples. Yet, default ($1\%$ to $20\%$) might still be a good choice.

```python
import sys
PATH = "/your/path/to"
sys.path.append(f"{PATH}/PRIVET")
from src.privet_class import *

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

NPL = delta_pi <= -3 #boolean for each synthetic samples: privacy leak or not 
```

## Reproducibility

**Works with the following tag: v1.1-refactor**

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

**Done**
- [x] Add Gumbel case (it's still estimated via linear regression though) and choose best fit
- [x] try using same rank on test for the basic score (cf score4)
- [x] change binom.sf and binom.cdf to binom.logsf and binom.logcdf when no under/over-flow alarm & implement them manually with numba when alarm case
- [x] add all the scores for `compute_scores()` function
- [x] do a class for privet (compute 1-NN, EVT fit, decide renormalization based on user or default, decide partition based on user or default, same for threshold ...)
- [x] make a quick start example (subsection in the readme) on how to use PRIVET in a minimal setting, and how to parse/read the results
