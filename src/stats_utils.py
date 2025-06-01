#TODO: add GUMBEL

import numpy as np
from scipy.stats import binom

CUTOFF = 1e-20

def cdf_weibull_extrapolate(x, A, alpha_slope): #used to project on the fit: cf \hat F(x) in paper
    px = 1-np.exp(-np.exp(A)*x**alpha_slope) #didn't forgot to multiply by N, \hat A is in fact log(NA)
    return px

def cdf_gumbel_extrapolate(x, A, B): #used to project on the fit: cf \hat F(x) in paper
    px = 1-np.exp(-np.exp(A)*np.exp(x*B))
    return px


#this is the survival function of a binomial random var cf how \pi_i^{\rm ref} is computed in paper 
#ie "[...] ~(\ref{eq:proba_r}) now reads [...] " in paper
#TODO: binom.sf and binom.cdf lead to underflow when proba are extremely low hence in further versions, logcdf and logsf will be manually coded in numba
def binomial_survival(px, k, N):
    z=binom.sf(k, N, px, loc=0)
    if z<=CUTOFF:z=CUTOFF
    return z


#TODO: do maximum likelihood instead
def fit_nearest_neighbor_cdf_weibull(z_star, start_idx, end_idx):
    """
    Fit parameters A and alpha for the model:
    P(z_i^* < x | N) ≈ 1 - exp(-N * A * x^alpha)
    
    Args:
        z_star: Array of minimum distances (shape: [n_samples]).
        start_idx: Start index of the partition (inclusive).
        end_idx: End index of the partition (exclusive).
    
    Returns:
        A: Estimated amplitude parameter.
        alpha: Estimated power-law exponent.
        std_err_A: Standard error of A.
        std_err_alpha: Standard error of alpha.
    """
    # Sort distances and compute survival probability
    z_sorted = np.sort(z_star)
    N = len(z_sorted)
    
    # Survival probability: P(z_i^* >= x) = 1 - CDF(x)
    survival = 1 - (np.arange(N) + 1) / N  # 1 - (i+1)/N
    
    # Select partition and filter invalid points (survival > 0)
    survival_part = survival[start_idx:end_idx]
    z_part = z_sorted[start_idx:end_idx]
    
    # Linearize: Y = ln(-ln(survival)), X = ln(z)
    Y = np.log(-np.log(survival_part))
    X = np.log(z_part)
    
    # Linear regression (Y = intercept + alpha * X)
    X_design = np.vstack([np.ones_like(X), X]).T
    beta, _, _, _ = np.linalg.lstsq(X_design, Y, rcond=None)
    intercept, alpha = beta[0], beta[1]
    
    # Covariance matrix for error estimates
    residuals = Y - X_design @ beta
    mse = np.sum(residuals**2) / (len(Y) - 2)
    XTX_inv = np.linalg.inv(X_design.T @ X_design)
    cov_matrix = mse * XTX_inv
    std_err_intercept, std_err_alpha = np.sqrt(np.diag(cov_matrix))

    X_bis = np.log(z_sorted)
    # Compute the standard error of prediction for each point
    sigma_Y_pred = np.sqrt(cov_matrix[0, 0] + 2 * X_bis * cov_matrix[0, 1] + cov_matrix[1, 1] * X_bis**2)

    return intercept, alpha, std_err_intercept, std_err_alpha, sigma_Y_pred, mse

def fit_nearest_neighbor_cdf_gumbel(z_star, start_idx, end_idx):
    """
    Fit parameters (ln A, B) for the model:
      P(z_i^* < x) ≈ 1 - exp(-A * exp(B * x))
    
    Args:
      z_star    : 1D array of nearest‐neighbor distances.
      start_idx : start index (inclusive) of sorted data for fitting.
      end_idx   : end index (exclusive) of sorted data for fitting.
    
    Returns:
      intercept        : estimated ln(A).
      B                : estimated slope.
      std_err_intercept: standard error of intercept.
      std_err_B        : standard error of B.
      sigma_Y_pred     : array of standard errors for Y_pred at each sorted z.
    """
    # 1) Sort and compute survival: S(i) = 1 - (i+1)/N
    z_sorted = np.sort(z_star)
    N = len(z_sorted)
    survival = 1 - (np.arange(N) + 1) / N
    
    # 2) Take the slice [start_idx:end_idx]
    survival_part = survival[start_idx:end_idx]
    z_part = z_sorted[start_idx:end_idx]
    
    # 3) Linearize: Y = ln(-ln(survival)), X = z
    Y = np.log(-np.log(survival_part))
    X = z_part
    
    # 4) Do ordinary least squares: Y = intercept + B * X
    X_design = np.vstack([np.ones_like(X), X]).T
    beta, _, _, _ = np.linalg.lstsq(X_design, Y, rcond=None)
    intercept, B = beta[0], beta[1]
    
    # 5) Estimate covariance of (intercept, B)
    residuals = Y - X_design @ beta
    mse = np.sum(residuals**2) / (len(Y) - 2)
    XTX_inv = np.linalg.inv(X_design.T @ X_design)
    cov_matrix = mse * XTX_inv
    std_err_intercept, std_err_B = np.sqrt(np.diag(cov_matrix))
    
    # 6) Compute standard error of predicted Y at each z_sorted:
    X_bis = z_sorted
    sigma_Y_pred = np.sqrt(
        cov_matrix[0, 0]
        + 2 * X_bis * cov_matrix[0, 1]
        + (X_bis**2) * cov_matrix[1, 1]
    )
    
    return intercept, B, std_err_intercept, std_err_B, sigma_Y_pred, mse