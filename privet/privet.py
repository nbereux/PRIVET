# -------------------------------------------------------------------------------
# PRIVET.py
# -------------------------------------------------------------------------------
import numpy as np
import torch
from torch import Tensor

from privet.misc_utils import sorting
from privet.nn_utils import gpu_nearest_neighbors
from privet.stats_utils import (
    cdf_gumbel_extrapolate,
    cdf_weibull_extrapolate,
    fit_nearest_neighbor_cdf_gumbel,
    fit_nearest_neighbor_cdf_weibull,
    safe_log10_cdf,
    safe_log10_sf,
)


class PRIVET:
    """

    Parameters
    ----------
    train : np.ndarray or Tensor
        The training data (each row = one data point).
    device : str or torch.device
        Which device to run GPU‐accelerated NN searches on (e.g. "cuda:0" or "cpu").
    partition_fit_start : float, default=0.01
        Lower quantile (as fraction of train size) to start EVT fit.
    partition_fit_end : float, default=0.2
        Upper quantile (as fraction of train size) to end EVT fit.
    distance : str, default="standard_euclidean"
        Distance metric to use in nearest‐neighbor (e.g. "hamming",
        "standard_euclidean", or "feature_normalized_euclidean").
    """

    # Constants for columns in the “score matrix” output (see compute_scores)
    (
        COL_DELTA_PI,  # (out_score)
        COL_DELTA_P,  # (out_score_implicit_decimation_different_ranks)
        COL_DELTA_PI_RANK_R,  # (out_score)
        COL_DELTA_P_RANK_R,  # (out_score)
        COL_BAR_DELTA_PI_RANK_R,  # (out_score_underfitting)
        COL_SCORE4,  # (out_score_alternative)
        COL_N_OVERFIT_RANK_R,  # (out_score_aggreg_overfit)
        COL_N_PRIVACY_LEAKS_RANK_R,  # (out_score_aggreg_underfit)
        COL_MIN_OVERFIT_LEAKS,  # (out_score_aggreg_??) (??)
        COL_PX_TRAIN,  # (stats_utils)
        COL_P_TRAIN,  # (out_score_order_stat_implicit_decimation)
        COL_PI_TRAIN,  # (out_score)
        COL_PX_TEST,  # (stats_utils)
        COL_PI_TEST,  # (utils_order_stats)
        COL_ONE_MINUS_PI_TRAIN,  # (utils_order_stats)
        COL_ONE_MINUS_PI_TEST,  # (utils_order_stats)
        COL_P_TEST,  # (utils_order_stats)
        COL_PX_TEST_RANK_R,  # (stats_utils)
        COL_PI_TEST_RANK_R,  # (utils_order_stats)
        COL_ONE_MINUS_PI_TEST_RANK_R,  # (utils_order_stats)
        COL_RANK_S_TR,  # (utils)
        COL_RANK_S_TE,  # (utils)
        COL_N_PRIVACY_LEAKS_RANK_R_BIS,  # (out_score_aggreg_underfit) (??)
        COL_DIST_TR_RANK_R,  # (utils)
        COL_DIST_TE_RANK_R_PRIME,  # (utils)
        COL_IDX_TRAIN,  # (utils)
        COL_IDX_TEST,  # (utils)
        COL_IDX_SYN,  # (utils)
        COL_GT,  # (utils)
    ) = range(29)

    def __init__(
        self,
        train,
        device,
        partition_fit_start=0.01,
        partition_fit_end=0.2,
        distance="standard_euclidean",
        groundtruth=None,
    ):
        self.train = train if isinstance(train, Tensor) else torch.tensor(train)
        self.device = device
        self.distance = distance

        # Number of training points:
        self.N_train = self.train.shape[0]

        # ——— Compute 1‐NN distances among training points ———
        dist_tr_tr, idx_tr_tr = self._compute_one_nn(
            src=self.train,
            dst=None,
            k=1,
            distance=self.distance,
        )
        sorted_distances, sorted_indices = sorting(
            dist_tr_tr.cpu().numpy(), idx_tr_tr.cpu().numpy()
        )

        self.p_tr_tr_NN_dist = np.asarray(sorted_distances)
        self.p_tr_tr_NN_idx = np.asarray(sorted_indices)

        # ——— Fit a CDF (Weibull vs. Gumbel) on training 1‐NN distances ———
        (
            self.label_best_fit,
            self.param1,
            self.param2,
        ) = self._fit_extreme_value(
            nn_distances=self.p_tr_tr_NN_dist,
            start_frac=partition_fit_start,
            end_frac=partition_fit_end,
        )

    def _compute_one_nn(
        self, src, dst, k=1, distance="standard_euclidean", chunk_size=128
    ):
        """
        Wrapper around gpu_nearest_neighbors to return distances and indices.

        src : Tensor of shape (N_src, D)
        dst : Tensor of shape (N_dst, D)
        """
        # We assume gpu_nearest_neighbors returns “(distances, indices)” unsorted.

        src = src if isinstance(src, Tensor) else torch.tensor(src)
        dst = dst if dst is None or isinstance(dst, Tensor) else torch.tensor(dst)

        dist_mat, idx_mat = gpu_nearest_neighbors(
            src,
            dst,
            k=k,
            distance=distance,
            chunk_size=chunk_size,
            device=self.device,
            verbose=False,
        )
        return dist_mat, idx_mat

    def _fit_extreme_value(
        self, nn_distances: np.ndarray, start_frac: float, end_frac: float
    ):
        """
        Fit both a Weibull and a Gumbel CDF to the lower tail of nn_distances.
        Return the label of the best fit ("Weibull" or "Gumbel") plus its estimated parameters.
        """
        N = self.N_train
        sorted_d = nn_distances.reshape(-1)  # already sorted, but we ensure 1D

        # Compute integer indices for the partition:
        start_idx = int(np.ceil(start_frac * N))
        end_idx = int(end_frac * N)

        # ----- Fit Weibull -----
        (
            intercept_w,
            alpha_w,
            stderr_int_w,
            stderr_alpha_w,
            sigmaY_w,
            mse_w,
        ) = fit_nearest_neighbor_cdf_weibull(
            sorted_d, start_idx=start_idx, end_idx=end_idx
        )

        # ----- Fit Gumbel -----
        (
            intercept_g,
            B_g,
            stderr_int_g,
            stderr_B_g,
            sigmaY_g,
            mse_g,
        ) = fit_nearest_neighbor_cdf_gumbel(sorted_d, start_idx, end_idx)

        # Compare MSEs and choose the better one:
        if mse_g < mse_w:
            best_label = "Gumbel"
            param1 = intercept_g
            param2 = B_g
        else:
            best_label = "Weibull"
            param1 = intercept_w
            param2 = alpha_w

        print(f"Best fit: {best_label}")
        return best_label, param1, param2

    def compute_scores_syn_to_ref(
        self, synthetic, test=None, renormalization=None, groundtruth=None
    ):
        """
        synthetic : np.ndarray or Tensor
            The synthetic data to be evaluated.
        test : np.ndarray or Tensor, optional
            If provided, compute synthetic‐to‐test nearest neighbors as well.
        renormalization : float, optional
            If provided, scale test distances by this factor.
        groundtruth : np.ndarray of bool or int, optional
            If provided, a 1D array indicating which synthetic points truly leak privacy.


        groundtruth_flag : bool
            If True, expect groundtruth to be a 1D array of size N_syn.
        """

        synthetic = (
            synthetic if isinstance(synthetic, Tensor) else torch.tensor(synthetic)
        )
        test = test if test is None or isinstance(test, Tensor) else torch.tensor(test)

        groundtruth_flag = False
        if groundtruth is not None:
            groundtruth_flag = True

        # 1) Build “synthetic→train” table
        mat_syn_train = self._build_syn_table(
            src=synthetic, dst=self.train, include_groundtruth=groundtruth
        )

        # If test is provided, build a second table
        if test is not None:
            mat_syn_test = self._build_syn_table(
                src=synthetic,
                dst=test,
                include_groundtruth=groundtruth,  # we only mark groundtruth on train side
            )

            N_tr = self.train.shape[0]
            N_te = test.shape[0]
            if renormalization is None:
                _lambda = N_tr / N_te
                if _lambda == 1 or self.label_best_fit == "Gumbel":
                    renormalization = 1
                if _lambda > 1 and self.label_best_fit == "Weibull":
                    renormalization = _lambda ** (-1 / self.param2)
                if _lambda < 1 and self.label_best_fit == "Weibull":
                    renormalization = _lambda ** (1 / self.param2)

            mat_syn_test[:, 0] *= renormalization

        else:
            mat_syn_test = None

        N_syn = mat_syn_train.shape[0]
        n_cols = 29
        store_in_mat = np.zeros((N_syn, n_cols), dtype=float)

        # dictionary mapping: synthetic‐index → rank in sorted table
        # mat_syn_train[:, 1] holds “i_s” (synthetic index) after sorting by NN distance
        train_lookup = {int(row[1]): i_rank for i_rank, row in enumerate(mat_syn_train)}
        if mat_syn_test is not None:
            test_lookup = {int(row[1]): i_rank for i_rank, row in enumerate(mat_syn_test)}

        # For each synthetic point i_syn = 0…(N_syn−1), find its row in mat_syn_train, mat_syn_test
        for i_syn in range(N_syn):
            # ---- Look up (train) rank of this synthetic point
            rank_s_tr = train_lookup[i_syn]
            row_tr = mat_syn_train[rank_s_tr]
            dist_train = float(row_tr[0])
            idx_train = int(row_tr[2])
            if groundtruth_flag:
                gt_i = int(row_tr[-1])

            # rank r−1 in train (clamp at 0)
            rank_tr_prev = max(rank_s_tr - 1, 0)
            row_tr_prev = mat_syn_train[rank_tr_prev]
            dist_train_prev = float(row_tr_prev[0])

            # Evaluate CDF (or survival function) on train distances:
            if self.label_best_fit == "Gumbel":
                cdf_func = cdf_gumbel_extrapolate
            else:
                cdf_func = cdf_weibull_extrapolate

            px_train = cdf_func(dist_train, self.param1, self.param2).item()
            px_train_prev = cdf_func(dist_train_prev, self.param1, self.param2).item()

            # ref is Tr and rank is r - n_excess_Tr_r_minus_one
            excess_tr_idx = min(int(1 + np.floor(N_syn * px_train_prev)), N_syn - 1)

            # Compute train‐side π, p, etc.
            pi_train = safe_log10_sf(rank_s_tr, N_syn, px_train)
            p_train = safe_log10_sf(excess_tr_idx, N_syn, px_train)
            one_minus_pi_train = safe_log10_cdf(rank_s_tr, N_syn, px_train)
            n_overfit_rank_r = rank_s_tr - int(N_syn * px_train)

            # Store train‐only columns:
            store_in_mat[i_syn, self.COL_PX_TRAIN] = px_train  # (stats_utils)
            store_in_mat[i_syn, self.COL_P_TRAIN] = p_train  # (out_score_order_stat)
            store_in_mat[i_syn, self.COL_PI_TRAIN] = pi_train  # (out_score_order_stat)
            store_in_mat[i_syn, self.COL_ONE_MINUS_PI_TRAIN] = (
                one_minus_pi_train  # (out_score_alternative_order_stat)
            )
            store_in_mat[i_syn, self.COL_N_OVERFIT_RANK_R] = (
                n_overfit_rank_r  # (out_score_aggreg)
            )
            store_in_mat[i_syn, self.COL_RANK_S_TR] = rank_s_tr  # (utils)
            store_in_mat[i_syn, self.COL_DIST_TR_RANK_R] = dist_train  # (utils)
            store_in_mat[i_syn, self.COL_IDX_TRAIN] = idx_train  # (utils)
            store_in_mat[i_syn, self.COL_IDX_SYN] = i_syn  # (utils)

            if groundtruth_flag:
                store_in_mat[i_syn, self.COL_GT] = gt_i  # (utils)

            # If test‐side exists, fill those in:
            if mat_syn_test is not None:
                # ref is Te and rank is r′
                rank_s_te = test_lookup[i_syn]
                row_te = mat_syn_test[rank_s_te]
                dist_test = float(row_te[0])
                idx_test = int(row_te[2])

                # ref is Te and rank is r
                row_te_rank_r = mat_syn_test[rank_s_tr]
                dist_test_rank_r = float(row_te_rank_r[0])

                # rank r′−1 in test (clamp at 0)
                rank_te_prev = max(rank_s_te - 1, 0)
                row_te_prev = mat_syn_test[rank_te_prev]
                dist_test_prev = float(row_te_prev[0])

                # ref is Te and rank is r-1
                row_te_rank_r_prev = mat_syn_test[max(rank_s_tr - 1, 0)]
                dist_test_rank_r_prev = float(row_te_rank_r_prev[0])

                # Evaluate test CDF / SF
                px_test = cdf_func(dist_test, self.param1, self.param2).item()
                px_test_rank_r = cdf_func(
                    dist_test_rank_r, self.param1, self.param2
                ).item()
                px_test_prev = cdf_func(dist_test_prev, self.param1, self.param2).item()
                px_test_rank_r_prev = cdf_func(
                    dist_test_rank_r_prev, self.param1, self.param2
                ).item()

                # π_test at rank r′ and r:
                pi_test = safe_log10_sf(rank_s_te, N_syn, px_test)
                pi_test_rank_r = safe_log10_sf(rank_s_tr, N_syn, px_test_rank_r)
                one_minus_pi_test = safe_log10_cdf(rank_s_te, N_syn, px_test)
                one_minus_pi_test_rank_r = safe_log10_cdf(
                    rank_s_tr, N_syn, px_test_rank_r
                )

                # ref is Te and rank is r_prime - n_excess_Te_r_prime_minus_one
                excess_te_idx = min(int(1 + np.floor(N_syn * px_test_prev)), N_syn - 1)
                p_test = safe_log10_sf(excess_te_idx, N_syn, px_test)

                # ref is Te and rank is r - n_excess_Te_r_minus_one
                excess_te_rank_r_idx = min(
                    int(1 + np.floor(N_syn * px_test_rank_r_prev)), N_syn - 1
                )
                p_test_rank_r = safe_log10_sf(excess_te_rank_r_idx, N_syn, px_test_rank_r)

                # Compute π_train, π_test differences:
                delta_pi = pi_train - pi_test
                delta_pi_rank_r = pi_train - pi_test_rank_r
                bar_delta_pi_rank_r = one_minus_pi_test_rank_r - one_minus_pi_train
                score4 = (
                    pi_train
                    - pi_test_rank_r
                    + one_minus_pi_test_rank_r
                    - one_minus_pi_train
                )
                n_privacy_leaks_rank_r = int(N_syn * px_test_rank_r) - int(
                    N_syn * px_train
                )
                delta_p = p_train - p_test
                delta_p_rank_r = p_train - p_test_rank_r

                # Store test‐side columns:
                store_in_mat[i_syn, self.COL_DELTA_PI] = delta_pi  # (out_score)
                store_in_mat[i_syn, self.COL_DELTA_P] = delta_p  # (out_score)
                store_in_mat[i_syn, self.COL_DELTA_PI_RANK_R] = (
                    delta_pi_rank_r  # (out_score)
                )
                store_in_mat[i_syn, self.COL_DELTA_P_RANK_R] = (
                    delta_p_rank_r  # (out_score)
                )
                store_in_mat[i_syn, self.COL_BAR_DELTA_PI_RANK_R] = (
                    bar_delta_pi_rank_r  # (out_score_underfitting)
                )
                store_in_mat[i_syn, self.COL_SCORE4] = score4  # (out_score_alternative)
                store_in_mat[i_syn, self.COL_N_PRIVACY_LEAKS_RANK_R] = (
                    n_privacy_leaks_rank_r  # (out_score_aggreg)
                )
                store_in_mat[i_syn, self.COL_N_PRIVACY_LEAKS_RANK_R_BIS] = min(
                    rank_s_tr, n_privacy_leaks_rank_r
                )  # (out_score_aggreg)
                store_in_mat[i_syn, self.COL_PI_TEST] = pi_test  # (out_score_order_stat)
                store_in_mat[i_syn, self.COL_ONE_MINUS_PI_TEST] = (
                    one_minus_pi_test  # (out_score_order_stat)
                )
                store_in_mat[i_syn, self.COL_P_TEST] = p_test  # (out_score_order_stat)
                store_in_mat[i_syn, self.COL_PX_TEST] = px_test  # (stats_utils)
                store_in_mat[i_syn, self.COL_PX_TEST_RANK_R] = (
                    px_test_rank_r  # (stats_utils)
                )
                store_in_mat[i_syn, self.COL_PI_TEST_RANK_R] = (
                    pi_test_rank_r  # (out_score_order_stat)
                )
                store_in_mat[i_syn, self.COL_ONE_MINUS_PI_TEST_RANK_R] = (
                    one_minus_pi_test_rank_r  # (out_score_order_stat)
                )
                store_in_mat[i_syn, self.COL_RANK_S_TE] = rank_s_te  # (utils)
                store_in_mat[i_syn, self.COL_DIST_TE_RANK_R_PRIME] = dist_test  # (utils)
                store_in_mat[i_syn, self.COL_IDX_TEST] = idx_test  # (utils)

        return store_in_mat

    def _build_syn_table(self, src, dst, include_groundtruth):
        """
        Build a table of shape (N_syn, 3 or 4) = [
            (1‐NN distance from each synthetic point → dst),
            (synthetic_index i_s),
            (index i_r^dst of the nearest‐neighbor in dst),
            (gt_i, optionally)
        ], keep unsorted and also sorted by ascending 1‐NN distance.

        Returns
        -------
        sorted_table : np.ndarray of shape (N_syn, 3 or 4)
        """
        # 1) Compute distances + indices:
        dist_syn_dst, idx_syn_dst = self._compute_one_nn(
            src=src,
            dst=dst,
            k=1,
            distance=self.distance,
        )

        # sorting() returns two lists: distances and indices in sorted order
        # sorted_dist_list, sorted_idx_list = sorting(dist_syn_dst, idx_syn_dst)
        # sorted_dist = np.array(sorted_dist_list).reshape(-1, 1)  # (N_syn, 1)
        # sorted_idx = np.array(sorted_idx_list).reshape(-1, 1)   # (N_syn, 1)

        N_syn = dist_syn_dst.shape[0]
        synthetic_indices = np.arange(N_syn).reshape(-1, 1)  # (N_syn, 1)

        if include_groundtruth is not None:
            groundtruth = include_groundtruth
            gt_col = groundtruth.reshape(-1, 1)  # (N_syn, 1)
            mat = np.hstack([dist_syn_dst, synthetic_indices, idx_syn_dst, gt_col])
        else:
            mat = np.hstack([dist_syn_dst, synthetic_indices, idx_syn_dst])

        sorted_mat = mat[mat[:, 0].argsort()]
        return sorted_mat
