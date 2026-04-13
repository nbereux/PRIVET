# # TODO: handle different fits

# import numpy as np

# from privet.privet import PRIVET

# threshold = -3

# ############################
# ## EVT FIT CDF on Train-Train ##
# ############################

# partition_start = 0.01
# partition_end = 0.2

# start = int(
#     np.ceil(partition_start * N)
# )  # int(0.01*N) # if N is small start = 0 --> problem with log
# end = int(partition_end * N)

# # Fit parameters (adjust start/end indices to avoid extremes)
# intercept, alpha, std_err_intercept, std_err_alpha, sigma_Y_pred = (
#     fit_nearest_neighbor_cdf(
#         p_tr_tr_NN_dist.numpy().reshape(
#             -1,
#         ),
#         start_idx=start,
#         end_idx=end,
#     )
# )

# print(f"Estimated intercept = {intercept:.2f} ± {std_err_intercept:.2f}")
# print(f"Estimated alpha = {alpha:.2f} ± {std_err_alpha:.2f}")


# ############
# ## PRIVET ##
# ############

# store_in_mat, p_syn_tr_NN_dist, p_syn_te_NN_dist = PRIVET(
#     train_torch,
#     test_torch,
#     fake,
#     intercept,
#     alpha,
#     renormalization=None,
#     distance="hamming",
#     device=device.type,
#     groundtruth=flist,
# )

# delta_pi = store_in_mat[:, 0]
# pred_with_delta_pi = delta_pi <= threshold
# NPL = pred_with_delta_pi.sum()
