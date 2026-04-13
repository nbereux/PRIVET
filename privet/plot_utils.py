import matplotlib.pyplot as plt
import numpy as np

from privet.misc_utils import log_rank_in_cumulative


def plot_single_CDF_and_EVD(p_tr_tr, label_fit, param1, param2, styles, FONTSIZE=13):
    plt.rcParams.update(
        {
            "axes.labelsize": FONTSIZE,
            "axes.titlesize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
        }
    )

    # Define style parameters
    real_style = styles[0]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

    log_p_real = log_rank_in_cumulative(p_tr_tr.shape[0])

    ax.scatter(p_tr_tr, log_p_real, **real_style)

    if label_fit == "Weibull":
        x = p_tr_tr.reshape(
            -1,
        ).astype(np.float64)
        Y_pred_log10 = np.log10(1 - np.exp(-np.exp(param1) * x**param2))
        ax.plot(
            p_tr_tr,
            Y_pred_log10,
            color="darkgreen",
            label=label_fit,
            linestyle="dashed",
            alpha=0.7,
        )
        ax.set_ylim([np.log10(1 / p_tr_tr.shape[0]) - 0.1, 0.1])

    if label_fit == "Gumbel":
        x = p_tr_tr.reshape(
            -1,
        ).astype(np.float64)
        Y_pred_log10 = np.log10(1 - np.exp(-np.exp(param1) * np.exp(param2 * x)))
        ax.plot(
            p_tr_tr,
            Y_pred_log10,
            color="darkgreen",
            label=label_fit,
            linestyle="dashed",
            alpha=0.7,
        )
        ax.set_ylim([np.log10(1 / p_tr_tr.shape[0]) - 0.1, 0.1])

    ax.set_xlabel("$(\log_{10})d_{ij}$")
    ax.set_ylabel("$(\log_{10}) F_{d_{ij}}$")
    ax.set_xscale("log")
    ax.grid(True)
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(0.95, 0.2),
        markerscale=10,
        fontsize=FONTSIZE,
    )
    return fig, ax, FONTSIZE


def plot_CDFs(p_tr_tr, p_syn_tr, styles, p_syn_te=None, FONTSIZE=13):
    plt.rcParams.update(
        {
            "axes.labelsize": FONTSIZE,
            "axes.titlesize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
        }
    )

    # Define style parameters
    real_style = styles[0]
    synth_style = styles[1]
    te_synth_style = styles[2]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

    log_p_real = log_rank_in_cumulative(p_tr_tr.shape[0])
    log_p_synth = log_rank_in_cumulative(p_syn_tr.shape[0])

    ax.scatter(p_tr_tr, log_p_real, **real_style)
    ax.scatter(p_syn_tr, log_p_synth, **synth_style)
    if p_syn_te is not None:
        ax.scatter(p_syn_te, log_p_synth, **te_synth_style)

    ax.set_xlabel("$(\log_{10})d_{ij}$")
    ax.set_ylabel("$(\log_{10}) F_{d_{ij}}$")
    ax.set_xscale("log")
    ax.grid(True)
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(0.95, 0.2),
        markerscale=10,
        fontsize=FONTSIZE,
    )
    return fig, ax, FONTSIZE
