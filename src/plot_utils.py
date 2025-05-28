from .misc_utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import types

def plot_CDFs(p_tr_tr, p_syn_tr, styles, p_syn_te=None, FONTSIZE = 13):
    
    plt.rcParams.update({
        'axes.labelsize': FONTSIZE,
        'axes.titlesize': FONTSIZE,
        'xtick.labelsize': FONTSIZE,
        'ytick.labelsize': FONTSIZE
    })

    # Define style parameters
    real_style = styles[0]
    synth_style = styles[1]
    te_synth_style = styles[2]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
    
    log_p_real = log_rank_in_cumulative(p_tr_tr.shape[0])
    log_p_synth = log_rank_in_cumulative(p_syn_tr.shape[0])

    ax.scatter(p_tr_tr, log_p_real, **real_style)
    ax.scatter(p_syn_tr, log_p_synth, **synth_style)
    if type(p_syn_te) != types.NoneType:
        ax.scatter(p_syn_te, log_p_synth, **te_synth_style)

    ax.set_xlabel('$(\log_{10})d_{ij}$')
    ax.set_ylabel('$(\log_{10}) F_{d_{ij}}$')
    ax.set_xscale('log')
    ax.grid(True)
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.95, 0.2), markerscale=10,fontsize=FONTSIZE)
    return fig, ax, FONTSIZE