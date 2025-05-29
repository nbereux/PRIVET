#author ran this script on 1x A100-40GB

import os
import sys
PATH = "your/path/to/PRIVET/repo"
sys.path.append(f"{PATH}/PRIVET")

import time

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
# https://github.com/Ciela-Institute/PQM
from pqm import pqm_pvalue, pqm_chi2

ngpu=1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

###############
###LOAD DATA###
###############
PATH = "your/path/to/data"
dat=np.load(f"{PATH}/65k_all_labels.npy",allow_pickle=True)

np.random.seed(42)
np.random.shuffle(dat)
dat = dat[:3*(dat.shape[0]//3)] #5006 is not divisible by 3, 5004 is
train, test, synth = dat[:dat.shape[0]//3,3:], dat[dat.shape[0]//3:2*(dat.shape[0]//3),3:], dat[2*(dat.shape[0]//3):,3:]
train, test, synth = train.astype(int), test.astype(int), synth.astype(int)

train_torch = torch.tensor(train).float()
test_torch = torch.tensor(test).float()
synth_torch = torch.tensor(synth).float()

N = train.shape[0]

###################################
###INITIALIZE LEAKAGE PARAMETERS###
###################################

# paper fig uses grid of 31*31 instead of 3*3

fake_range = np.linspace(0,0.4,3)
fake_range[0] = 0.001

copy_range = np.linspace(0,0.2,3)
copy_range[0] = 0.001

#############################
###INITIALIZE PRIVACY MAPS###
#############################
#GLOBAL
heatmap_NPL=np.zeros((len(fake_range),len(copy_range)))
heatmap_authen=np.zeros((len(fake_range),len(copy_range)))
heatmap_privacy_loss=np.zeros((len(fake_range),len(copy_range)))
heatmap_ct=np.zeros((len(fake_range),len(copy_range)))
heatmap_fld_gen_gap=np.zeros((len(fake_range),len(copy_range)))
heatmap_pqm_chi2_gap=np.zeros((len(fake_range),len(copy_range)))

#LOCAL
tp_grid_npl = np.zeros((len(fake_range), len(copy_range)))
fp_grid_npl = np.zeros((len(fake_range), len(copy_range)))
tn_grid_npl = np.zeros((len(fake_range), len(copy_range)))
fn_grid_npl = np.zeros((len(fake_range), len(copy_range)))

tp_grid_authen = np.zeros((len(fake_range), len(copy_range)))
fp_grid_authen = np.zeros((len(fake_range), len(copy_range)))
tn_grid_authen = np.zeros((len(fake_range), len(copy_range)))
fn_grid_authen = np.zeros((len(fake_range), len(copy_range)))

############################
## COMPUTE 1-NN distances ##
############################

#For the fit
#Train-Train
dist_NN_tr_tr,  dist_NN_tr_tr_idx = gpu_nearest_neighbors(torch.tensor(train), k=1,distance='hamming',chunk_size=128,device=device.type,verbose=False)
p_tr_tr_NN_dist, p_tr_tr_NN_idx = sorting(dist_NN_tr_tr, dist_NN_tr_tr_idx)
p_tr_tr_NN_dist, p_tr_tr_NN_idx = np.array(p_tr_tr_NN_dist), np.array(p_tr_tr_NN_idx)

#Needed to construct the pseudo-synthetic data
#Synth-Train
dist_NN_syn_tr_INIT, dist_NN_syn_tr_INIT_idx = gpu_nearest_neighbors(torch.tensor(synth), torch.tensor(train), k=1,distance='hamming',chunk_size=128,device=device.type,verbose=False)
p_syn_tr_NN_dist_INIT, p_syn_tr_NN_idx_INIT = sorting(dist_NN_syn_tr_INIT, dist_NN_syn_tr_INIT_idx)
p_syn_tr_NN_dist_INIT, p_syn_tr_NN_idx_INIT = np.array(p_syn_tr_NN_dist_INIT), np.array(p_syn_tr_NN_idx_INIT)
indices_s_tr_INIT = dist_NN_syn_tr_INIT_idx.squeeze(1)

################################
## EVT FIT CDF on Train-Train ##
################################

partition_start = 0.01
partition_end = 0.2

start = int(np.ceil(partition_start*N)) #int(0.01*N) # if N is small start = 0 --> problem with log
end = int(partition_end*N)

# Fit parameters (adjust start/end indices to avoid extremes)
intercept, alpha, std_err_intercept, std_err_alpha, sigma_Y_pred = fit_nearest_neighbor_cdf_weibull(p_tr_tr_NN_dist.reshape(-1,), start_idx=start, end_idx=end)

print(f"Estimated intercept = {intercept:.2f} ± {std_err_intercept:.2f}")
print(f"Estimated alpha = {alpha:.2f} ± {std_err_alpha:.2f}")

#######################
###FILL PRIVACY MAPS###
#######################

threshold = -3

time_lst_privet, time_lst_auth, time_lst_aats, time_lst_ct, time_lst_fld, time_lst_pqm = [], [], [], [], [], []

for i,f_fake in enumerate(fake_range):
    print(f_fake)
    n = int(np.ceil(N*f_fake))
    flist=np.zeros((N,)).astype(bool) #this is the groundtruth
    flist[:n]=True
    for j,f_copy in enumerate(copy_range):

        fake = generate_fake_synth(train, synth, indices_s_tr_INIT, f_fake=f_fake, f_copy=f_copy)

        start_i_j = time.time()
        store_in_mat, p_syn_tr_NN_dist, p_syn_te_NN_dist = PRIVET(torch.tensor(train), torch.tensor(test), torch.tensor(fake), intercept, alpha, renormalization = None, distance='hamming', device=device.type, groundtruth = flist)
        end_i_j = time.time()
        time_lst_privet.append(end_i_j-start_i_j)

        delta_pi = store_in_mat[:,0]
        flist_bis = store_in_mat[:,-1].astype(bool)

        NPL = delta_pi<=threshold

        tp_npl, fn_npl, fp_npl, tn_npl = get_predictions(NPL, flist_bis)

        heatmap_NPL[i,j] = NPL.sum()

        tp_grid_npl[i, j] = tp_npl
        fp_grid_npl[i, j] = fp_npl
        tn_grid_npl[i, j] = tn_npl
        fn_grid_npl[i, j] = fn_npl
        
        if j%10==0:
            print(rf"Δπ f_fake={f_fake}, f_copy={f_copy}, tp={tp_npl}, fp={fp_npl}, tn={tn_npl}, fn={fn_npl}")
            
        fake = torch.tensor(fake).float()
        ##################
        ###AUTHENTICITY###
        ##################
        start_i_j = time.time()
        auth, auth_samples = AuthPct().compute_metric(train_torch, test_torch, fake)
        end_i_j = time.time()
        time_lst_auth.append(end_i_j-start_i_j)
        auth_samples = ~auth_samples.detach().cpu().numpy()
        heatmap_authen[i,j] = auth_samples.sum()

        tp_auth, fn_auth, fp_auth, tn_auth = get_predictions(auth_samples, flist)

        tp_grid_authen[i, j] = tp_auth
        fp_grid_authen[i, j] = fp_auth
        tn_grid_authen[i, j] = tn_auth
        fn_grid_authen[i, j] = fn_auth

        ############
        ### AATS ###
        ############
        start_i_j = time.time()
        AAtruth_train, AAsyn_train, AATS_train = AATS(train_torch,fake,p=0)
        AAtruth_test, AAsyn_test, AATS_test = AATS(test_torch,fake,p=0)
        end_i_j = time.time()
        time_lst_aats.append(end_i_j-start_i_j)
        heatmap_privacy_loss[i, j] = AATS_test - AATS_train

        ##################
        ###  CT score  ###
        ##################
        start_i_j = time.time()
        ct = CTTest().compute_metric(train_torch, test_torch, fake)
        end_i_j = time.time()
        time_lst_ct.append(end_i_j-start_i_j)
        heatmap_ct[i, j] = ct

        #############
        ###  FLD  ### 
        #############
        std_vals = test_torch.std(dim=0)
        start_i_j = time.time()
        gen_gap = FLD(eval_feat="gap").compute_metric(train_torch[:,(std_vals!=0)], test_torch[:,(std_vals!=0)], fake[:,(std_vals!=0)])
        end_i_j = time.time()
        time_lst_fld.append(end_i_j-start_i_j)
        heatmap_fld_gen_gap[i, j] = gen_gap

        #############
        ###  PQM  ###
        #############
        start_i_j = time.time()
        chi2_stat_synth_tr = pqm_chi2(fake.to(device), train_torch.to(device), re_tessellation = 1000)
        chi2_stat_synth_te = pqm_chi2(fake.to(device), test_torch.to(device), re_tessellation = 1000)
        end_i_j = time.time()
        time_lst_pqm.append(end_i_j-start_i_j)
        chi2_stat_synth_tr = np.array(chi2_stat_synth_tr)
        chi2_stat_synth_te = np.array(chi2_stat_synth_te)
        heatmap_pqm_chi2_gap[i, j] = chi2_stat_synth_te.mean() - chi2_stat_synth_tr.mean()
        
        del fake

rounding = 4

print(f"PRIVET execution time: average={np.round(np.mean(time_lst_privet),rounding)}s, std={np.round(np.std(time_lst_privet),rounding)}")
print(f"Auth execution time: average={np.round(np.mean(time_lst_auth),rounding)}s, std={np.round(np.std(time_lst_auth),rounding)}")
print(f"AATS (Privacy Loss) execution time: average={np.round(np.mean(time_lst_aats),rounding)}s, std={np.round(np.std(time_lst_aats),rounding)}")
print(f"FLD execution time: average={np.round(np.mean(time_lst_fld),rounding)}s, std={np.round(np.std(time_lst_fld),rounding)}")
print(f"CT execution time: average={np.round(np.mean(time_lst_ct),rounding)}s, std={np.round(np.std(time_lst_ct),rounding)}")
print(f"PQM execution time: average={np.round(np.mean(time_lst_pqm),rounding)}s, std={np.round(np.std(time_lst_pqm),rounding)}")

os.makedirs("./output", exist_ok=True)  # succeeds even if directory exists.

np.save('./output/heatmap_global_NPL.npy',heatmap_NPL)
np.save('./output/heatmap_global_fld_gen_gap.npy',heatmap_fld_gen_gap)
np.save('./output/heatmap_global_privacy_loss.npy',heatmap_privacy_loss)
np.save('./output/heatmap_global_pqm_chi2_gap.npy',heatmap_pqm_chi2_gap)
np.save('./output/heatmap_global_authenticity.npy',heatmap_authen)
np.save('./output/heatmap_global_CT.npy',heatmap_ct)

np.save('./output/heatmap_local_NPL_tp.npy',tp_grid_npl)
np.save('./output/heatmap_local_NPL_tn.npy',tn_grid_npl)
np.save('./output/heatmap_local_NPL_fp.npy',fp_grid_npl)
np.save('./output/heatmap_local_NPL_fn.npy',fn_grid_npl)

np.save('./output/heatmap_local_authenticity_tp.npy',tp_grid_authen)
np.save('./output/heatmap_local_authenticity_tn.npy',tn_grid_authen)
np.save('./output/heatmap_local_authenticity_fp.npy',fp_grid_authen)
np.save('./output/heatmap_local_authenticity_fn.npy',fn_grid_authen)