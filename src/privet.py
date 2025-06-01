from .misc_utils import *
from .nn_utils import *
from .stats_utils import *
import types

#mat_syn_train is actually sorted_mat_syn_train
#TODO: handle GUMBEL parameters
def compute_scores(mat_syn_train, mat_syn_test, param1, param2, label_fit, groundtruth_flag):

    N = mat_syn_train.shape[0] #size of d_{STr}^*

    #\delta \pi, \delta \pi train, \delta \pi test, px train, px test, rank train, rank test, dist train, dist test
    store_in_mat = np.zeros((N, 10))

    train_lookup = {int(row[1]): i for i, row in enumerate(mat_syn_train)}
    test_lookup = {int(row[1]): i for i, row in enumerate(mat_syn_test)}
    
    for idx_syn in range(N):
        rank_s_te = test_lookup.get(idx_syn)
        rank_s_tr = train_lookup.get(idx_syn)

        row_train = mat_syn_train[rank_s_tr]
        if groundtruth_flag:
            groundtruth_i = row_train[-1].item()
        dist_train = row_train[0].item()
    
        row_test = mat_syn_test[rank_s_te]
        dist_test = row_test[0].item()

        if label_fit == "Gumbel":
            func = cdf_gumbel_extrapolate
        else:
            func = cdf_weibull_extrapolate
        
        px_train = func(dist_train, param1, param2)
        px_test  = func(dist_test,  param1, param2)
    
        pi_train_i=binomial_survival(px_train, rank_s_tr, N) #TODO: use correct func
        pi_test_i=binomial_survival(px_test, rank_s_te, N)
   
        delta_pi_i=np.log10(pi_train_i)-np.log10(pi_test_i) #TODO: add all scores

        store_in_mat[idx_syn,0] = delta_pi_i
        store_in_mat[idx_syn,1] = pi_train_i
        store_in_mat[idx_syn,2] = pi_test_i
        store_in_mat[idx_syn,3] = px_train
        store_in_mat[idx_syn,4] = px_test
        store_in_mat[idx_syn,5] = rank_s_tr
        store_in_mat[idx_syn,6] = rank_s_te
        store_in_mat[idx_syn,7] = dist_train
        store_in_mat[idx_syn,8] = dist_test
        if groundtruth_flag:
            store_in_mat[idx_syn,9] = groundtruth_i

    return store_in_mat

def PRIVET(train, test, synthetic, param1, param2, label_fit, renormalization = None, distance='standard_euclidean', device=None, groundtruth = None):
    
    ############################
    ## COMPUTE 1-NN distances ##
    ############################
    
    ## SYNTHETIC TO TRAIN (start)
    #inputs of gpu_nearest_neighbors() should be pytorch tensors
    # dist_NN_syn_tr,  dist_NN_syn_tr_idx are unsorted
    dist_NN_syn_tr,  dist_NN_syn_tr_idx = gpu_nearest_neighbors(synthetic, train, k=1, distance=distance,chunk_size=128,device=device,verbose=False)
    p_syn_tr_NN_dist, p_syn_tr_NN_idx = sorting(dist_NN_syn_tr, dist_NN_syn_tr_idx)
    p_syn_tr_NN_dist, p_syn_tr_NN_idx = np.array(p_syn_tr_NN_dist), np.array(p_syn_tr_NN_idx)
    ## SYNTHETIC TO TRAIN (end)

    ## SYNTHETIC TO TEST (start)
    dist_NN_syn_te, dist_NN_syn_te_idx = gpu_nearest_neighbors(synthetic, test, k=1,distance=distance,chunk_size=128,device=device,verbose=False)

    # if N_tr = \lambda * N_te with strictly positive \lambda, then renormalization = \lambda ** -1/d 
    # where d is intrinsic dimension of data
    # d should be the slope of the fit, but often is manually tuned to align d_TeTe^* with d_TrTr^*
    # if N_te = \lambda * N_tr with stricly positive \lambda, then renormalization = \lambda ** 1/d
    if renormalization != None:
        dist_NN_syn_te = dist_NN_syn_te * renormalization
    
    p_syn_te_NN_dist, p_syn_te_NN_idx = sorting(dist_NN_syn_te, dist_NN_syn_te_idx)
    p_syn_te_NN_dist, p_syn_te_NN_idx = np.array(p_syn_te_NN_dist), np.array(p_syn_te_NN_idx)
    ## SYNTHETIC TO TEST (end)

    #table containing:  d_STr^*, i_s, i_r^Tr (sorted on i_s)
    mat_syn_train = np.concatenate([dist_NN_syn_tr, np.arange(synthetic.shape[0]).reshape(-1,1), dist_NN_syn_tr_idx],axis=1)
    #table containing:  d_STe^*, i_s, i_r^Te (sorted on i_s)
    mat_syn_test = np.concatenate([dist_NN_syn_te, np.arange(synthetic.shape[0]).reshape(-1,1), dist_NN_syn_te_idx],axis=1)

    #for controlled experiment where you know which synthetic has privacy leak (groundtruth is a boolean list over each synth)
    groundtruth_flag=False
    if type(groundtruth) != types.NoneType:
        #table containing:  d_STr^*, i_s, i_r^Tr, groundtruth (sorted on i_s)
        mat_syn_train = np.hstack((mat_syn_train, groundtruth.reshape(-1,1)))
        #table containing:  d_STe^*, i_s, i_r^Te, groundtruth (sorted on i_s)
        mat_syn_test = np.hstack((mat_syn_test, groundtruth.reshape(-1,1)))
        groundtruth_flag=True

    #table containing:  d_STr^*, i_s, i_r^Tr (sorted on d_STr^*) -- plus the groundtruth if existing
    sorted_mat_syn_train = mat_syn_train[mat_syn_train[:, 0].argsort()]
    #table containing:  d_STe^*, i_s, i_r^Te (sorted on d_STe^*) -- plus the groundtruth if existing
    sorted_mat_syn_test = mat_syn_test[mat_syn_test[:, 0].argsort()]
    
    store_in_mat = compute_scores(sorted_mat_syn_train, sorted_mat_syn_test, param1, param2, label_fit, groundtruth_flag=groundtruth_flag)
    
    return store_in_mat, p_syn_tr_NN_dist, p_syn_te_NN_dist

##########################################################################
#############################!!!!!!!!!####################################
##########################################################################
# inverse privet for membership attack (below code). Use only train and synthetic
# TODO: below code is redundant, above code should handle this case (ie no test set and possibility to inverse role of synthetic and train)
##########################################################################
#############################!!!!!!!!!####################################
##########################################################################

def compute_scores_inverse(mat_train_syn, intercept, alpha, groundtruth_flag):

    N = mat_train_syn.shape[0] #size of d_{TrS}^*

    #\delta \pi, \delta \pi train, \delta \pi test, px train, px test, rank train, rank test, dist train, dist test
    store_in_mat = np.zeros((N, 7))

    train_lookup = {int(row[1]): i for i, row in enumerate(mat_train_syn)}
    
    for idx_tr in range(N):
        rank_s_syn = train_lookup.get(idx_tr)

        row_train = mat_train_syn[rank_s_syn]
        if groundtruth_flag:
            groundtruth_i = row_train[-1].item()
        dist_train = row_train[0].item()
        idx_train = row_train[1].item() #this is in fact train
        idx_synth_tr = row_train[2].item() #this is synth
    
        px_train = cdf_weibull_extrapolate(dist_train, intercept, alpha) #TODO: handle different cases
    
        pi_train_i=binomial_survival(px_train, rank_s_syn, N) #TODO: use correct func

        store_in_mat[idx_tr,0] = pi_train_i
        store_in_mat[idx_tr,1] = px_train
        store_in_mat[idx_tr,2] = rank_s_syn
        store_in_mat[idx_tr,3] = dist_train
        store_in_mat[idx_tr,4] = idx_synth_tr
        store_in_mat[idx_tr,5] = idx_train
        if groundtruth_flag:
            store_in_mat[idx_tr,6] = groundtruth_i

    return store_in_mat

def PRIVET_inverse(train, synthetic, intercept, alpha, renormalization = None, distance='standard_euclidean', device=None, groundtruth = None):
    
    ############################
    ## COMPUTE 1-NN distances ##
    ############################
    
    ## SYNTHETIC TO TRAIN (start)
    #inputs of gpu_nearest_neighbors() should be pytorch tensors
    # dist_NN_syn_tr,  dist_NN_syn_tr_idx are unsorted
    dist_NN_tr_syn,  dist_NN_tr_syn_idx = gpu_nearest_neighbors(train, synthetic, k=1, distance=distance,chunk_size=128,device=device,verbose=False)

    if renormalization != None:
        dist_NN_tr_syn = dist_NN_tr_syn * renormalization
        
    p_tr_syn_NN_dist, p_tr_syn_NN_idx = sorting(dist_NN_tr_syn,  dist_NN_tr_syn_idx)
    p_tr_syn_NN_dist, p_tr_syn_NN_idx = np.array(p_tr_syn_NN_dist), np.array(p_tr_syn_NN_idx)
    ## SYNTHETIC TO TRAIN (end)    

    #table containing:  d_STr^*, i_s, i_r^Tr (sorted on i_s)
    mat_train_syn = np.concatenate([dist_NN_tr_syn, np.arange(train.shape[0]).reshape(-1,1), dist_NN_tr_syn_idx],axis=1)

    #for controlled experiment where you know which synthetic has privacy leak (groundtruth is a boolean list over each synth)
    groundtruth_flag=False
    if type(groundtruth) != types.NoneType:
        #table containing:  d_STr^*, i_s, i_r^Tr, groundtruth (sorted on i_s)
        mat_train_syn = np.hstack((mat_train_syn, groundtruth.reshape(-1,1)))
        groundtruth_flag=True

    #table containing:  d_STr^*, i_s, i_r^Tr (sorted on d_STr^*) -- plus the groundtruth if existing
    sorted_mat_train_syn = mat_train_syn[mat_train_syn[:, 0].argsort()]
    
    store_in_mat = compute_scores_inverse(sorted_mat_train_syn, intercept, alpha, groundtruth_flag=groundtruth_flag)
    
    return store_in_mat, p_tr_syn_NN_dist