#mat_syn_train is actually sorted_mat_syn_train
#TODO: handle GUMBEL parameters
def compute_scores(mat_syn_train, mat_syn_test, intercept, alpha, groundtruth_flag):

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
    
        px_train = cdf_extrapolate(dist_train, intercept, alpha) #TODO: handle different cases
        px_test = cdf_extrapolate(dist_test, intercept, alpha)
    
        z_train_i=pi_i_log_sorted(px_train, rank_s_tr, N) #TODO: use correct func
        z_test_i=pi_i_log_sorted(px_test, rank_s_te, N)
   
        delta_pi_i_new=np.log10(z_train_i)-np.log10(z_test_i) #TODO: add all scores

        store_in_mat[idx_syn,0] = delta_pi_i_new
        store_in_mat[idx_syn,1] = z_train_i
        store_in_mat[idx_syn,2] = z_test_i
        store_in_mat[idx_syn,3] = px_train
        store_in_mat[idx_syn,4] = px_test
        store_in_mat[idx_syn,5] = rank_s_tr
        store_in_mat[idx_syn,6] = rank_s_te
        store_in_mat[idx_syn,7] = dist_train
        store_in_mat[idx_syn,8] = dist_test
        if groundtruth_flag:
            store_in_mat[idx_syn,9] = groundtruth_i

    return store_in_mat

def PRIVET(train, test, synthetic, intercept, alpha, renormalization = None, distance='standard_euclidean', device=None, groundtruth = None):

    #TODO: stick with np arrays not torch tensors
    
    ############################
    ## COMPUTE 1-NN distances ##
    ############################
    
    ## SYNTHETIC TO TRAIN
    #should be pytorch tensor here
    dist_NN_syn_tr = gpu_nearest_neighbors(synthetic, train, k=1, distance=distance,chunk_size=128,device=device,verbose=False)
    p_syn_tr_NN_dist, p_syn_tr_NN_idx = sorting(dist_NN_syn_tr)

    ## SYNTHETIC TO TEST
    dist_NN_syn_te = gpu_nearest_neighbors(synthetic, test, k=1,distance=distance,chunk_size=128,device=device,verbose=False)
    p_syn_te_NN_dist, p_syn_te_NN_idx = sorting(dist_NN_syn_te)

    tmp = dist_NN_syn_te[0] # for renormalization when train and test have different sample sizes.

    if renormalization != None: #TODO: don't split into multiply or divide case, if you manage right from beginning then it's always multiply
        tmp = tmp / renormalization 
        p_syn_te_NN_dist = p_syn_te_NN_dist / renormalization #for plotting cdf
        if test.shape[0] > train.shape[0]:
            tmp = tmp * renormalization  
            p_syn_te_NN_dist = p_syn_te_NN_dist * renormalization #for plotting cdf


    #table containing:  d_STr^*, i_s, i_r^Tr (sorted on i_s)
    mat_syn_train = torch.concatenate([dist_NN_syn_tr[0], torch.arange(synthetic.shape[0]).reshape(-1,1), dist_NN_syn_tr[1]],axis=1)
    #table containing:  d_STe^*, i_s, i_r^Te (sorted on i_s)
    mat_syn_test = torch.concatenate([tmp, torch.arange(synthetic.shape[0]).reshape(-1,1), dist_NN_syn_te[1]],axis=1)

    groundtruth_flag=False
    if groundtruth != None: #for controlled experiments (should be improvable, that's a bit ugly)
        #table containing:  d_STr^*, i_s, i_r^Tr (sorted on i_s)
        mat_syn_train = torch.concatenate([dist_NN_syn_tr[0], torch.arange(synthetic.shape[0]).reshape(-1,1), dist_NN_syn_tr[1], groundtruth.reshape(-1,1)],axis=1)
        #table containing:  d_STe^*, i_s, i_r^Te (sorted on i_s)
        mat_syn_test = torch.concatenate([tmp, torch.arange(synthetic.shape[0]).reshape(-1,1), dist_NN_syn_te[1], groundtruth.reshape(-1,1)],axis=1)
        groundtruth_flag=True

    #table containing:  d_STr^*, i_s, i_r^Tr (sorted on d_STr^*)
    sorted_mat_syn_train=mat_syn_train[mat_syn_train[:, 0].argsort()]
    #table containing:  d_STe^*, i_s, i_r^Te (sorted on d_STe^*)
    sorted_mat_syn_test = mat_syn_test[mat_syn_test[:, 0].argsort()]
    
    store_in_mat = compute_scores(sorted_mat_syn_train, sorted_mat_syn_test, intercept, alpha, groundtruth_flag=groundtruth_flag)
    
    return store_in_mat, p_syn_tr_NN_dist, p_syn_te_NN_dist