import torch

ngpu = 1
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def AATS(A, B, p):  # p=0 (hamming) or 2 (euclidean)
    A = torch.tensor(A).to(DEVICE)
    B = torch.tensor(B).to(DEVICE)

    # TODO: replace cdist with nn_utils func
    dTT = torch.cdist(A, A, p=p)
    dTT = dTT.fill_diagonal_(torch.inf)
    dTT = dTT.min(axis=1)

    dAB = torch.cdist(B, A, p=p)
    dST = dAB.min(axis=1)
    dTS = dAB.min(axis=0)

    dSS = torch.cdist(B, B, p=p)
    dSS = dSS.fill_diagonal_(torch.inf)
    dSS = dSS.min(axis=1)

    n = dSS[0].shape[0]
    AAtruth = ((dTS[0] > dTT[0]) / n).sum().cpu().item()
    AAsyn = ((dST[0] > dSS[0]) / n).sum().cpu().item()
    AATS = (AAtruth + AAsyn) / 2

    return AAtruth, AAsyn, AATS
