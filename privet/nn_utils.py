import torch
from tqdm.autonotebook import tqdm


# TODO: use FAISS CPU/GPU instead!
def gpu_nearest_neighbors(
    X, Y=None, k=1, distance="hamming", chunk_size=128, device="cuda", verbose=False
):
    """
    Compute the k nearest neighbors for each sample in X. If Y is provided, for each sample in X
    find the k nearest neighbors among Y. When Y is None, find nearest neighbors within X (self-comparison
    with self-distance excluded).

    This function expect samples to be flattened to 1D vectors, ie X is a tensor of shape (N_samples, N_features)

    distances taken into account are: "hamming", "standard_euclidean" and "feature_normalized_euclidean"
    feature_normalized_euclidean is euclidean distance further divided by sqrt(N_features)
    (feature_normalized_euclidean is used in 4.1 https://arxiv.org/abs/2301.13188)

    The chunking prevents memory overflows when computing NN for very large datasets, it avoids storing a N_X * N_Y matrix

    Returns:
        A tuple (dists, indices):
         - dists: Array of shape (n_samples_X, k) with nearest neighbor distances.
         - indices: Array of shape (n_samples_X, k) with indices of the neighbors (relative to Y, or X if Y is None).
    """
    # Device
    device = torch.device(device)
    X = X.to(device)
    same = Y is None
    Y = X if same else Y.to(device)

    nX, dim = X.shape
    nY = Y.shape[0]

    # For Euclidean distances
    if distance in ("standard_euclidean", "feature_normalized_euclidean"):
        x_sq = (X.float() ** 2).sum(dim=1)
        y_sq = x_sq if same else (Y.float() ** 2).sum(dim=1)
        normalize = distance == "feature_normalized_euclidean"
        factor = dim if normalize else 1

    # Output buffers
    best_d = torch.full((nX, k), float("inf"), device=device)
    best_i = torch.full((nX, k), -1, device=device, dtype=torch.long)

    outer = range(0, nX, chunk_size)
    if verbose:
        outer = tqdm(outer, desc="Rows")

    for i in outer:
        end_i = min(i + chunk_size, nX)
        Xc = X[i:end_i]
        if distance in ("standard_euclidean", "feature_normalized_euclidean"):
            xn = x_sq[i:end_i]

        chunk_d = torch.full((end_i - i, k), float("inf"), device=device)
        chunk_i = torch.full((end_i - i, k), -1, device=device, dtype=torch.long)

        inner = range(0, nY, chunk_size)
        if verbose and not same:
            inner = tqdm(inner, desc="Cols", leave=False)

        for j in inner:
            end_j = min(j + chunk_size, nY)
            Yc = Y[j:end_j]

            if distance == "hamming":
                # exact integer mismatch counts
                d = (
                    (Xc.unsqueeze(1) != Yc.unsqueeze(0))
                    .to(torch.int32)
                    .sum(dim=2)
                    .to(torch.float32)
                )
            else:
                yn = y_sq[j:end_j]
                xy = torch.mm(Xc.float(), Yc.float().t())
                sq = (xn[:, None] + yn[None, :] - 2 * xy) / factor
                d = torch.sqrt(torch.clamp(sq, min=0.0))

            if same:
                # mask self-distances
                rows = torch.arange(i, end_i, device=device)
                cols = torch.arange(j, end_j, device=device)
                mask = rows.unsqueeze(1) == cols.unsqueeze(0)
                d.masked_fill_(mask, float("inf"))

            # merge into top-k
            concat_d = torch.cat([chunk_d, d], dim=1)
            idx_block = torch.arange(j, end_j, device=device).expand(end_i - i, -1)
            concat_i = torch.cat([chunk_i, idx_block], dim=1)

            chunk_d, pos = torch.topk(concat_d, k, largest=False, sorted=True)
            chunk_i = torch.gather(concat_i, 1, pos)

        best_d[i:end_i] = chunk_d
        best_i[i:end_i] = chunk_i

    return best_d.cpu(), best_i.cpu()
