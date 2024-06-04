import logging

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


def greedy_optimize(real_feats, gen_feats, dist, init_idxs=None) -> np.ndarray:
    """
    Args:
        real_feats (np.ndarray): An n x k array of real features.
        gen_feats (np.ndarray): An m x d x k array of generated features.
        dist (callable): A distance function that takes two arrays and returns a distance value.

    Returns:
        np.ndarray: Indices of selected generated features for each real feature.
    """
    A, B, _ = gen_feats.shape
    if init_idxs is None:
        select_idxs = np.zeros(shape=(A,), dtype=int)
    else:
        select_idxs = np.asarray(init_idxs, dtype=int)
        assert select_idxs.shape == (A,)

    pbar = tqdm(range(A))
    for i in pbar:
        min_dist = np.inf
        best_j = 0
        for j in range(B):
            select_idxs[i] = j
            curr_dist = dist(real_feats, gen_feats[np.arange(select_idxs.size), select_idxs, :])
            if curr_dist < min_dist:
                min_dist = curr_dist
                best_j = j
        select_idxs[i] = best_j
        pbar.set_postfix({"FID": min_dist})
    return select_idxs, float(min_dist)


def random_search_optimize(real_feats: np.ndarray, gen_feats: np.ndarray, dist: callable, size: int = 1000) -> np.ndarray:
    """
    Args:
        real_feats (np.ndarray): An n x k array of real features.
        gen_feats (np.ndarray): An m x d x k array of generated features.
        dist (callable): A distance function that takes two arrays and returns a distance value.
        size (int): Number of random samples to try. Default is 1000.

    Returns:
        np.ndarray: Indices of selected generated features for each real feature.
    """
    A, B, _ = gen_feats.shape
    
    best_idxs = None
    min_dist = np.inf

    for _ in tqdm(range(size)):
        curr_idxs = np.random.randint(0, B, A)
        # Adjust the indexing to get the correct features
        selected_feats = gen_feats[np.arange(A), curr_idxs, :]
        curr_dist = dist(real_feats, selected_feats)
        if curr_dist < min_dist:
            min_dist = curr_dist
            best_idxs = curr_idxs
        tqdm.write(f'Best FID {min_dist:.4f}')
    return best_idxs