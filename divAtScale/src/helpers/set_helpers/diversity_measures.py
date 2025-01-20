import numpy as np
from tqdm import tqdm

def compute_percentile_score(R):
    """
    Strict percentile score

    Args:
        R (np.array): array of redundancy scores

    Returns:
        np.array : percentile score for each session.
    """
    return np.array([len(R[R < x]) / len(R) for x in tqdm(R)])


def comp_R(s):
    """
    Compute Redundancy (R = 1 - A/P)

    Args:
        s (list): a session of artist ids

    Returns:
        float : R score
    """
    return 1 - (len(set(s)) / len(s))
