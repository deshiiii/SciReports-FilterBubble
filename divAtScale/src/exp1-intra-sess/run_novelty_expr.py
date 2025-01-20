"""
RELEASE

* scale: intra-session
* div type: novelty
* extra info: Redundancy computations will yield slightly different results since log_binning for percentile
              normalisation computed on multi-affordance users not the whole affordance centric dataset.


"""

import numpy as np
from divAtScale.src.helpers.dataset_helpers.read_x_process import load_fs1, read_config, grab_sess, \
    sess_meta_data, sub_sample_lh_df
from sklearn.preprocessing import KBinsDiscretizer
from divAtScale.src.helpers.set_helpers.diversity_measures import comp_R, compute_percentile_score
from itertools import chain
import pandas as pd
from tqdm import tqdm


def grab_fandom_and_multiA_lh(S):
    """
    From listening hist df divide sessions into fandom & multi-artist versions

    Args:
        S (pandas.DataFrame): a listening history df

    Returns:
        list : fandom sessions
        list : multi-artist sessions
    """
    S_meta = sess_meta_data(S)
    fandom_df_S = S_meta[S_meta.n_a == 1]
    non_fandom_df_S = S_meta[S_meta.n_a > 1]
    fandom_df_S.set_index(['anon_user_id', 'session_n'], inplace=True)
    non_fandom_df_S.set_index(['anon_user_id', 'session_n'], inplace=True)
    S.set_index(['anon_user_id', 'session_n'], inplace=True)
    fandom_df = S.loc[fandom_df_S.index].reset_index()
    non_fandom_df = S.loc[non_fandom_df_S.index].reset_index()
    return grab_sess(fandom_df), grab_sess(non_fandom_df)


def compute_users_p_fandom_and_len_fandom(fandom_sess, multiA_sess):
    """
    Devise some meta data about fandom sessions

    Args:
        fandom_sess (list): sessions with only 1 artist
        multiA_sess (list): sessions with multiple artists

    Return:
        float : p(fandom)
        float : avg length of fandom sessions
    """

    n_sess_in_aff = len(fandom_sess) + len(multiA_sess)
    p_fandom = len(fandom_sess) / n_sess_in_aff
    if len(fandom_sess) > 0:
        l = np.mean([len(s) for s in fandom_sess])
    else:
        l = -1 # invalid option
    return p_fandom, l


def compute_p_fandom_and_len_fandom_per_u(fandom_vs_multiA):
    """
    Top level function to call fandom meta function per. user

    Args:
        fandom_vs_multiA (list): 2D per user (fandom, multiA)

    Returns:
        list : p(fandom) per. user
        list : len fandom per. user
    """
    P_sess_fandom = []
    avg_len_fandom = []
    for fandom_sess, multiA_sess in fandom_vs_multiA:
        p_fandom, fandom_l = compute_users_p_fandom_and_len_fandom(fandom_sess, multiA_sess)
        P_sess_fandom.append(p_fandom)
        if fandom_l != -1:
            avg_len_fandom.append(fandom_l)
    return P_sess_fandom, avg_len_fandom

def print_fandom_stats(aff, p_sess_fandom, avg_len_fandom):
    """
    Print & stats about fandom sessions

    Args:
        aff (str): affordance-type
        p_sess_fandom (float) : to print...
        avg_len_fandom (float) : to print...
    """
    print("* avg p fandom {}: {}, {}".format(aff, np.mean(p_sess_fandom), np.std(p_sess_fandom)))
    print("** avg len fandom sess {}: {}, {}".format(aff, np.mean(avg_len_fandom), np.std(avg_len_fandom)))


def get_fs2_users_fandom_vs_multiA_sess(lh_df):
    """
    Split users listening histories into fandom vs mutiA sets

    Args:
        lh_df (pandas.DataFrame) : users' listening history df

    Returns:
        list : fandom_vs_multiA_P
        list : fandom_vs_multiA_Q
        list : fandom_vs_multiA_A
        list : fandom_vs_multiA_E
    """
    fandom_vs_multiA_P = grab_fandom_and_multiA_lh(lh_df[lh_df.affordance == "o_pl"])
    fandom_vs_multiA_Q = grab_fandom_and_multiA_lh(lh_df[lh_df.affordance == "o_s"])
    fandom_vs_multiA_A = grab_fandom_and_multiA_lh(lh_df[lh_df.affordance == "a"])
    fandom_vs_multiA_E = grab_fandom_and_multiA_lh(lh_df[lh_df.affordance == "e"])
    return fandom_vs_multiA_P, fandom_vs_multiA_Q, fandom_vs_multiA_A, fandom_vs_multiA_E


def get_fandom_fs2(lh_df, uid):
    """
    Query per. affordance fandom and length data for a given user.

    Args:
        lh_df (pandas.DataFrame): listening history df
        uid (str) : hashed user id

    Returns:
        list : p(fandom) whereby elements correspond to P,Q,A,E affordances
        list : avg len fandom whereby elements correspond to P,Q,A,E affordances
    """
    lh_df = lh_df[lh_df.anon_user_id == uid]
    fandom_vs_multiA_P, fandom_vs_multiA_Q, fandom_vs_multiA_A, fandom_vs_multiA_E = get_fs2_users_fandom_vs_multiA_sess(lh_df)

    P_sess_fandom_P, fandom_avg_len_P = compute_users_p_fandom_and_len_fandom(fandom_vs_multiA_P[0], fandom_vs_multiA_P[1])
    P_sess_fandom_Q, fandom_avg_len_Q = compute_users_p_fandom_and_len_fandom(fandom_vs_multiA_Q[0], fandom_vs_multiA_Q[1])
    P_sess_fandom_A, fandom_avg_len_A = compute_users_p_fandom_and_len_fandom(fandom_vs_multiA_A[0], fandom_vs_multiA_A[1])
    P_sess_fandom_E, fandom_avg_len_E = compute_users_p_fandom_and_len_fandom(fandom_vs_multiA_E[0], fandom_vs_multiA_E[1])

    p_fandom_vec = [P_sess_fandom_P, P_sess_fandom_Q, P_sess_fandom_A, P_sess_fandom_E]
    l_fandom_vec = [fandom_avg_len_P, fandom_avg_len_Q, fandom_avg_len_A, fandom_avg_len_E]
    return p_fandom_vec, l_fandom_vec



def compute_adjusted_R(S, binned_data, aff_labels, n_sess_per_user_P, n_sess_per_user_Q,
                       n_sess_per_user_A, n_sess_per_user_E, n_bins=5):
    """
    Compute Redundancy scores adjusted using log-binning and percentile ranking

    Args:
        S (int) : list of all sessions
        binned_data (list): bin assignment for each session.
        aff_labels (list): indicates which affordance was used for each session
        n_sess_per_user_P (list): used to chop S per user
        n_sess_per_user_Q (list): used to chop S per user
        n_sess_per_user_A (list): used to chop S per user
        n_sess_per_user_E (list): used to chop S per user
        n_bins (int): number bins for log binning

    Returns:
        list : $R_P$ over all users
        list : $R_Q$ over all users
        list : $R_A$ over all users
        list : $R_E$ over all users
    """
    N = len(S)
    percentile_rank_scores = np.zeros(N)

    for bin_l in range(n_bins):
        sess_binned = [S[i] for i in np.argwhere(binned_data == bin_l).flatten()]
        R_sess = np.array([comp_R(s) for s in sess_binned])
        percentile_rank_bin = compute_percentile_score(R_sess)
        percentile_rank_scores[binned_data == bin_l] = percentile_rank_bin

    # split back into P, Q, A, E
    R_P = percentile_rank_scores[aff_labels == "P"]
    R_Q = percentile_rank_scores[aff_labels == "Q"]
    R_A = percentile_rank_scores[aff_labels == "A"]
    R_E = percentile_rank_scores[aff_labels == "E"]

    # compute averages per user...
    R_P = [np.mean(R_P[n_sess_per_user_P[i] : n_sess_per_user_P[i+1]]) for i in range(len(n_sess_per_user_P)-1)]
    R_Q = [np.mean(R_Q[n_sess_per_user_Q[i] : n_sess_per_user_Q[i+1]]) for i in range(len(n_sess_per_user_Q)-1)]
    R_A = [np.mean(R_A[n_sess_per_user_A[i] : n_sess_per_user_A[i+1]]) for i in range(len(n_sess_per_user_A)-1)]
    R_E = [np.mean(R_E[n_sess_per_user_E[i] : n_sess_per_user_E[i+1]]) for i in range(len(n_sess_per_user_E)-1)]

    return R_P, R_Q, R_A, R_E




if __name__ == "__main__":
    print("starting monad experiments")

    aff = ['P', 'Q', 'A', 'E']
    config = read_config('./expr_config.json')
    YEAR = config['year']
    N_SAMPLE = config['n_sample']
    SEED = config['r_seed']
    data_dir = "./data/" + str(YEAR) + "_release"
    fig_dir = data_dir + "/Figs"
    res_dir = data_dir + "/res"

    print("\n*** Experiment 1.2.: intra-sess novelty")
    fs2 = pd.read_csv(data_dir + "/FS2/PQAE", index_col=0)
    fs2_uids = fs2.anon_user_id.unique()

    # (1) fandom analysis
    pfandom_x_len = [get_fandom_fs2(fs2, uid) for uid in tqdm(fs2_uids)]
    pfandom = [v[0] for v in pfandom_x_len] # extract fandom
    pfandom_avg = np.mean(pfandom, axis=0)

    fandom_len = [v[1] for v in pfandom_x_len]
    avg_len_P = np.mean([v[0] for v in fandom_len if v != -1])
    avg_len_Q = np.mean([v[1] for v in fandom_len if v != -1])
    avg_len_A = np.mean([v[2] for v in fandom_len if v != -1])
    avg_len_E = np.mean([v[3] for v in fandom_len if v != -1])

    # (2) R analysis
    print("** computing R for FS2 user set")
    multiA_P_all = []
    multiA_Q_all = []
    multiA_A_all = []
    multiA_E_all = []

    uids_keep = []
    for uid in tqdm(fs2_uids):
        lh_df = fs2[fs2.anon_user_id == uid]

        fandom_vs_multiA_P, fandom_vs_multiA_Q, \
            fandom_vs_multiA_A, fandom_vs_multiA_E = get_fs2_users_fandom_vs_multiA_sess(lh_df)
        multi_A_P = fandom_vs_multiA_P[1]
        multi_A_Q = fandom_vs_multiA_Q[1]
        multi_A_A = fandom_vs_multiA_A[1]
        multi_A_E = fandom_vs_multiA_E[1]

        if (len(multi_A_P) > 0) & (len(multi_A_Q) > 0) & (len(multi_A_A) > 0) & (len(multi_A_E) > 0):
            multiA_P_all.append(multi_A_P)
            multiA_Q_all.append(multi_A_Q)
            multiA_A_all.append(multi_A_A)
            multiA_E_all.append(multi_A_E)
            uids_keep.append(uid)


    n_sess_per_user_P = np.cumsum([0] + [len(S) for S in multiA_P_all])
    multiA_P_all = list(chain(*multiA_P_all))

    n_sess_per_user_Q = np.cumsum([0] + [len(S) for S in multiA_Q_all])
    multiA_Q_all = list(chain(*multiA_Q_all))

    n_sess_per_user_A = np.cumsum([0] + [len(S) for S in multiA_A_all])
    multiA_A_all = list(chain(*multiA_A_all))

    n_sess_per_user_E = np.cumsum([0] + [len(S) for S in multiA_E_all])
    multiA_E_all = list(chain(*multiA_E_all))

    aff_labels = np.array(['P'] * len(multiA_P_all) + ['Q'] * len(multiA_Q_all) +
                          ['A'] * len(multiA_A_all) + ['E'] * len(multiA_E_all))

    S = multiA_P_all + multiA_Q_all + multiA_A_all + multiA_E_all
    N = len(S)

    S_len = np.array([len(x) for x in S])
    S_len_log = np.log10(S_len)
    X_len = np.array([[x] for x in S_len_log])


    # note: will give slightly different results to paper since computed on multi-affordance users' sessions alone.
    #       principal remains the same.
    kbins = KBinsDiscretizer(n_bins=5, strategy='uniform', encode='ordinal')
    kbins.fit(X_len)
    binned_data = kbins.transform(X_len).flatten()

    R_P, R_Q, R_A, R_E = compute_adjusted_R(S, binned_data, aff_labels, n_sess_per_user_P, n_sess_per_user_Q,
                                            n_sess_per_user_A, n_sess_per_user_E, n_bins=5)

    # znorm per user...
    R_all = [np.array([x1,x2,x3,x4]) for x1,x2,x3,x4 in zip(R_P, R_Q, R_A, R_E)]
    R_all = [(x-np.mean(x)) / np.std(x) if np.std(x) != 0 else x-np.mean(x) for x in R_all]
    R_avg = np.mean(R_all, axis=0)
    R_std = np.std(R_all, axis=0)

    pd.DataFrame({"uids":uids_keep,"R_P":R_P, "R_Q":R_Q, "R_A":R_A, "R_E":R_E}).to_csv("./R_for_all_u")

    ### save all experiments :
    fandom_df = pd.DataFrame({"affordance":["P","Q","A","E"],
                              "p_fandom": [pfandom_avg[0],  pfandom_avg[1],  pfandom_avg[2],  pfandom_avg[3]],
                              "avg_len_of_fandom": [avg_len_P, avg_len_Q, avg_len_A, avg_len_E],
                              "R_avg_without_no_rep_znormed:": [R_avg[0], R_avg[1], R_avg[2], R_avg[3]],
                              "R_std_without_no_rep_znormed:": [R_std[0], R_std[1], R_std[2], R_std[3]]})
    print(fandom_df)
    fandom_df.to_csv(res_dir + "/intra_set_fs2")
