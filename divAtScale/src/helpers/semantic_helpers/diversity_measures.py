"""
Spatial disparity helper functions

"""


import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine as cos_dis
from tqdm import tqdm
import random


def comp_w_centroid(s, e, aid2mid):
    """
    Compute a sessions activity weighted centroid in embedding space

    Args:
        s (list) : session (artist ids)
        e (numpy.array) : embedding matrix
        aid2mid (dict) : matrix index lookup

    Returns:
        numpy.array : centroid
    """
    S_len = len(s)
    w = dict(Counter(s))
    a_unique = list(set(s))
    e_all = np.array([e[int(aid2mid[aid])] for aid in a_unique])
    e_w = [e * w[aid] for e, aid in zip(e_all, a_unique)]
    w_centroid = np.sum(e_w, axis=0) / S_len
    return w_centroid


def gs_score(s, e, aid2mid):
    """
    Compute Anderson et al.'s GS-score

    Args:
        s (list): session
        e (numpy.array): embedding matrix
        aid2mid (dict): matrix index lookup

    Returns:
        float : gs_score for a given session
    """
    S_len = len(s)
    w = dict(Counter(s))
    a_unique = list(set(s))
    w_vec = np.array([w[aid] for aid in a_unique])
    e_all = np.array([e[int(aid2mid[aid])] for aid in a_unique])
    e_w = [e * w[aid] for e, aid in zip(e_all, a_unique)]
    w_centroid = np.sum(e_w, axis=0) / S_len

    sim = cosine_similarity([w_centroid], e_all)[0]
    sim = sim * w_vec
    gs = sum(sim) / S_len
    return gs


def gs_score_inter_sess(S, e, aid2mid):
    """
    Compute Anderson et al.'s GS-score between multiple sessions

    Args:
        S (list): list of sessions (artist ids)
        e (numpy.array) : embedding matrix
        aid2mid (dict) : matrix index lookup

    Returns:
        float : gs_score over all sessions
    """

    sess_rep = [comp_w_centroid(s, e, aid2mid) for s in S]
    centroid = np.mean(sess_rep, axis=0)
    sim = cosine_similarity([centroid], sess_rep)[0]
    gs = np.mean(sim)
    return gs


def sim_inter_aff(S1, S2, e, aid2mid):
    """
    Compute cosine similarity between avg. rep of two session centroids
    generated from two distinct affordances.

    Args:
        S (list): list of sessions (artist ids)
        e (numpy.array) : embedding matrix
        aid2mid (dict): matrix index lookup

    Returns:
        float : cosine similarity between two affordances
    """

    sess_rep1 = [comp_w_centroid(s, e, aid2mid) for s in S1]
    centroid1 = np.mean(sess_rep1, axis=0)

    sess_rep2 = [comp_w_centroid(s, e, aid2mid) for s in S2]
    centroid2 = np.mean(sess_rep2, axis=0)

    sim = 1 - cos_dis(centroid1, centroid2)
    return sim


def filter_niche_sessions(sess, u_df, subset_a, pop_thresh=0.8):
    """
    Remove sessions with too much long tail.
    Do not have good structural rep. for these artists.

    Args:
        sess (list) : collection of sesssions
        u_df (pandas.DataFrame) : listening history df
        subset_a (list) : pop artists to keep

    Returns:
        list : filtered sessions
    """

    streamed_artists = u_df.artist_id.unique()
    pop_a = set(streamed_artists).intersection(set(subset_a))
    S_filt = []
    for s in sess:
        pop_a_in_s = pop_a.intersection(set(s))
        p_pop_in_session = len(pop_a_in_s) / len(set(s))
        if p_pop_in_session > pop_thresh:
            S_filt.append(list(pop_a_in_s))
    return S_filt


def comp_gs_intra_or_inter_fs2(lh_df, e, aid2mid, n_sample=-1, scale='intra', subset_a=[], sess_thresh=3):
    """
    Top level function to compute gs-scores between (inter)
    or wihtin (intra) sessions w.r.t. fs2 filtering.

    Args:
        lh_df (pandas.DataFrame): listening event df
        e (numpy.array) : embedding matrix
        aid2mid (dict): matrix index lookup for e
        n_sample (int) : if != -1 sample n_sample users at random
        scale (str) : 'intra' OR 'inter'

    Returns:
        list : all users gs scores in Q,P,A,E. Each row corresponds to a user
    """
    pdis_res = []
    n_failed = 0
    P_of_sess_pop = []
    uids_keep = []

    uids = list(lh_df.anon_user_id.unique())

    if n_sample != -1:
        uids = random.sample(uids, n_sample)

    for uid in tqdm(uids):
        spread_os = -1
        spread_op = -1
        spread_a = -1
        spread_e = -1

        u_df = lh_df[lh_df.anon_user_id == uid]

        sess_os = [list(s) for s in list(u_df[u_df.affordance == 'o_s'].groupby('session_n').artist_id.apply(list))]
        sess_op = [list(s) for s in list(u_df[u_df.affordance == 'o_pl'].groupby('session_n').artist_id.apply(list))]
        sess_a = [list(s) for s in list(u_df[u_df.affordance == 'a'].groupby('session_n').artist_id.apply(list))]
        sess_e = [list(s) for s in list(u_df[u_df.affordance == 'e'].groupby('session_n').artist_id.apply(list))]

        if len(subset_a) != 0:
            sess_os_filt = filter_niche_sessions(sess_os, u_df, subset_a)
            p_pop_os = len(sess_os_filt) / len(sess_os)
            sess_os = sess_os_filt

            sess_op_filt = filter_niche_sessions(sess_op, u_df, subset_a)
            p_pop_op = len(sess_op_filt) / len(sess_op)
            sess_op = sess_op_filt

            sess_a_filt = filter_niche_sessions(sess_a, u_df, subset_a)
            p_pop_a = len(sess_a_filt) / len(sess_a)
            sess_a = sess_a_filt

            sess_e_filt = filter_niche_sessions(sess_e, u_df, subset_a)
            p_pop_e = len(sess_e_filt) / len(sess_e)
            sess_e = sess_e_filt


        if scale == "intra":
            if len(sess_os) >= sess_thresh:
                spread_os = np.mean([gs_score(aids, e, aid2mid) for aids in sess_os])

            if len(sess_op) >= sess_thresh:
                spread_op = np.mean([gs_score(aids, e, aid2mid) for aids in sess_op])

            if len(sess_a) >= sess_thresh:
                spread_a = np.mean([gs_score(aids, e, aid2mid) for aids in sess_a])

            if len(sess_e) >= sess_thresh:
                spread_e = np.mean([gs_score(aids, e, aid2mid) for aids in sess_e])

        elif scale == "inter":
            if len(sess_os) >= sess_thresh:
                spread_os = gs_score_inter_sess(sess_os, e, aid2mid)

            if len(sess_op) >= sess_thresh:
                spread_op = gs_score_inter_sess(sess_op, e, aid2mid)

            if len(sess_a) >= sess_thresh:
                spread_a = gs_score_inter_sess(sess_a, e, aid2mid)

            if len(sess_e) >= sess_thresh:
                spread_e = gs_score_inter_sess(sess_e, e, aid2mid)

        all_spreads = [spread_os, spread_op, spread_a, spread_e]

        if (-1 in all_spreads) == False:
            pdis_res.append([spread_os, spread_op, spread_a, spread_e])
            P_of_sess_pop.append([p_pop_os, p_pop_op, p_pop_a, p_pop_e])
            uids_keep.append(uid)

        else:
            n_failed += 1

    avg_sess_left_post_niche_filt = np.mean(P_of_sess_pop, axis=0)
    pdis_res = np.array(pdis_res)
    p_failed = n_failed / len(uids)
    return pdis_res, p_failed, avg_sess_left_post_niche_filt, uids_keep


def comp_gs_intra_or_inter_fs1(monad_df_filt, e, aid2mid, scale="intra", subset_a=[], sess_thresh=3):
    """
    Top level function to compute gs-scores between (inter) or wihtin (intra) sessions w.r.t. fs1 filtering.

    Args:
        monad_df_filt (pandas.DataFrame): listening history for monadic sessions (i.e. either P|Q|A|E)
        e (numpy.array): embedding matrix
        aid2mid (dict): matrix index lookup for e

    Returns:
        list : gs for all users. rows correspond to users
    """
    uids = list(monad_df_filt.anon_user_id.unique())
    n_failed = 0

    GS_all = []
    P_of_sess_pop = []

    for uid in tqdm(uids):
        GS_avg = -1
        u_df = monad_df_filt[monad_df_filt.anon_user_id == uid]

        sess = [list(s) for s in list(u_df.groupby('session_n').artist_id.apply(list))]

        if len(subset_a) != 0:
            sess_filt = filter_niche_sessions(sess, u_df, subset_a)
            p_sess_post_filt = len(sess_filt) / len(sess)
            sess = sess_filt

        if scale == "intra":
            # asssert we stll have some sessions to work with...
            if len(sess) >= sess_thresh:
                GS_avg = np.mean([gs_score(s, e, aid2mid) for s in sess])

        elif scale == "inter":
            if len(sess) >= sess_thresh:
                GS_avg = gs_score_inter_sess(sess, e, aid2mid)

        if GS_avg != -1:
            GS_all.append(GS_avg)
            P_of_sess_pop.append(p_sess_post_filt)

        else:
            n_failed +=1

    p_failed = n_failed / len(uids)
    return GS_all, p_failed, np.mean(P_of_sess_pop)


def get_sim_between_aff(le_df, e, aid2mid, subset_a=[]):
    """
    Top level function to compute semantic (cosine) similarity between all affordances for all users.

    Args:
        le_df (pandas.DataFrame): listening  history df
        e (numpy.array): embedding matrix
        aid2mid (dict):  matrix index lookup for e

    Returns:
        list :similarity for [Q-P, Q-A, Q-E, P-A, P-E, A-E]. Rows correspond to users
    """
    psim_res = []
    uids = le_df.anon_user_id.unique()
    n_failed = 0


    for uid in tqdm(uids):
        u_df = le_df[le_df.anon_user_id == uid]
        sess_os = [list(s) for s in list(u_df[u_df.affordance == 'o_s'].groupby('session_n').artist_id.apply(list))]
        sess_op = [list(s) for s in list(u_df[u_df.affordance == 'o_pl'].groupby('session_n').artist_id.apply(list))]
        sess_a = [list(s) for s in list(u_df[u_df.affordance == 'a'].groupby('session_n').artist_id.apply(list))]
        sess_e = [list(s) for s in list(u_df[u_df.affordance == 'e'].groupby('session_n').artist_id.apply(list))]

        if len(subset_a) != 0:
            sess_os = filter_niche_sessions(sess_os, u_df, subset_a)
            sess_op = filter_niche_sessions(sess_op, u_df, subset_a)
            sess_a = filter_niche_sessions(sess_a, u_df, subset_a)
            sess_e = filter_niche_sessions(sess_e, u_df, subset_a)

        if (len(sess_op) > 0) & (len(sess_os) > 0) & (len(sess_a) > 0) & (len(sess_e) > 0):
            sim_Q_P = sim_inter_aff(sess_os, sess_op, e, aid2mid)
            sim_Q_A = sim_inter_aff(sess_os, sess_a, e, aid2mid)
            sim_Q_E = sim_inter_aff(sess_os, sess_e, e, aid2mid)
            sim_P_A = sim_inter_aff(sess_op, sess_a, e, aid2mid)
            sim_P_E = sim_inter_aff(sess_op, sess_e, e, aid2mid)
            sim_A_E = sim_inter_aff(sess_a, sess_e, e, aid2mid)

            avg_sim = np.mean([sim_Q_P, sim_Q_A, sim_Q_E, sim_P_A, sim_P_E, sim_A_E])
            std_sim = np.std([sim_Q_P, sim_Q_A, sim_Q_E, sim_P_A, sim_P_E, sim_A_E])

            sim_vec = np.array([sim_Q_P, sim_Q_A, sim_Q_E, sim_P_A, sim_P_E, sim_A_E])

            # znorm vec
            sim_vec = (sim_vec - avg_sim) / std_sim

            psim_res.append(sim_vec)
        else:
            n_failed += 1
    p_failed = n_failed / len(uids)
    return psim_res, p_failed
