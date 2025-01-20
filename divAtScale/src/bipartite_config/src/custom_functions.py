import scipy.sparse
from divAtScale.src.bipartite_config.src.bicm import BiCM
import numpy as np


def threshold_array(arr, threshold=0.05):
    binary_array = (arr <= threshold).astype(int)
    return binary_array

def comp_gamma(A, matrix_cutoff=0):
    """
    * overlap score
    :return:
    """
    #return np.mean([1 if len(a.nonzero()[0]) > 0 else 0 for a in A])

    if matrix_cutoff != 0:
        A = A[:matrix_cutoff, matrix_cutoff:]
        gamma1 = np.mean((np.sum(A, axis=1) >= 1).astype(int))
        gamma2 = np.mean((np.sum(A, axis=0) >= 1).astype(int))
        return np.mean([gamma1, gamma2])
    else:
        return np.mean((np.sum(A, axis=1) >= 1).astype(int))


def run_bicm(mat, alpha=0.05, k1=0, mat_cut=0):
    """

    :param mat:
    :param alpha:
    :return:
    """

    n_sessions = mat.shape[0]

    try:
        cm = BiCM(bin_mat=mat)
        cm.make_bicm()
        solver_status = cm.sol.success
        #print('solver:', solver_status)

        # try again with least-squares solver
        if solver_status == False:
            cm = BiCM(bin_mat=mat)
            cm.make_bicm(method="lm")
            solver_status = cm.sol.success
            #print('solver:', solver_status)

        # assert that the solver has not crashed...
        if solver_status:
            A = cm.adj_matrix

            k2 = int(cm.get_triup_dim(True) - 1)
            nlam = cm.get_lambda_motif_block(mat, k1, k2)

            edge_tuple_pvals = cm.get_plambda_block(A, k1, k2)
            pv = cm.get_pvalues_q(edge_tuple_pvals, nlam, k1, k2 + 1, parallel=True)

            # constructed mono-partite projection for sessions
            pv = threshold_array(pv, threshold=alpha)
            mat_i = np.array([cm.flat2triumat_idx(i, n_sessions) for i in range(len(pv))])
            r = mat_i[:, 0] # rows
            c = mat_i[:, 1] # cols

            M_pval = scipy.sparse.csr_array((pv, (r, c)), shape=(n_sessions, n_sessions)).todense()
            M_pval = M_pval + M_pval.T - np.diag(np.diag(M_pval))  # copy diag onto the bottom...

            overlap = comp_gamma(M_pval, matrix_cutoff=mat_cut)
        else:
            #print("now setting overlap to -1")
            overlap = -1
        return overlap

    except Exception as e:
        return -1


def compute_gamma_per_aff(aff, u_df, debug=False):
    """
    ** Get gamma overlap score for user sessions

    :param aff:
    :param u_df:
    :param debug:
    :return:
    """
    sess_os = [list(set(s)) for s in
               list(u_df[u_df.affordance == aff].groupby('session_n').artist_id.apply(list))]
    all_artists = u_df.artist_id.unique()
    aid2matid = {aid: i for i, aid in enumerate(all_artists)}

    mat = np.zeros(shape=(len(sess_os), len(all_artists)))
    for i, s in enumerate(sess_os):
        update_cols = [aid2matid[a] for a in s]
        mat[i, update_cols] = 1

    gamma = run_bicm(mat)  # overlap
    return gamma


def evaluate(O, aff):
    """

    :param O:
    :param aff:
    :return:
    """
    #print('--------------------------')
    #print('* affordance:', aff)
    O = np.array(O)
    p_failed = len(O[O == -1]) / len(O)
    #print('* p solver failed:', p_failed)

    O = O[O > -1]
    O_avg = np.mean(O)
    O_std = np.std(O)
    #print('** final gamma: {},{}'.format(O_avg, O_std))
    return O_avg, O_std, p_failed


def compute_gamma_inter_aff(aff1, aff2, u_df):
    """
    ** Get gamma overlap score for user sessions

    :param aff1:
    :param af2:
    :param u_df:
    :param debug:
    :return:
    """

    sess_aff1 = [list(set(s)) for s in
               list(u_df[(u_df.affordance == aff1)].groupby('session_n').artist_id.apply(list))]

    sess_aff2 = [list(set(s)) for s in
               list(u_df[(u_df.affordance == aff2)].groupby('session_n').artist_id.apply(list))]

    n_sessions = len(sess_aff1) + len(sess_aff2)
    all_artists = u_df.artist_id.unique()
    n_artists = len(all_artists)
    aid2matid = {aid: i for i, aid in enumerate(all_artists)}


    mat = np.zeros(shape=(n_sessions,n_artists))

    for i, s in enumerate(sess_aff1):
        update_cols = [aid2matid[a] for a in s]
        mat[i, update_cols] = 1

    j = len(sess_aff1)
    for s in sess_aff2:
        update_cols = [aid2matid[a] for a in s]
        mat[j, update_cols] = 1
        j+=1

    gamma = run_bicm(mat, mat_cut=len(sess_aff1))  # overlap
    return gamma
