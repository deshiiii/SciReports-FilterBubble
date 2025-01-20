"""
RELEASE

* scale: inter-affordance
* div type: novelty
* extra info: again, znormed per user.

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from divAtScale.src.bipartite_config.src.custom_functions import evaluate, compute_gamma_inter_aff
from divAtScale.src.helpers.dataset_helpers.read_x_process import load_fs2_inter_condition, read_config, sub_sample_lh_df


if __name__ == '__main__':

    config = read_config('./expr_config.json')
    YEAR = config['year']
    N_SAMPLE = config['n_sample']
    SEED = config['r_seed']
    data_dir = "./data/" + str(YEAR) + "_release"
    fig_dir = data_dir + "/Figs"
    res_dir = data_dir + "/res"

    print("\n*** Experiment 3.2.: inter-aff novelty")

    PQAE = load_fs2_inter_condition(data_dir)
    if N_SAMPLE != -1:
        PQAE = sub_sample_lh_df(PQAE, N_SAMPLE, SEED)
    uids = PQAE.anon_user_id.unique()

    O = []  # <- all overlap scores
    n_failed = 0
    i = 0
    for uid in tqdm(uids):
        i+=1
        # construct session-artist bipartite net.
        u_df = PQAE[PQAE.anon_user_id == uid]
        gamma_PQ = compute_gamma_inter_aff('o_s', 'o_pl', u_df)
        gamma_PA = compute_gamma_inter_aff('o_pl', 'a', u_df)
        gamma_PE = compute_gamma_inter_aff('o_pl', 'e', u_df)
        gamma_QA = compute_gamma_inter_aff('o_s', 'a', u_df)
        gamma_QE = compute_gamma_inter_aff('o_s', 'e', u_df)
        gamma_AE = compute_gamma_inter_aff('a', 'e', u_df)

        if (gamma_PQ != -1) & (gamma_PA != -1) & (gamma_PE != -1) & (gamma_QA != -1) & (gamma_QE != -1) & (gamma_AE != -1):
            gamma_all = np.array([gamma_PQ, gamma_PA, gamma_PE, gamma_QA, gamma_QE, gamma_AE])
            gamma_std = np.std(gamma_all)
            gamma_avg = np.mean(gamma_all)

            # z-norm
            if gamma_std != 0:
                gamma_all = (gamma_all - gamma_avg) / gamma_std
            else:
                gamma_all = gamma_all - gamma_avg
            O.append(gamma_all)
        else:
            n_failed +=1

    p_failed = n_failed / len(uids)
    print("* p failed:", p_failed)

    ## evaluate
    O_PQ, O_PA, O_PE, O_QA, O_QE, O_AE = map(list, zip(*[[element[0], element[1], element[2],
                                                          element[3], element[4], element[5]] for element in O]))

    o_pq_avg, o_pq_std, pq_p_failed = evaluate(O_PQ, "P-Q")
    o_pa_avg, o_pa_std, pa_p_failed = evaluate(O_PA, "P-A")
    o_pe_avg, o_pe_std, pe_p_failed = evaluate(O_PE, "P-E")
    o_qa_avg, o_qa_std, aq_p_failed = evaluate(O_QA, "Q-A")
    o_qe_avg, o_qe_std, qe_p_failed = evaluate(O_QE, "Q-E")
    o_ae_avg, o_ae_std, ae_p_failed = evaluate(O_AE, "A-E")

    df = pd.DataFrame({'aff': ["P-Q", "P-A", "P-E", "Q-A", "Q-E", "A-E"],
                  "gamma_avg": [o_pq_avg, o_pa_avg, o_pe_avg, o_qa_avg, o_qe_avg, o_ae_avg],
                  "gamma_std": [o_pq_std, o_pa_std, o_pe_std, o_qa_std, o_qe_std, o_ae_std],
                  "p_failed":[p_failed, p_failed, p_failed, p_failed, p_failed, p_failed]})
    df.to_csv(res_dir + "/inter_aff_set")
    print(df)
