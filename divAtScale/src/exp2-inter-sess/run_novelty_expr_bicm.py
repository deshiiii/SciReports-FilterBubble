"""
RELEASE

* scale: inter-session
* div type: spacial disparity
* extra info: all users have >= 3 sessions per. affordance even after filtering non-pop artists.


"""

from divAtScale.src.bipartite_config.src.custom_functions import compute_gamma_per_aff, evaluate
from divAtScale.src.helpers.dataset_helpers.read_x_process import load_fs1, load_fs2_inter_condition, \
    filter_u_with_less_than_thresh_sess, sub_sample_lh_df, read_config
import pandas as pd
from tqdm import tqdm
import sys
import numpy as np


if __name__ == '__main__':
    fs = int(sys.argv[1]) # filtering-stratergy (1|2)
    config = read_config('./expr_config.json')
    YEAR = config['year']
    N_SAMPLE = config['n_sample']
    SEED = config['r_seed']
    data_dir = "./data/" + str(YEAR) + "_release"
    fig_dir = data_dir + "/Figs"
    res_dir = data_dir + "/res"

    print("\n*** Experiment 2.2.: inter-sess novelty")
    print("* Computing set-based overlap")

    if fs == 2:
        PQAE = load_fs2_inter_condition(data_dir, thresh=3)
        if N_SAMPLE != -1:
            PQAE = sub_sample_lh_df(PQAE, N_SAMPLE, SEED)

        uids = PQAE.anon_user_id.unique()

        print("** constructing bicm per u...")
        O = []  # <- all overlap scores
        failed_user = 0
        for uid in tqdm(uids):
            gamma_Q = -1
            gamma_P = -1
            gamma_A = -1
            gamma_E = -1

            u_df = PQAE[PQAE.anon_user_id == uid]
            # construct session-artist bipartite net.
            failed_status = 0
            while failed_status != -1:
                gamma_A = compute_gamma_per_aff('a', u_df)
                failed_status = gamma_A

                gamma_E = compute_gamma_per_aff('e', u_df)
                failed_status = gamma_E

                gamma_Q = compute_gamma_per_aff('o_s', u_df)
                failed_status = gamma_Q

                gamma_P = compute_gamma_per_aff('o_pl', u_df)
                failed_status = gamma_P

                failed_status = -1

            gamma_all = np.array([gamma_Q, gamma_P, gamma_A, gamma_E])

            if -1 in gamma_all:
                failed_user+=1
            else:
                if np.std(gamma_all) == 0:
                    o = np.array([0,0,0,0])
                    O.append(o)
                else:
                    # z-norm
                    gamma_all = (gamma_all - np.mean(gamma_all)) / np.std(gamma_all)
                    O.append(gamma_all)

        p_failed = failed_user / len(uids)
        print("* p failed:", p_failed)

        ## evaluate
        O_Q, O_P, O_A, O_E = map(list, zip(*[[element[0], element[1], element[2], element[3]] for element in O]))
        o_q_avg, o_q_std, q_p_failed = evaluate(O_Q, 'Q')
        o_p_avg, o_p_std, p_p_failed = evaluate(O_P, 'P')
        o_a_avg, o_a_std, a_p_failed = evaluate(O_A, 'A')
        o_e_avg, o_e_std, e_p_failed = evaluate(O_E, 'E')

    # finally save!
    df = pd.DataFrame({'aff': ["Q", "P", "A", "E"],
                  "gamma_avg": [o_q_avg, o_p_avg, o_a_avg, o_e_avg],
                  "gamma_std": [o_q_std, o_p_std, o_a_std, o_e_std],
                  "p_failed":[p_failed, p_failed, p_failed, p_failed]})
    df.to_csv(res_dir + "/inter_set_fs" + str(fs))
    print('\n')
    print(df)