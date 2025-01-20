"""
RELEASE

* scale: inter-affordance
* div type: spatial diversity
* extra info: again, znormed per user.

"""

from divAtScale.src.helpers.dataset_helpers.read_x_process import read_config, load_fs2_inter_condition, \
    sub_sample_lh_df
from divAtScale.src.helpers.semantic_helpers.diversity_measures import get_sim_between_aff
import json
import numpy as np
import pandas as pd


if __name__ == '__main__':

    config = read_config('./expr_config.json')
    YEAR = config['year']
    N_SAMPLE = config['n_sample']
    SEED = config['r_seed']
    data_dir = "./data/" + str(YEAR) + "_release"
    fig_dir = data_dir + "/Figs"
    res_dir = data_dir + "/res"
    pop_a = np.load(data_dir + "/pop_artists.npy")
    e = np.load(data_dir + "/e_balanced.npy")
    with open(data_dir + "/mid2aid_balanced", 'r') as json_file:
        mid2aid = json.load(json_file)  # matrix id --> artist_id
    aid2mid = {value: key for key, value in mid2aid.items()}

    print("\n*** Experiment 3.1.: inter-aff spatial disparity")

    multi_a_sess = load_fs2_inter_condition(data_dir)
    if N_SAMPLE != -1:
        multi_a_sess = sub_sample_lh_df(multi_a_sess, N_SAMPLE, SEED)

    # in order QP, QA, QE, PA, PE, AE
    aff_sim_res, p_failed = get_sim_between_aff(multi_a_sess, e, aid2mid, subset_a=pop_a)
    avg_sim = np.mean(aff_sim_res, axis=0)
    std_sim = np.std(aff_sim_res, axis=0)
    res_df = pd.DataFrame({"affordance_pair":["QP", "QA", "QE", "PA", "PE", "AE"],
                           "avg_cos_sim":avg_sim,
                           "std_cos_sim":std_sim,
                           "p_failed":[p_failed, p_failed, p_failed, p_failed, p_failed, p_failed]})
    res_df.to_csv(res_dir + "/inter_aff_semantic")
    print(res_df)
