"""
RELEASE

* scale: inter-session
* div type: novelty
* extra info: values returned are z-normed per users..

"""

import json
import numpy as np
import pandas as pd
from divAtScale.src.helpers.dataset_helpers.read_x_process import read_config, load_fs2_inter_condition, load_fs1, \
    sub_sample_lh_df, filter_u_with_less_than_thresh_sess
from divAtScale.src.helpers.semantic_helpers.diversity_measures import comp_gs_intra_or_inter_fs1, \
    comp_gs_intra_or_inter_fs2


if __name__ == '__main__':

    config = read_config('./expr_config.json')
    YEAR = config['year']
    N_SAMPLE = config['n_sample']
    SEED = config['r_seed']
    data_dir = "./data/" + str(YEAR) + "_release"
    fig_dir = data_dir + "/Figs"
    res_dir = data_dir + "/res"
    pop_a = np.load(data_dir + "/pop_artists.npy")     # load artists with ok structural rep.
    e = np.load(data_dir + "/e_balanced.npy")
    with open(data_dir + "/mid2aid_balanced", 'r') as json_file:
        mid2aid = json.load(json_file)  # matrix id --> artist_id
    aid2mid = {value: key for key, value in mid2aid.items()}

    print("\n*** Experiment 2.1.: inter-sess spatial disparity")

    multi_a_sess = load_fs2_inter_condition(data_dir)
    if N_SAMPLE != -1:
        multi_a_sess = sub_sample_lh_df(multi_a_sess, N_SAMPLE, SEED)

    gs_res, p_failed, avg_p_sess_left_post_niche_filt, uids_keep = comp_gs_intra_or_inter_fs2(multi_a_sess, e,
                                                                                              aid2mid, scale='inter',
                                                                                              subset_a=pop_a)

    # znorm
    gs_res = [np.array(x) for x in gs_res]
    gs_res_znormed = [(x - np.mean(x)) / np.std(x) if np.std(x) != 0 else x - np.mean(x) for x in gs_res]
    gs_avg = np.mean(gs_res_znormed, axis=0)
    gs_std = np.std(gs_res_znormed, axis=0)

    res_df = pd.DataFrame({"Aff":['P','Q','A','E'],
                           "GS_avg":[gs_avg[1], gs_avg[0], gs_avg[2], gs_avg[3]],
                           "GS_std":[gs_std[1], gs_std[0], gs_std[2], gs_std[3]],
                           "p-failed":[p_failed, p_failed, p_failed, p_failed],
                           "p_sess_left_post_niche_filt": [avg_p_sess_left_post_niche_filt[1],
                                                           avg_p_sess_left_post_niche_filt[0],
                                                           avg_p_sess_left_post_niche_filt[2],
                                                           avg_p_sess_left_post_niche_filt[3]]})
    res_df.to_csv(res_dir + "/inter_gs_fs2")
    print(res_df)
