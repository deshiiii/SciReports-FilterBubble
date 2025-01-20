"""
RELEASE

* scale: intra-session
* div type: spacial disparity
* extra info: all users have >= 3 sessions per. affordance even after filtering non-pop artists.

"""


from divAtScale.src.helpers.semantic_helpers.diversity_measures import comp_gs_intra_or_inter_fs1, \
    comp_gs_intra_or_inter_fs2
from divAtScale.src.helpers.dataset_helpers.read_x_process import read_config, load_fs1, sub_sample_lh_df
from divAtScale.src.helpers.dataset_helpers.plotting_helpers import in_box, dataMirror, KDEplot
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


if __name__ == "__main__":

    config = read_config('./expr_config.json')
    YEAR = config['year']
    N_SAMPLE = config['n_sample']
    SEED = config['r_seed']
    data_dir = "./data/" + str(YEAR) + "_release"
    fig_dir = data_dir + "/Figs"
    res_dir = data_dir + "/res"
    pop_a = np.load(data_dir + "/pop_artists.npy")     # load niche artists
    plt.style.use('bmh')
    e = np.load(data_dir + "/e_balanced.npy")
    with open(data_dir + "/mid2aid_balanced", 'r') as json_file:
        mid2aid = json.load(json_file)  # matrix id --> artist_id
    aid2mid = {value: key for key, value in mid2aid.items()}

    # -------------------------------------------------------------------------------------------------

    print("\n*** Experiment 1.1.: intra-sess spatial disparity")

    ### Exp2. FS2
    print("** computing GS scores for FS2")
    multi_a_sess = pd.read_csv(data_dir + "/FS2/PQAE", index_col=0)

    # returned in order Q, P, A, E
    pdis_res, p_failed, avg_p_sess_left_post_niche_filt, uids_keep = comp_gs_intra_or_inter_fs2(multi_a_sess, e, aid2mid,
                                                                                                n_sample=N_SAMPLE, subset_a=pop_a)
    GS_p = [x[1] for x in pdis_res]
    GS_q = [x[0] for x in pdis_res]
    GS_a = [x[2] for x in pdis_res]
    GS_e = [x[3] for x in pdis_res]

    # znormlise GS
    pdis_res = [np.array(x) for x in pdis_res]

    pdis_res_znormed = [(x - np.mean(x)) / np.std(x) if np.std(x) != 0 else x - np.mean(x) for x in pdis_res]
    avg_gs_fs2 = np.mean(pdis_res_znormed, axis=0)
    std_gs_fs2 = np.std(pdis_res_znormed, axis=0)

    res_df = pd.DataFrame({"Aff":['P','Q','A','E'],
                           "GS_avg":[avg_gs_fs2[1], avg_gs_fs2[0], avg_gs_fs2[2], avg_gs_fs2[3]],
                           "GS_std":[std_gs_fs2[1], std_gs_fs2[0], std_gs_fs2[2], std_gs_fs2[3]],
                           "p_failed:" : [p_failed, p_failed, p_failed, p_failed],
                           "p_sess_left_post_niche_filt":[avg_p_sess_left_post_niche_filt[1],
                                                          avg_p_sess_left_post_niche_filt[0],
                                                          avg_p_sess_left_post_niche_filt[2],
                                                          avg_p_sess_left_post_niche_filt[3]]})
    res_df.to_csv(res_dir + "/intra_gs_fs2")
    print(res_df)

    # plot
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(4):
        y_data = [pdis[i] for pdis in pdis_res]

        for j in range(4):
            ax = axs[i,j]
            x_data = [pdis[j] for pdis in pdis_res]
            ymin = xmin = 0
            xmax = 1
            ymax = 1
            values = np.vstack([x_data, y_data])  # format into correct mde
            bounding_box = (xmin, xmax, ymin, ymax)
            values = dataMirror(values.T, bounding_box)
            KDEplot(xmin, xmax, ymin, ymax, values, ax)
            ax.scatter(*values, s=0.05, c='red', alpha=0.05)
            eq_line_x = np.linspace(0, 1, 100)
            ax.plot(eq_line_x, eq_line_x, color='black', linestyle='-', linewidth=1)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0.5, 1])
            ax.set_ylim([0.5, 1])

    plt.tight_layout()
    plt.savefig(fig_dir+"/fs2_gs_dist",  format='eps')
    plt.show()
