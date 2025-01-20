"""
Helper functions for reading & processing session based datasets 

"""

import pandas as pd
import json
import random


def u_activity_2_sessions(df, s_interval_thresh=20.0):
    """
    Segment users list. hist into sessions based on inter-stream thresh


    Args:
        df (pandas.DataFrame): listening history dataframe
        s_interval_thresh (float): inter-stream thresh to define new sess

    Returns:
        pandas df : one user's listening histories with a new col which indicates session num.
    """

    # Calculate the time differences between consecutive rows
    time_diff = df['ts_listen'].diff()

    # Identify the rows where the time difference exceeds the threshold
    session_start_indices = time_diff.gt(pd.Timedelta(minutes=s_interval_thresh))

    # Create a 'Session' column indicating the session number
    df['session'] = session_start_indices.cumsum()
    df['ts_listen'] = df.ts_listen.apply(lambda x: str(x))
    return df

def grab_sess(lh_df):
    """
    Get sessions from listening hist. df

    Args:
        lh_df (pandas.DataFrame): listening history df

    Returns:
        list : sessions
    """

    # updated to have group keys
    S = lh_df.groupby(["anon_user_id", 'session_n'], group_keys=False).artist_id.apply(list).to_frame().artist_id.values
    S = [list(s) for s in S]
    return S


def read_config(file_path):
    """
    Read configuration file used to run all experiments


    Args:
        file_path (str): path to config

    Returns:
        dict : config data
    """
    with open(file_path, 'r') as file:
        config_data = json.load(file)
    return config_data


def reformat_2022(sess_df):
    """
    Reformat to match 2018 year


    Args:
        sess_df (pandas.DataFrame): listening hist df

    Returns:
        pandas.DataFrame : formatted listening hist df
    """
    sess_df = sess_df.drop('ts_listen', axis=1)

    # Update the 'Age' column for rows where 'IsStudent' is True
    sess_df.loc[(sess_df.context_type == 'playlist_page') & (sess_df.context_4 == 'organic'), 'context_4'] = 'P'

    sess_df.rename(columns={'context_type': 'sub_affordance', 'context_4': 'affordance',
                            'hashed_uid': 'anon_user_id', 'session': 'session_n'}, inplace=True)

    # parse other affordance
    sess_df = sess_df[sess_df.affordance != 'ext']
    sess_df = sess_df[sess_df.affordance != 'other']

    aff_map = {'organic': 'o_s', 'P': 'o_pl', 'reco_algo': 'a', 'edito': 'e'}
    sess_df['affordance'] = sess_df.affordance.apply(lambda x: aff_map[x])
    return sess_df


def filter_u_with_less_than_thresh_sess(sess_df, thresh=3):
    """
    Subset df to only consider users who have > thresh sessions


    Args:
        sess_df (pandas.DataFrame): listening history df
        thresh (int): thresh indicates min number of sessions per. user

    Returns:
        pandas.DataFrame : filtered listening hist df.
    """
    n_sess_per_aff = sess_df.groupby(['anon_user_id']).session_n.nunique().to_frame().reset_index()
    uids = n_sess_per_aff[n_sess_per_aff.session_n >= thresh].anon_user_id.values
    sess_df = sess_df[sess_df.anon_user_id.isin(uids)]
    return sess_df


def sess_meta_data(sess_df):
    """
    Generate meta data: i.e. per (uid, sess) type: number of artists (n_a), session length (sess_l)


    Args:
        sess_df (pandas.DataFrame): listening history df

    Returns:
        pandas.DataFrame : meta_data df.
    """
    l_df = sess_df.groupby(['anon_user_id', 'session_n']).artist_id.count().to_frame().rename(
        columns={'artist_id': 'sess_l'})
    unique_a_df = sess_df.groupby(['anon_user_id', 'session_n']).artist_id.nunique().to_frame().rename(
        columns={'artist_id': 'n_a'})
    sess_meta_data = pd.merge(l_df, unique_a_df, left_index=True, right_index=True)
    # sess_meta_data['R'] = 1 - (sess_meta_data.n_a / sess_meta_data.sess_l)
    sess_meta_data.reset_index(inplace=True)
    return sess_meta_data


def print_le_stats(u_sess_df):
    """
    Print low level measures w.r.t listening history set


    Args:
        u_sess_df (pandas.DataFrame): listening history df
    """
    uids = u_sess_df.anon_user_id.unique()
    a_ids = u_sess_df.artist_id.unique()
    n_sessions = u_sess_df[['anon_user_id', 'session_n']].drop_duplicates().shape[0]

    n_users = len(uids)
    n_artists = len(a_ids)
    n_le = len(u_sess_df)
    print('* n users:', n_users)
    print('* n artists:', n_artists)
    print('* n sessions:', n_sessions)
    print('* n le:', n_le)
    print('')


def load_fs1(data_path):
    """
    Load all 4 fs1 datasets


    Args:
        data_path (str): optional, path to data_dir

    Returns:
        pandas.DataFrame : P,Q,A,E monadic, non-fandom sessions.

    """

    P = pd.read_csv(data_path + "/FS1/P", index_col=0)
    Q = pd.read_csv(data_path + "/FS1/Q", index_col=0)
    A = pd.read_csv(data_path + "/FS1/A", index_col=0)
    E = pd.read_csv(data_path + "/FS1/E", index_col=0)
    return P, Q, A, E


def sub_sample_lh_df(le_df, n_sample, seed):
    """
    Filter df to only contain n randomly sampled users

    Args:
        le_df (pandas.DataFrame) : listening history df
        n_sample (int) : n. users to sample
        seed (int) : random_seed

    Returns:
        pandas.DataFrame : sampled listening hist df.
    """
    random.seed(seed)
    uids = list(le_df.anon_user_id.unique())
    uids_sample = random.sample(uids, n_sample)
    le_df = le_df[le_df.anon_user_id.isin(uids_sample)]
    return le_df


def load_fs2_inter_condition(data_path="../../data/", thresh=3):
    """
    Load fs2 (multi-affordance u) dataset
    sub-sample based on users having > thresh sessions in each affordance.


    Args:
        y (int): year of listening hist.
        data_path (str) : path to fs2 dataset
        thresh (int):  thresh+1 indicates min number of sessions per. user

    Returns:
        pandas.DataFrame : filtered lh dataset for given affordance.
    """
    print('** loading df:')
    multi_a_sess = pd.read_csv(data_path + "/FS2/PQAE", index_col=0)
    n_sess_per_aff = multi_a_sess.groupby(['anon_user_id', 'affordance']).session_n.nunique().to_frame().reset_index()

    # must have at least 3 sessions per affordance
    n_multiple_aff_sess = n_sess_per_aff[n_sess_per_aff.session_n >= thresh].groupby(
        'anon_user_id').affordance.nunique().to_frame()
    keep_uids = list(n_multiple_aff_sess[n_multiple_aff_sess.affordance == 4].index)
    multi_a_sess = multi_a_sess[multi_a_sess.anon_user_id.isin(keep_uids)]
    return multi_a_sess


def load_session_df(data_path, y=2023, release=False):
    """
    Load sess df which contains all sess (monad,dyad,triad,tetrad) in March y

    Args:
        y (int): year to load

    Returns:
        pandas.DataFrame : listening hist df
    """

    if y == 2018:
        sess_df = pd.read_csv(data_path + "/u_session_df_2018", index_col=0)[['anon_user_id', 'artist_id',
                                                                                           'ts_listen', 'affordance',
                                                                                           'session_n']]
    elif y == 2022:
        sess_df = pd.read_csv(data_path + '/u_session_df_2022', index_col=0)
        if release == False:
            sess_df = reformat_2022(sess_df)
        else:
            sess_df.rename(columns={"session": "session_n"}, inplace=True)

    elif y == 2023:
        sess_df = pd.read_csv(data_path + '/u_session_df_2023', index_col=0)
        sess_df.rename(columns={"session": "session_n"}, inplace=True)
        #print('do something!')
    return sess_df
