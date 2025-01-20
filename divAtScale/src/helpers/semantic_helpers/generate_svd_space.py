import numpy as np
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import os
import scipy
import random
import warnings
from divAtScale.src.helpers.dataset_helpers.read_x_process import load_fs1
import json


class SVD_builder(object):
    """SVD builder

    Contains various fine-grained capabilities including:
    1. svd construction with balanced affordances
    2. saving ppmi matrix pre-embedding for later analysis

    """

    def __init__(self, balanced, base_path):
        """
        Args:
            balanced (bool): option for balanced affordance construction
            base_path (str): path to data dir
            y (int): year
        """
        self.e = None  # solution of the equation system
        self.mid2aid_lookup = None
        self.balanced = balanced
        self.data_dir = base_path

    def create_folder_if_not_exists(self, folder_path):
        """
        As per function name

        Args:
            folder_path (str): path to check and create dir
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created.")
        else:
            print(f"Folder '{folder_path}' already exists.")

    def create_co_occurences_matrix(self, allowed_artists, sessions):
        """
        Construct artist-artist co-occurance matrix given list of sessions

        Args:
            allowed_artists (list): if we wish to focus on a subset of artists to construct matirx.
            sessions (list) : a list of sessions containing artist_ids streamed by users.

        Returns:
            scipy.sparse.csr_matrix : co-occurance matrix
            dict : artist id 2 matrix index lookup
        """
        artist_to_id = dict(zip(allowed_artists, range(len(allowed_artists))))
        documents_as_ids = [np.sort([artist_to_id[w] for w in s if w in artist_to_id]).astype('uint32') for s in
                            sessions]
        row_ind, col_ind = zip(*itertools.chain(*[[(i, w) for w in doc] for i, doc in enumerate(documents_as_ids)]))
        data = np.ones(len(row_ind), dtype='uint32')  # use unsigned int for better memory utilization
        max_word_id = max(itertools.chain(*documents_as_ids)) + 1
        docs_words_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(documents_as_ids), max_word_id))
        words_cooc_matrix = docs_words_matrix.T * docs_words_matrix
        words_cooc_matrix.setdiag(0)
        return words_cooc_matrix, artist_to_id

    def ppmi(self, A):
        """
        Compute positive point wise mutual information matrix from A

        Args:
            A (scipy.sparse.csr_matrix): co-occurance matrix to run ppmi on

        Returns:
            scipy.sparse.csr_matrix : ppmi matrix
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            total = A.sum()
            pr = total / A.sum(axis=1).A1
            pc = total / A.sum(axis=0).A1
            pr[~np.isfinite(pr)] = 0
            pc[~np.isfinite(pc)] = 0

            # Calculate the joint probability p_ij
            A = A / total

            # Calculate p_ij / p_i * p_j
            A = A.multiply(pr[:, None]).multiply(pc[None, :])
            A.eliminate_zeros()

            # Calculate your metric
            A.data = np.log2(A.data)
            A.data[A.data < 0] = 0  # only take positive
        return A

    def grab_sess(self, sess_df):
        """
        Get sessions from listening history & filter artist repeats & fandom sessions

        Args:
            sess_df (pandas.DataFrame): listening history df

        Returns:
            list : filtered sessions
        """
        X = sess_df.groupby(["anon_user_id", 'session_n']).artist_id.unique().values
        X = [list(x) for x in X if len(x) > 1]  # drop fandom sess
        return X

    def generate_e(self, sess_df, save_pmi_matrix=True):
        """
        Pipeline to generate svd-based artist embedding from session data.
        saves embedding matrix and mid2aid lookup as class variables.

        Args:
            sess_df (pandas.DataFrame): listening history df
            save_pmi_matrix (bool): optional - save ppmi matrix
        """
        print("* Computing SVD based embedding space:")
        balance_affordances = self.balanced
        save_path = self.data_dir

        # optional : only utilise monadic sessions & re-balance sessions:
        if balance_affordances:

            print("* balancing affordances in matrix consturction")
            P_sess, Q_sess, A_sess, E_sess = load_fs1(data_path=self.data_dir)
            P_sess = self.grab_sess(P_sess)
            Q_sess = self.grab_sess(Q_sess)
            A_sess = self.grab_sess(A_sess)
            E_sess = self.grab_sess(E_sess)

            train_data = [P_sess, Q_sess, A_sess, E_sess]
            max_sess_l = max([len(P_sess), len(Q_sess), len(A_sess), len(E_sess)])

            # over-sample minortiy classes
            updated_sess = []
            for S in train_data:
                if len(S) < max_sess_l:
                    diff_n = max_sess_l - len(S)
                    duplicated_sess = random.choices(S, k=diff_n)
                    S = S + duplicated_sess
                updated_sess.append(S)
            X = updated_sess[0] + updated_sess[1] + updated_sess[2] + updated_sess[3]
        else:
            X = sess_df.groupby(["anon_user_id", 'session_n']).artist_id.unique().values

        documents = [list(x) for x in X if len(x) > 1]  # drop fandom sess
        print('n sess:', len(documents))  # n sess
        f = list(itertools.chain.from_iterable(documents))  # flat
        vocab = list(set(f))

        M, aid2mid = self.create_co_occurences_matrix(vocab, documents)
        M_pmi = self.ppmi(M)
        print(M.shape)

        M_pmi = M_pmi.astype(float)

        if save_pmi_matrix:
            if balance_affordances:
                scipy.sparse.save_npz(save_path + '/pmi_M_aff_balanced.npz', M_pmi)
            else:
                scipy.sparse.save_npz(save_path + '/pmi_M.npz', M_pmi)
            print('* saved pmi matrix!')

        u, s, vT = svds(M_pmi, k=128, random_state=3)
        E = u @ np.diag(s)
        print(E.shape)
        self.e = E
        mid2aid = {value: key for key, value in aid2mid.items()}
        self.mid2aid_lookup = mid2aid

    def save_all(self):
        """
        Save embedding (e) and mid2aid lookup to memory

        """
        self.create_folder_if_not_exists(self.data_dir)
        if self.balanced:
            np.save(self.data_dir + "/e_balanced", self.e)
            with open(self.data_dir + "/mid2aid_balanced", 'w') as f:
                json.dump(self.mid2aid_lookup, f, default=int)
        else:
            np.save(self.data_dir + "/e", self.e)
            with open(self.data_dir + "/mid2aid", 'w') as f:
                json.dump(self.mid2aid_lookup, f, default=int)
