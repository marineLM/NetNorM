from __future__ import division
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import NMF
import random
from comput_functions import *


def stratification_ccNMF(M_file, M_sep, A_file, A_sep, method_rep, method_norm,
                         k, alpha, randomize, rs_rand, N):

    """
    Preprocess the mutation data according to the method specified, and then
    performs consensus clustering (NMF). Consensus clustering is done in
    2 steps: first a subsampling is performed, ie 80% of the samples and 80%
    of the features are randomly chosen. Then NMF is applied to this subsample
    and each patient is assigned to the component that represents him best
    (highest weight). The subsampling followed by NMF and cluster assignment
    is repeated 1000 times.
    A consensus matrix is then derived from these 1000 assignements.

    Parameters
    ----------
    M_file: str
        Path to the raw binary mutation matrix file.
        It must be a csv file with patients as rows and genes as columns.The
        first row must be gene names. The first column must be patient IDs.
        The last columns must be clinical features, and the 2 first clinical
        features must be named 'Days_survival', 'Vital_Status'.
        'Days_survival' (int) indicate how many days a patient has survived
        since diagnosis.
        'Vital_Status' (string) take values 'deceased' and 'living', according
        to the censoring information of the patient.

    M_sep: str
        Delimiter to read the csv file located at M_file.

    A_file: str
        Path to the adjacency matrix
        It must be a csv file with the first row and the first column being
        gene names.

    A_sep: str
        Delimiter to read the csv file located at A_file.

    method_rep: str
        The method used to preprocess the mutation matrix before normalisation
        The options are 'raw', 'rawrestricted', 'smoothing',
        'smoothing_1iter_DAD', 'neighbours', 'neighbours_DAD'

        'raw':
            the mutation matrix is kept as a raw binary matrix

        'rawrestricted': the mutation matrix is kept as a raw binary matrix but
            restricted to the genes that are listed in the gene network.

        'smoothing':
            the mutation matrix is smoothed over a network.

        'smoothing_1iter_DAD':
            the mutation matrix is smoothed over a network, but the smoothing
            process is restricted to happen between a gene and its direct
            neighbours only.

        'neighbours':
            for each patient independently, mutated genes are set to
            values between 1 and 2 according their respective number of
            neighbours; non mutated genes neighbouring a mutated gene are set
            to values between 0 and 1 according to the number of mutated genes
            they actually neighbour.

        'neighbours_DAD':
            for each patient independently, genes are set to a
            score equal to their number of mutated neighbours normalised by
            their degree and by the degree of their neighbours. Then mutated
            genes' scores are increased by a constant C to ensure that their
            score is always higher than those of non mutated genes.

    method_norm: str
        The method used to normalize mutation profiles. The options are
        'none', 'qn', 'stair'

        'none':
            no normalization is performed.

        'qn':
            quantile normalization.

        'stair':
            every mutation profile is binarised with the same number of 0s
            and 1s.

    randomize: boolean
        Specify wether the adjacency matrix should be randomized (True) or
        not (False).

    rs_rand: int
        Random state used to randomize this adjacency matrix.

    k: int
        The value for the parameter k (consensus number of mutations when
        'method_norm' is set to 'stair')

    alpha: int
        The value for the parameter alpha (smoothing parameter
        used when 'method_rep' is set to 'smoothing' or 'smoothing_1iter_DAD')

    N: int
        The number of patient subtypes (parameter for the NMF)

    IMPORTANT:
    Gene names in M_file and A_file must be written in the same format
    (for ex HUGO) so that it is possible to match genes between the 2 files.

    Returns
    -------
    freq: (n_samples, n_samples) array.
        The consensus matrix. Each entry corresponds to the frequency at which
        two patients where clustered in the same group over all samplings
        where both patients were retained.
    """

    # Loads the mutation file
    X_lab = pd.read_csv(M_file, sep=M_sep, index_col=0, header=0)

    # Loads the adjacency matrix directly in a sparse format
    if method_rep != 'raw':
        d = csv_to_csr(A_file, sep=A_sep)
        if randomize:
            random.seed(int(rs_rand))
            tmp = random.sample(d['colnames'], d['A'].shape[1])
            d['colnames'] = np.array(tmp)

    # Separates mutation profiles from the clinical data
    tmp = X_lab.loc[:, 'Days_survival':]
    # X records the raw binary mutation matrix
    X = X_lab.drop(list(tmp.columns), axis=1)

    # Get the mutation profiles and the adjacency matrix to have the same
    # columns in the same order
    if method_rep != 'raw':
        X = get_mut_cols_as_adj_cols(X, d['colnames'])

    # Transform the mutation profiles according to a given representation and
    # a given normalisation scheme.
    if method_rep == 'smoothing':
        X = network_smooth_DAD(X, float(alpha), d['A'], qn='qn0')
    elif method_rep == 'smoothing_1iter_DAD':
        X = network_smooth_1iter_DAD(X, float(alpha), d['A'], qn='qn0')
    elif method_rep == 'neighbours':
        X = get_mut_neigh(X, d['A'])
    elif method_rep == 'neighbours_DAD':
        X = get_mut_neigh_DAD(X, d['A'])

    if method_norm == 'qn':
        X = quantile_norm(X)
    elif method_norm == 'stair':
        # Define the stair function
        p = X.shape[1]
        f = np.zeros((p, ))
        f[-int(k):] = 1
        # Match every patient mutation profile to the function f
        for i in range(X.shape[0]):
            rk = rankdata(X.iloc[i, :], method='ordinal') - 1
            rk = rk.astype(int)
            X.iloc[i, :] = f[rk]

    np.random.seed(0)
    # np.random.seed(1)
    s_c = int(X.shape[1] * 0.8)
    s_r = int(X.shape[0] * 0.8)
    count = np.zeros((X.shape[0], X.shape[0]))
    count = pd.DataFrame(count, index=X.index, columns=X.index)
    cooc = np.zeros((X.shape[0], X.shape[0]))
    cooc = pd.DataFrame(cooc, index=X.index, columns=X.index)
    for i in range(1000):
        print i
        ind_c = np.random.choice(X.shape[1], size=s_c, replace=False)
        ind_r = np.random.choice(X.shape[0], size=s_r, replace=False)
        X_tmp = X.iloc[ind_r, ind_c]
        model = NMF(n_components=N, init='nndsvda', random_state=0)
        W = model.fit_transform(np.array(X_tmp))
        ass = np.argmax(W, axis=1)
        ass_all = pd.DataFrame(np.ones(X.shape[0]) * N, index=X.index)
        ass_all.loc[X_tmp.index, 0] = ass
        ass_d = pd.get_dummies(ass_all.loc[:, 0])
        ass_d2 = 1 - ass_d.loc[:, N]
        cooc_tmp = np.outer(ass_d2, ass_d2)
        ass_d.loc[:, N] = 0
        count_tmp = ass_d.dot(ass_d.T)
        cooc = cooc + cooc_tmp
        count = count + count_tmp
    freq = count / cooc

    return freq
