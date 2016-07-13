import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from comput_functions import *


def stratification_KMeans(M_file, M_sep, A_file, A_sep):
    """
    This script computes the total number of mutations per patient
    and uses it to stratify patients with KMeans.

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

    Returns
    -------
    ass: pandas dataframe
        The patients' cluster assignment.
    """

    # Loads the mutation file
    X_lab = pd.read_csv(M_file, sep=M_sep, index_col=0, header=0)

    # Loads the adjacency matrix directly in a sparse format
    d = csv_to_csr(A_file, sep=A_sep)

    # Separates mutation profiles from the clinical data
    tmp = X_lab.loc[:, 'Days_survival':]
    # X records the raw binary mutation matrix
    X = X_lab.drop(list(tmp.columns), axis=1)

    # Get the mutation profiles and the adjacency matrix to have the same
    # columns in the same order
    X = get_mut_cols_as_adj_cols(X, d['colnames'])

    # Get the number of mutations per patient
    M = X.sum(axis=1)

    # Remove the patient who has more than 7000 mutations because he creates
    # one cluster where he is alone and this cluster influences too much the
    # log rank results (case SKCM).
    X = X.loc[M < 6000, :]
    M = M.loc[M < 6000]

    for N in range(2, 7):
        model = KMeans(n_clusters=N, init='k-means++', n_init=1000,
                       random_state=0, max_iter=1000)
        ass = model.fit_predict(np.array(M).reshape((len(M), 1)))
        ass = pd.DataFrame(ass, index=M.index)

    return ass
