import scipy.sparse as sp
import csv
import numpy as np
from scipy.stats import rankdata
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn import svm
import sys


##############################################################################
# ARGUMENT:
# file: string
# path to a CSV file.
# The first row of the CSV file should correspond to column names.
# The first element of the following rows should correspond to row names.
# Note: row names and column names have to be equal.

# sep: string
# delimiter of the CSV file

# VALUE:
# A dictionary whose keys give access to the following variables:
# key:'A', value: CSR sparse matrix corresponding to the CSV file witout

# row and column names.
# key: 'colnames', value: array of column names given in the CSV file.

# Example
# f = open('Desktop/test.csv')
# tmp = csv_to_csr(f)


def csv_to_csr(file, sep):
    """Read content of CSV file f, return as CSR matrix."""
    A = open(file)
    data = np.array([])
    indices = np.array([])
    indptr = np.array([0])
    rownames = np.array([])

    for i, row in enumerate(csv.reader(A, delimiter=sep)):
        if i == 0:
            colnames = np.array(row)
        else:
            rownames = np.append(rownames, row[0])
            row = np.array(map(float, row[1:]))
            shape1 = len(row)
            nonzero = np.where(row)[0]
            data = np.append(data, row[nonzero])
            indices = np.append(indices, nonzero)
            indptr = np.append(indptr, len(data))

    if np.array_equal(colnames, rownames) is not True:
        raise ValueError('The row names and column names in the CSV file are '
                         'not the same.')

    return {'A': sp.csr_matrix((
        data, indices, indptr), dtype=int, shape=(i, shape1)),
        'colnames': colnames}


##############################################################################
def quantile_norm(X):
    # Arguments:
    # X: Pandas DataFrame or numpy array
    # Value:
    # X_qn: numpy array or Pandas DataFrame according to input type.
    # X_qn is the quantile normalized version of X.

    typ = isinstance(X, pd.DataFrame)

    if typ:
        i = X.index
        c = X.columns
        X = X.as_matrix()

    ranks = np.zeros(X.shape)
    X_sorted = np.zeros(X.shape)

    for row in range(X.shape[0]):
        ranks[row, :] = rankdata(X[row, :], method='min')
        X_sorted[row, :] = sorted(X[row, :])

    ranks = ranks.astype(int)
    means = X_sorted.mean(axis=0)

    X_qn = means[ranks - 1]

    if typ:
        X_qn = pd.DataFrame(X_qn, index=i, columns=c)

    return X_qn


##############################################################################
def stair_norm(X, k):
    # """
    # Parameters
    # ----------
    # X: dataframe
    # k: int

    # Returns
    # -------
    # X: dataframe
    # """

    # Define the stair function
    p = X.shape[1]
    f = np.zeros((p, ))
    f[-int(k):] = 1
    # Match every patient mutation profile to the function f
    for i in range(X.shape[0]):
        rk = rankdata(X.iloc[i, :], method='ordinal') - 1
        rk = rk.astype(int)
        X.iloc[i, :] = f[rk]

    return X


##############################################################################
# Gets the comparable pairs of patients from their survival times and vital
# status. A pair of patients is considered comparable if the patient that has
# the shortest survival time is deceased, regardless of the vital status of
# the other patient. If two patients have the same survival times, then the
# pair is considered comparable is one of the patients is living while the
# other is deceased.
#
# Argument:
# y: The n times 2 labels matrix. The first column gives patients' survival
# times as floats and the second column gives patients' vital status
# ('deceased' if the patient is deceased and 'living' otherwise).
#
# Value:
# A matrix with n rows and 2 columns. Each row gives the indices of a
# comparable pair of patients, where the patient in the second column lives
# longer than the patient in the first column. If the reported pair concerns
# patients with the same survival time, then the patient in the first column
# is the one that is deceased.
#
# Examples
# days_surv = np.array([0, 0, 1, 1, 1, 2, 3, 4, 5])
# vital_st = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0])
# y = np.vstack([days_surv, vital_st]).T

# res = to_pairs(y)

def to_pairs(y):
    n = y.shape[0]
    V1 = np.array([])
    V2 = np.array([])

    for i in range(n):
        if y[i, 1] == 'deceased':
            mask = (y[:, 0] > y[i, 0]) +\
                   ((y[:, 0] == y[i, 0]) & (y[:, 1] == 'living'))
            V1 = np.hstack([V1, np.ones(sum(mask))*i])
            V2 = np.hstack([V2, np.arange(n)[mask]])

    return np.vstack([V1, V2]).T.astype('int')


##############################################################################
# Function description:
# For the smoothing process as well as for other tasks, the adjacency matrix
# and the mutation matrix need to have the same genes in the same order.
# This function takes a mutation dataframe X, removes from it the genes that
# are not in the adjacency dataframe A, and adds to it the genes that are in A
# but not in X. (These added genes are 0 vectors, i.e there are no
# mutations in these genes)

# Arguments
# X: patient x genes pandas DATAFRAME.
# colnames: column names of the adjacency matrix.

# Value:
# df: a new mutation DATAFRAME with the same genes as A,
# (and in the same order).

# set parameters
# X = pd.read_csv('/Users/lemorvan/Desktop/Mut_impact_Project/'
#                      'Data_preparation/LUAD/'
#                      'luad_mut_matrix_GA_ordered2.csv',
#                      sep=';', index_col=0, header=0)

# A = pd.read_csv('/Users/lemorvan/Desktop/Mut_impact_Project/'
#                 'reproduce_Hofree2014_results/A_PC',
#                 sep=' ', index_col=0, header=0)


def get_mut_cols_as_adj_cols(X, colnames):

    df = pd.DataFrame(0, columns=colnames, index=X.index)
    intersec = np.intersect1d(colnames, X.columns, assume_unique=True)
    df.loc[:, intersec] = X.loc[:, intersec]
    return df


##############################################################################
# Desription
# Takes a raw mutation dataframe (binary patient x genes dataframe)
# and transform it into a new dataframe where all mutated genes have
# been assigned values between 1 and 2 according their respective
# number of neighbours; and all genes neighbouring a mutated gene in a
# given patient are assigned values between 0 and 1 according to the
# number of mutated genes they actually neighbour.

# Argument:
# X: Binary mutation DATAFRAME
# A: adjacency SPARSE MATRIX
# X and A must have the same columns in the same order.

# Value:
# tmp: New mutation DATAFRAME

def get_mut_neigh(X, A):

    nrow = X.shape[0]

    # Define the mutation weights for each gene
    # (i.e the normalized number of neighbours of each gene +1).
    weight = np.array(A.sum(axis=1))
    weight = weight/float(max(weight)) + 1

    tXA = A.T.dot(X.T)
    norm_vec = np.max(tXA, axis=0).astype('float')
    tmp = (tXA/norm_vec).T*(1-X) + np.repeat(weight.T, nrow, 0)*X

    return tmp


##############################################################################
# Desription
# Takes a raw mutation dataframe (binary patient x genes dataframe)
# and transform it into a new dataframe where genes are ranked first according
# to their mutational status and then according to their number of mutated
# neighbours

# Argument:
# X: Binary mutation DATAFRAME
# A: adjacency SPARSE MATRIX
# X and A must have the same columns in the same order.

# Value:
# tmp: New mutation DATAFRAME

def get_mut_neigh_DAD(X, A):

    # Define the diagonal degree matrix
    weight = np.array(A.sum(axis=1))[:, 0]
    sD = sp.spdiags(1.0 / np.sqrt(weight), 0, A.shape[0], A.shape[0])
    A_norm = sD.dot(A).dot(sD)
    iter1 = (A_norm.T.dot(X.T)).T

    tmp = iter1 + np.max(iter1)*X

    return tmp


##############################################################################
# Function description:
# Smoothes a mutation dataframe the way Idecker did in Nature methods 2013

# Parameters:
# X_data_0: DATAFRAME
# mutation dataframe obtained with get_mut_cols_as_adj_cols.py
# (important because X_data_0 and A need to have the same genes in the same
# order)

# alpha: float between 0 and 1
# tuning parameter governing the distance that a mutation signal is
# allowed to diffuse through the network during propagation

# A: SPARSE matrix
# adjacency matrix of a gene interaction network.

# Value:
# X_data_new: DATAFRAME
# the smoothed muation dataframe.

# set parameters
# from get_mut_cols_as_adj_cols import get_mut_cols_as_adj_cols

# X_data_0 = pd.read_csv('/Users/lemorvan/Desktop/Mut_impact_Project/'
#                        'Data_preparation/LUAD/'
#                        'luad_mut_matrix_GA_ordered2.csv',
#                        sep=';', index_col=0, header=0)

# A = pd.read_csv('/Users/lemorvan/Desktop/mutimpact/data/Networks/'
#                 'data_preprocessed/A_PC',
#                 sep=' ', index_col=0, header=0)

# X_data_0 = get_mut_cols_as_adj_cols(X_data_0, A)

# A = sp.csr_matrix(A)

# alpha = 0.5


def network_smooth_DAD(X_data_0, alpha, A, qn='qn1'):

    sD = sp.spdiags(1.0 / np.sqrt(np.array(A.sum(axis=1))[:, 0]), 0,
                    A.shape[0], A.shape[0])
    A_norm = sD.dot(A).dot(sD)

    tmp = (1-alpha)*X_data_0
    X_data_new = X_data_0
    n = 1

    while(n > 1e-5):
        X_data = X_data_new
        X_data_new = alpha * (A_norm.T.dot(X_data.T)).T + tmp
        n = np.linalg.norm(X_data_new-X_data, ord='fro')

    if qn == 'qn1':
        X_data_new = quantile_norm(X_data_new)

    return X_data_new


##############################################################################
# Function description:
# Takes a raw mutation dataframe (binary patient x genes dataframe)
# and transform it into a new dataframe with only one iteration
# of Idecker smoothing process.

# Parameters:
# X_data_0: DATAFRAME
# mutation dataframe obtained with get_mut_cols_as_adj_cols.py
# (important because X_data_0 and A need to have the same genes in the same
# order)

# alpha: float between 0 and 1
# tuning parameter governing the distance that a mutation signal is
# allowed to diffuse through the network during propagation

# A: SPARSE matrix
# adjacency matrix of a gene interaction network.

# Value:
# X_data_new: DATAFRAME
# the smoothed muation dataframe.


def network_smooth_1iter_DAD(X_data_0, alpha, A, qn='qn1'):

    sD = sp.spdiags(1.0 / np.sqrt(np.array(A.sum(axis=1))[:, 0]), 0,
                    A.shape[0], A.shape[0])
    A_norm = sD.dot(A).dot(sD)

    X_data_new = alpha * (A_norm.T.dot(X_data_0.T)).T + (1-alpha)*X_data_0

    if qn == 'qn1':
        X_data_new = quantile_norm(X_data_new)

    return X_data_new


##############################################################################
# This function computes exactly the same concordance index as the function
# estC from the R package compareC.

# There are two differences with CI_score.py:

# - the first one is about pairs of patients that have the same observed
# survival times. In CI_score.py, such pairs where considered as non
# comparable. In this function, such a pair is considered comparable when one
# of the patients is living while the other is deceased. This pair is
# concordant with the prediction if the living patient lives longer than the
# deceased one.

# - the second difference is about pairs of patients that have the same
# predicted survival times. In CI_score.py, such a pair contributes to 0
# to the concordance index. In this function, it contributes to 0.5 to the
# concordance index. (in both cases, a concordant pair contributes to 1 and
# a non concordant pair contributes to 0)

# Arguments
# y: DATAFRAME
# The n times 2 labels dataframe. The first column gives patients' survival
# times as floats and the second column gives patients' vital status
# ('deceased' if the patient is deceased and 'living' otherwise).
# prediction: array
# The n times 0 labels array. Gives the predicted survival times as floats
# for each patient.

# value: float
# The obtained concordance index.

# Example
# import numpy as np
# import pandas as pd
# days_surv = np.array([0, 0, 1, 1, 1, 2, 3, 4, 5])
# vital_st = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0])
# y = np.vstack([days_surv, vital_st]).T
# y = pd.DataFrame(y)
# y.columns = ['Days_survival', 'Vital_Status']


def CI_score_estC(y, prediction):

    y = y.get_values()
    pairs = to_pairs(y)
    n = len(pairs)

    # Compute the concordance index for the patients in the test set.
    tmp1 = y[:, 0][pairs[:, 1]] - y[:, 0][pairs[:, 0]]
    tmp2 = prediction[pairs[:, 1]] - prediction[pairs[:, 0]]
    tmp = (tmp1*tmp2) > 0
    equals1 = tmp1 == 0
    equals2 = tmp2 == 0
    CI_test = (tmp.sum() + (equals1*tmp2 > 0).sum() +
               equals2.sum()/2. + (equals1*equals2).sum())/n
    return CI_test


##############################################################################
class L1survSVM_stair3(BaseEstimator):

    """
    PARAMETERS:
    C: (float) regularisation parameter for the VanBelle model with L1 penalty.
    k: (int) parameter controlling the stair function.

    ATTRIBUTES:
    None

    METHODS:
    fit(X, y)
    predict(X)

    PARAMETERS OF THE METHODS:
    X: patients times genes dataframe, concatenated with a
       patients times clinical features dataframe (used for training in
       fit, for predicting in predict).
    y: patients times ['Days_survival', 'Vital_Status'] dataframe.
       Patients must obviously be given in the same order as in X.
    """

    def __init__(self, C, k):
        self.C = C
        self.k = k

    def fit(self, X, y):

        # Define the stair function
        p = X.shape[1]
        f = np.zeros((p, ))
        f[-self.k:] = 1
        # Match every patient mutation profile to the function f
        for i in range(X.shape[0]):
            rk = rankdata(X.iloc[i, :], method='ordinal') - 1
            rk = rk.astype(int)
            X.iloc[i, :] = f[rk]

        X = sp.csr_matrix(X)

        y = y.get_values()
        pairs = to_pairs(y)

        # --------------------------------------------
        # Create and solve a one-class SVM problem
        # --------------------------------------------
        X2 = X[pairs[:, 1], :] - X[pairs[:, 0], :]

        # svm.LinearSVC needs at least two different classes.
        # So we take the opposite of half the samples and we
        # assign a class y=-1 to these new samples
        # compared to a class y=1 to the original samples.
        r = X2.shape[0]/2
        lab = np.ones(X2.shape[0])
        lab[r:] = -1
        X2 = (X2.T.multiply(sp.csr_matrix(lab))).T

        # Return the lowest bound for C such that for C in (l1_min_C, infinity)
        # the model is guaranteed not to be empty
        # min_C = svm.l1_min_c(X_sub_train2, y)

        clf = svm.LinearSVC(C=self.C, loss='squared_hinge', penalty='l1',
                            fit_intercept=False, dual=False, random_state=0)
        clf.fit(X2, lab)
        self.clf_ = clf

        print('SVM done')
        sys.stdout.flush()

        return self

    def predict(self, X):

        # Define the stair function
        p = X.shape[1]
        f = np.zeros((p, ))
        f[-self.k:, ] = 1
        # Match every patient mutation profile to the function f
        for i in range(X.shape[0]):
            rk = rankdata(X.iloc[i, :], method='ordinal') - 1
            rk = rk.astype(int)
            X.iloc[i, :] = f[rk]

        X = sp.csr_matrix(X)

        pred = X.dot(self.clf_.coef_.T)
        pred = pred[:, 0]
        return pred


##############################################################################
class L1survSVM_qn3(BaseEstimator):

    """
    PARAMETERS:
    C: (float) regularisation parameter for the VanBelle model with L1 penalty.
    qn: 'qn1' if quantile normalisation should be performed, 'qn0' otherwise.
        withclin: (bool) if True, clinical features will be used along with
        mutation data in the model.

    ATTRIBUTES:
    None

    METHODS:
    fit(X, y)
    predict(X)

    PARAMETERS OF THE METHODS:
    X: patients times genes dataframe, concatenated with a
       patients times clinical features dataframe (used for training in
       fit, for predicting in predict).
    y: patients times ['Days_survival', 'Vital_Status'] dataframe.
       Patients must obviously be given in the same order as in X.
    """

    def __init__(self, C, qn):
        self.C = C
        self.qn = qn

    def fit(self, X, y):

        if self.qn == 'qn1':
            # Perform quantile normalisation
            X = quantile_norm(X)

        y = y.get_values()
        pairs = to_pairs(y)

        # --------------------------------------------
        # Create and solve a one-class SVM problem
        # --------------------------------------------
        X = X.get_values()
        X2 = X[pairs[:, 1], :] - X[pairs[:, 0], :]

        # svm.LinearSVC needs at least two different classes.
        # So we take the opposite of half the samples and we
        # assign a class y=-1 to these new samples
        # compared to a class y=1 to the original samples.
        r = X2.shape[0]/2
        lab = np.ones(X2.shape[0])
        lab[r:] = -1
        X2 = (X2.T*lab).T

        # Return the lowest bound for C such that for C in (l1_min_C, infinity)
        # the model is guaranteed not to be empty
        # min_C = svm.l1_min_c(X2, lab)

        clf = svm.LinearSVC(C=self.C, loss='squared_hinge', penalty='l1',
                            fit_intercept=False, dual=False, random_state=0)
        clf.fit(X2, lab)
        self.clf_ = clf

        print('SVM done')
        sys.stdout.flush()

        return self

    def predict(self, X):

        if self.qn == 'qn1':
            # Perform quantile normalisation
            X = quantile_norm(X)

        X = X.get_values()

        pred = X.dot(self.clf_.coef_.T)
        pred = np.array(pred)[:, 0]
        return pred


##############################################################################
class L1survSVM_qn_sm3(BaseEstimator):

    """
    PARAMETERS:
    C: (float) regularisation parameter for the VanBelle model with L1 penalty.
    ind_a: (int) index of the value of the smoothing parameter alpha to be
           used for smoothing. (the smoothing parameters alpha to be tested are
           given as a list)
    qn: 'qn1' if quantile normalisation should be performed, 'qn0' otherwise.
    q: (int) number of genes in the smoothed mutation matrix (before
        concatenation), or equivalently index of the first clinical feature in
        the smoothed mutation matrix.


    ATTRIBUTES:
    None

    METHODS:
    fit(X, y)
    predict(X)

    PARAMETERS OF THE METHODS:
    X: patients times genes dataframe constructed by the concatenation of
       the smoothed patients times genes matrices. The smoothed matrices are
       concatenated by increasing value of alpha.
       (used for training in fit, for predicting in predict).
    y: patients times ['Days_survival', 'Vital_Status'] dataframe.
       Patients must obviously be given in the same order as in X.
    """

    def __init__(self, C, ind_a, qn, q):
        self.C = C
        self.ind_a = ind_a
        self.qn = qn
        self.q = q

    def fit(self, X, y):

        X = X.iloc[:, self.ind_a*self.q:(self.ind_a+1)*self.q]

        if self.qn == 'qn1':
            # Perform quantile normalisation
            X = quantile_norm(X)

        y = y.get_values()
        pairs = to_pairs(y)

        # --------------------------------------------
        # Create and solve a one-class SVM problem
        # --------------------------------------------
        X = X.get_values()
        X2 = X[pairs[:, 1], :] - X[pairs[:, 0], :]

        # svm.LinearSVC needs at least two different classes.
        # So we take the opposite of half the samples and we
        # assign a class y=-1 to these new samples
        # compared to a class y=1 to the original samples.
        r = X2.shape[0]/2
        lab = np.ones(X2.shape[0])
        lab[r:] = -1
        X2 = (X2.T*lab).T

        # Return the lowest bound for C such that for C in (l1_min_C, infinity)
        # the model is guaranteed not to be empty
        # min_C = svm.l1_min_c(X2, lab)

        clf = svm.LinearSVC(C=self.C, loss='squared_hinge', penalty='l1',
                            fit_intercept=False, dual=False, random_state=0)
        clf.fit(X2, lab)
        self.clf_ = clf

        print('SVM done')
        sys.stdout.flush()

        return self

    def predict(self, X):

        X = X.iloc[:, self.ind_a*self.q:(self.ind_a+1)*self.q]

        if self.qn == 'qn1':
            # Perform quantile normalisation
            X = quantile_norm(X)

        X = X.get_values()

        pred = X.dot(self.clf_.coef_.T)
        pred = np.array(pred)[:, 0]
        return pred
