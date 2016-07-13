from __future__ import division
from sklearn.metrics import make_scorer
from sklearn import cross_validation
from sklearn import grid_search
import numpy as np
import pandas as pd
import random
from comput_functions import *


def survpred(M_file, M_sep, A_file, A_sep, method_rep, method_norm, rs_folds,
             randomize, rs_rand, k_start, k_step, k_end, alpha_start,
             alpha_step, alpha_end, data_type, n_jobs):

    """
    Preprocess the mutation data according to the method specified, and split
    the dataset into train/test tests according to a 5-fold
    cross-validation scheme. Then for each split into train/test sets, a
    L1-penalized survival SVM model is learned on the train set and tested on
    the test set. Hyperparameters are learned thanks to an inner
    cross-validation on the train set, and then used to train the model on the
    whole train set.

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
        The options are 'nb_mut', raw', 'rawrestricted', 'smoothing',
        'smoothing_1iter_DAD', 'neighbours', 'neighbours_DAD'

        'nb_mut':
            the mutation matrix is restricted to one feature indicating the
            total number of mutations for each patient.

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
            quantile normalization (performed after each splitting in
            train/test).

        'stair':
            every mutation profile is binarised with the same number of 0s
            and 1s.

    rs_folds: int
        Random state used to generate the 5 folds for cross-validation.

    randomize: boolean
        Specify wether the adjacency matrix should be randomized (True) or
        not (False).

    rs_rand: int
        Random state used to randomize this adjacency matrix.

    k_start: int
        smallest value tested for the parameter k (consensus number of
        mutations when 'method_norm' is set to 'stair') during the grid
        search.

    k_step: int
        step controlling the values of k that will be tested between
        k_start and k_end.

    k_end: int
        largest value tested for the parameter k (consensus number of
        mutations when 'method_norm' is set to 'stair') during the grid
        search.

    alpha_start: int
        smallest value tested for the parameter alpha (smoothing parameter
        used when 'method_rep' is set to 'smoothing' or 'smoothing_1iter_DAD')
        during the grid search.

    alpha_step: int
        step controlling the values of alpha that will be tested between
        alpha_start and alpha_end.

    alpha_end: int
        largest value tested for the parameter alpha (smoothing parameter
        used when 'method_rep' is set to 'smoothing' or 'smoothing_1iter_DAD')
        during the grid search.

    data_type: str
        Indicate whether a survival prediction model should be learned using
        mutation data only, clinical data only, or both (in this case the 2
        are still learned separately)

        'clin':
            the survival prediction model is learned with clinical data
        'mut':
            the survival prediction model is learned with mutation data
        'both':
            survival prediction models are learned with both types of data
            (separately though)

    n_jobs: int
        the number of cores to use.

    IMPORTANT:
    Gene names in M_file and A_file must be written in the same format
    (for ex HUGO) so that it is possible to match genes between the 2 files.


    Returns
    -------
    l: list
        If 'data_type' is set to 'clin' or 'mut', the list contains 12
        elements. Each of these elements is itself a list or an array of
        length 5 corresponding to the 5 splits in train/test of the
        cross-validation. These containers of length 5 record:
        #0: Concordance index (CI) obtained on the train set.
        #1: Concordance index (CI) obtained on the test set.
        #2: the variables selected in the model (genes or clinical features)
        #3: the parameters ((k and C) or (alpha and C) according to the
            method used) that was learned with the inner cross-validation.
        #4: the coefficients learned for each variable.
        #5: Concordance index (CI) obtained on the train set for patients with
            more than the mean number of mutations.
        #6: Concordance index (CI) obtained on the test set for patients with
            more than the mean number of mutations.
        #7: Concordance index (CI) obtained on the train set for patients with
            less than the mean number of mutations.
        #8: Concordance index (CI) obtained on the test set for patients with
            less than the mean number of mutations.
        #9: Mean concordance index (CI) obtained across in the inner
            cross-validation folds with the set of parameters that yields the
            best such mean.
        #10: survival predictions for patients in the train set
        #11: survival predictions for patients in the test set
    """

    # Loads the mutation data
    X_lab = pd.read_csv(M_file, sep=M_sep, index_col=0, header=0)

    # Loads the adjacency matrix directly in a sparse format and randomises it
    # if 'randomize' is set to True.
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
    # y records the variables 'Days_survival' and 'Vital_Status'
    y = tmp.loc[:, ['Days_survival', 'Vital_Status']]
    # clin records all the available clinical variables apart from those
    # recorded in y
    clin = tmp.drop(['Days_survival', 'Vital_Status'], axis=1)

    # Get the mutation profiles and the adjacency matrix to have the same
    # columns in the same order
    if method_rep != 'raw':
        X = get_mut_cols_as_adj_cols(X, d['colnames'])

    # Get the number of mutations per patient
    M = X.sum(axis=1)
    med = np.median(M)
    # Get the number of genes in the design matrix
    q = X.shape[1]

    if method_rep == 'smoothing':
        alpha = list(np.arange(float(alpha_start), float(alpha_end),
                     float(alpha_step)))
        ind_a = range(len(alpha))
        X_tot = network_smooth_DAD(X, float(alpha[0]), d['A'], qn='qn0')
        if len(alpha) > 1:
            for a in alpha[1:]:
                X_tmp = network_smooth_DAD(X, float(a), d['A'], qn='qn0')
                X_tot = pd.concat([X_tot, X_tmp], axis=1)
        X = X_tot
        del X_tot

    elif method_rep == 'smoothing_1iter_DAD':
        alpha = list(np.arange(float(alpha_start), float(alpha_end),
                     float(alpha_step)))
        ind_a = range(len(alpha))
        X_tot = network_smooth_1iter_DAD(X, float(alpha[0]), d['A'], qn='qn0')
        if len(alpha) > 1:
            for a in alpha[1:]:
                X_tmp = network_smooth_1iter_DAD(
                    X, float(a), d['A'], qn='qn0')
                X_tot = pd.concat([X_tot, X_tmp], axis=1)
        X = X_tot
        del X_tot

    elif method_rep == 'neighbours':
        X = get_mut_neigh(X, d['A'])
    elif method_rep == 'neighbours_DAD':
        X = get_mut_neigh_DAD(X, d['A'])

    elif method_rep == 'nb_mut':
        X = pd.DataFrame(M, index=M.index, columns=['nb_mut'])

    kf_out = cross_validation.StratifiedKFold(y['Vital_Status'], n_folds=5,
                                              shuffle=True,
                                              random_state=int(rs_folds))
    train_idx = []
    test_idx = []
    for train_index, test_index in kf_out:
        train_idx.append(train_index)
        test_idx.append(test_index)

    # For each fold, perform a grid search on the train set and
    # use the best estimator to train it on the whole train set. Then
    # test the obtained model on the test set.

    # Array containing the CI obtained on the train sets
    CI_train_mut = np.zeros(5)
    CI_train_clin = np.zeros(5)
    CI_train_both = np.zeros(5)
    # Array containing the CI obtained on the test sets
    CI_test_mut = np.zeros(5)
    CI_test_clin = np.zeros(5)
    CI_test_both = np.zeros(5)
    # List containing the genes with non zero coefficients for
    # each fold
    l_genes_mut = []
    l_genes_clin = []
    # List containing the best parameters obtained with the grid search
    l_best_params_mut = []
    l_best_params_clin = []
    # Array containing the best mean CI obtained over the cross-validation
    # folds
    CI_CV_mut = np.zeros(5)
    CI_CV_clin = np.zeros(5)
    # List containing the vector fitted by the survival SVM
    l_coef_mut = []
    l_coef_clin = []
    # Array containing the CI obtained on the train sets restricted to people
    # with more than the mean number of mutations
    CI_train_mut_more = np.zeros(5)
    CI_train_clin_more = np.zeros(5)
    CI_train_both_more = np.zeros(5)
    # Array containing the CI obtained on the train sets restricted to people
    # with less than the mean number of mutations
    CI_train_mut_less = np.zeros(5)
    CI_train_clin_less = np.zeros(5)
    CI_train_both_less = np.zeros(5)
    # Array containing the CI obtained on the test sets restricted to people
    # with more than the mean number of mutations
    CI_test_mut_more = np.zeros(5)
    CI_test_clin_more = np.zeros(5)
    CI_test_both_more = np.zeros(5)
    # Array containing the CI obtained on the test sets restricted to people
    # with less than the mean number of mutations
    CI_test_mut_less = np.zeros(5)
    CI_test_clin_less = np.zeros(5)
    CI_test_both_less = np.zeros(5)
    # List containing the predictions made with clinical data
    l_pred_train_clin = []
    l_pred_test_clin = []
    # List containing the predictions made with mutation data
    l_pred_train_mut = []
    l_pred_test_mut = []
    # List containing the predictions made with both mutation and clinical
    # data
    l_pred_train_both = []
    l_pred_test_both = []

    for i in range(5):
        X_train_out = X.iloc[train_idx[i], :]
        X_test_out = X.iloc[test_idx[i], :]
        clin_train_out = clin.iloc[train_idx[i], :]
        clin_test_out = clin.iloc[test_idx[i], :]
        y_train_out = y.iloc[train_idx[i], :]
        y_test_out = y.iloc[test_idx[i], :]
        M_tr = M.iloc[train_idx[i]]
        M_te = M.iloc[test_idx[i]]

        kf_in = cross_validation.StratifiedKFold(y_train_out['Vital_Status'],
                                                 n_folds=5, shuffle=True,
                                                 random_state=0)

        C = [0.0005, 0.00075, 0.001, 0.0025, 0.005]

        if data_type == 'clin' or\
           data_type == 'both':
            est_clin = grid_search.GridSearchCV(
                L1survSVM_qn3(C=C, qn='qn0'),
                param_grid={'C': C},
                scoring=make_scorer(CI_score_estC), n_jobs=n_jobs, cv=kf_in,
                refit=True)

        if data_type == 'mut' or\
           data_type == 'both':
            if method_rep == 'smoothing' or\
               method_rep == 'smoothing_1iter_DAD':
                if method_norm == 'qn':
                    est_mut = grid_search.GridSearchCV(
                        L1survSVM_qn_sm3(C=C, ind_a=ind_a, qn='qn1', q=q),
                        param_grid={'C': C, 'ind_a': ind_a},
                        scoring=make_scorer(CI_score_estC), n_jobs=n_jobs,
                        cv=kf_in, refit=True)
                if method_norm == 'none':
                    est_mut = grid_search.GridSearchCV(
                        L1survSVM_qn_sm3(C=C, ind_a=ind_a, qn='qn0', q=q),
                        param_grid={'C': C, 'ind_a': ind_a},
                        scoring=make_scorer(CI_score_estC), n_jobs=n_jobs,
                        cv=kf_in, refit=True)

            if method_rep == 'neighbours' or\
               method_rep == 'neighbours_DAD':
                if method_norm == 'stair':
                    if k_start == 'med':
                        k = [med]
                    else:
                        k = range(int(k_start), int(k_end), int(k_step))
                    est_mut = grid_search.GridSearchCV(
                        L1survSVM_stair3(C=C, k=k),
                        param_grid={'C': C, 'k': k},
                        scoring=make_scorer(CI_score_estC),
                        n_jobs=n_jobs, cv=kf_in, refit=True)

            if method_rep == 'raw' or\
               method_rep == 'nb_mut_neighbours' or\
               method_rep == 'nb_mut' or\
               method_rep == 'raw_and_nb_mut' or\
               method_rep == 'neighbours_DAD' or\
               method_rep == 'rawrestricted':
                if method_norm == 'none':
                    est_mut = grid_search.GridSearchCV(
                        L1survSVM_qn3(C=C, qn='qn0'),
                        param_grid={'C': C},
                        scoring=make_scorer(CI_score_estC), n_jobs=n_jobs,
                        cv=kf_in, refit=True)
                if method_norm == 'qn':
                    est_mut = grid_search.GridSearchCV(
                        L1survSVM_qn3(C=C, qn='qn1'),
                        param_grid={'C': C},
                        scoring=make_scorer(CI_score_estC), n_jobs=n_jobs,
                        cv=kf_in, refit=True)

        if data_type == 'clin' or\
           data_type == 'both':
            res_clin = est_clin.fit(clin_train_out, y_train_out)
            pred_test_clin = res_clin.best_estimator_.predict(clin_test_out)
            pred_train_clin = res_clin.best_estimator_.predict(clin_train_out)
            CI_test_clin[i] = CI_score_estC(y_test_out, pred_test_clin)
            CI_train_clin[i] = CI_score_estC(y_train_out, pred_train_clin)
            CI_CV_clin[i] = res_clin.best_score_
            colnames = np.array(clin_train_out.columns)
            l_genes_clin.append(
                colnames[(res_clin.best_estimator_.clf_.coef_ != 0)[0, :]])
            l_best_params_clin.append(res_clin.best_params_)
            l_coef_clin.append(res_clin.best_estimator_.clf_.coef_[0, :])
            l_pred_train_clin.append(pred_train_clin)
            l_pred_test_clin.append(pred_test_clin)

        if data_type == 'mut' or\
           data_type == 'both':
            res_mut = est_mut.fit(X_train_out, y_train_out)
            pred_test_mut = res_mut.best_estimator_.predict(X_test_out)
            pred_train_mut = res_mut.best_estimator_.predict(X_train_out)
            CI_test_mut[i] = CI_score_estC(y_test_out, pred_test_mut)
            CI_train_mut[i] = CI_score_estC(y_train_out, pred_train_mut)
            CI_CV_mut[i] = res_mut.best_score_
            colnames = np.array(X_train_out.columns)
            l_genes_mut.append(
                colnames[(res_mut.best_estimator_.clf_.coef_ != 0)[0, :]])
            l_best_params_mut.append(res_mut.best_params_)
            l_coef_mut.append(res_mut.best_estimator_.clf_.coef_[0, :])
            l_pred_train_mut.append(pred_train_mut)
            l_pred_test_mut.append(pred_test_mut)

        if data_type == 'both':
            pred_test_both = (rankdata(pred_test_mut) +
                              rankdata(pred_test_clin))
            pred_train_both = pred_train_mut + pred_train_clin
            CI_test_both[i] = CI_score_estC(y_test_out, pred_test_both)
            CI_train_both[i] = CI_score_estC(y_train_out, pred_train_both)
            l_pred_train_both.append(pred_train_both)
            l_pred_test_both.append(pred_test_both)

        #######################################################################
        # To see wether the concordance index rises because we remove mutations
        # to patients that bear a lot of mutations, or because we add mutations
        # to patients that have few, we separateley calculate the concordance
        # index for people with less/more than k mutations
        m = np.mean(M)
        y_train_out_more = y_train_out[M_tr >= m]
        y_test_out_more = y_test_out[M_te >= m]
        y_train_out_less = y_train_out[M_tr <= m]
        y_test_out_less = y_test_out[M_te <= m]

        if data_type == 'clin' or\
           data_type == 'both':
            pred_train_clin_more = pred_train_clin[(M_tr >= m).get_values()]
            pred_train_clin_less = pred_train_clin[(M_tr <= m).get_values()]
            pred_test_clin_more = pred_test_clin[(M_te >= m).get_values()]
            pred_test_clin_less = pred_test_clin[(M_te <= m).get_values()]
            CI_train_clin_more[i] = CI_score_estC(y_train_out_more,
                                                  pred_train_clin_more)
            CI_test_clin_more[i] = CI_score_estC(y_test_out_more,
                                                 pred_test_clin_more)
            CI_train_clin_less[i] = CI_score_estC(y_train_out_less,
                                                  pred_train_clin_less)
            CI_test_clin_less[i] = CI_score_estC(y_test_out_less,
                                                 pred_test_clin_less)

        if data_type == 'mut' or\
           data_type == 'both':
            pred_train_mut_more = pred_train_mut[(M_tr >= m).get_values()]
            pred_train_mut_less = pred_train_mut[(M_tr <= m).get_values()]
            pred_test_mut_more = pred_test_mut[(M_te >= m).get_values()]
            pred_test_mut_less = pred_test_mut[(M_te <= m).get_values()]
            CI_train_mut_more[i] = CI_score_estC(y_train_out_more,
                                                 pred_train_mut_more)
            CI_test_mut_more[i] = CI_score_estC(y_test_out_more,
                                                pred_test_mut_more)
            CI_train_mut_less[i] = CI_score_estC(y_train_out_less,
                                                 pred_train_mut_less)
            CI_test_mut_less[i] = CI_score_estC(y_test_out_less,
                                                pred_test_mut_less)

        if data_type == 'both':
            pred_train_both_more = pred_train_both[(M_tr >= m).get_values()]
            pred_train_both_less = pred_train_both[(M_tr <= m).get_values()]
            pred_test_both_more = pred_test_both[(M_te >= m).get_values()]
            pred_test_both_less = pred_test_both[(M_te <= m).get_values()]
            CI_train_both_more[i] = CI_score_estC(y_train_out_more,
                                                  pred_train_both_more)
            CI_test_both_more[i] = CI_score_estC(y_test_out_more,
                                                 pred_test_both_more)
            CI_train_both_less[i] = CI_score_estC(y_train_out_less,
                                                  pred_train_both_less)
            CI_test_both_less[i] = CI_score_estC(y_test_out_less,
                                                 pred_test_both_less)

    print('Doooonneee!!!')

    # Record the results in a list
    if data_type == 'clin':
        result_clin = [CI_train_clin, CI_test_clin, l_genes_clin,
                       l_best_params_clin, l_coef_clin, CI_train_clin_more,
                       CI_test_clin_more, CI_train_clin_less,
                       CI_test_clin_less, CI_CV_clin, l_pred_train_clin,
                       l_pred_test_clin]
        return result_clin

    if data_type == 'mut':
        result_mut = [CI_train_mut, CI_test_mut, l_genes_mut,
                      l_best_params_mut, l_coef_mut, CI_train_mut_more,
                      CI_test_mut_more, CI_train_mut_less, CI_test_mut_less,
                      CI_CV_mut, l_pred_train_mut, l_pred_test_mut]
        return result_mut

    if data_type == 'both':
        result_mut = [CI_train_mut, CI_test_mut, l_genes_mut,
                      l_best_params_mut, l_coef_mut, CI_train_mut_more,
                      CI_test_mut_more, CI_train_mut_less, CI_test_mut_less,
                      CI_CV_mut, l_pred_train_mut, l_pred_test_mut]
        result_clin = [CI_train_clin, CI_test_clin, l_genes_clin,
                       l_best_params_clin, l_coef_clin, CI_train_clin_more,
                       CI_test_clin_more, CI_train_clin_less,
                       CI_test_clin_less, CI_CV_clin, l_pred_train_clin,
                       l_pred_test_clin]
        result_both = [CI_train_both, CI_test_both, CI_train_both_more,
                       CI_test_both_more, CI_train_both_less,
                       CI_test_both_less, l_pred_train_both, l_pred_test_both]

        l = [result_mut, result_clin, result_both]
        return l
