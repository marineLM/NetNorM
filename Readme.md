
## Overview

This repo contains python code to reproduce experiments from:

**NetNorM: capturing cancer-relevant information in somatic exome mutation data with gene networks for cancer stratification and prognosis.** 2016. Marine Le Morvan, Andrei Zinovyev, Jean-Philippe Vert. [hal-01341856](https://hal.archives-ouvertes.fr/hal-01341856)

The code is split into the following pieces:
* *NetNorM.py* :
    * compute the NetNorM representation of a mutation dataset.
* *survpred.py* :
     * reproduce the survival prediction results presented in the paper.
* *stratification_NMF.py*, *stratification_ccNMF.py* and *stratification_KMeans.py*:
     * reproduce the patient stratification results presented in the paper.
* *comput_functions.py*:
     * python module necassary for the functions introduced above.

## Packages required

The source code was written with python 2.7 and uses the following pakages:
* csv (>=1)
* numpy (>=1.8)
* pandas (>=0.18)
* scipy (>=0.17)
* sklearn  (>=0.17)  

To install these packages, you can use pip (for example: pip install numpy==1.8).
If you don't have pip installed, you can follow [these instructions](https://packaging.python.org/installing/).


```python
# Load packages
import os
import sys
import csv
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn import svm
import pickle

# Load additional functions
from comput_functions import *
```

## Datasets
### Mutation datasets
NetNorM was tested on exome somatic mutation datasets from 8 cancer types, downloaded from The Cancer Genome Atlas (TCGA) data portal. The data was preprocessed to obtain raw binary patients x genes matrices where 1s indicate non-silent point mutations or indels in a patient-gene pair and 0s indicate the absence of such mutations. The preprocessed *raw binary mutation matrices* have the following properties:  

|Cancer type | Patients | Genes | Deaths | Download date|
|:----------:|:--------:|:-----:|:------:|:------------:|
|LUAD (Lung adenocarcinoma) | 430 | 20 596 | 110 | 6/22/2015 |
|SKCM (Skin cutaneous melanoma) | 307 | 17 461 | 129 | 11/18/2015 |
|GBM (Glioblastoma multiform) | 265 | 14 748 | 195 | 11/18/2015 |
|BRCA (Breast invasive carcinoma) | 945 | 16 806 | 97 | 11/25/2015 |
|KIRC (Kidney renal clear cell carcinoma) | 411 | 10 608 | 136 | 11/25/2015 |
|HNSC (Head and Neck squamous cell carcinoma) | 388 | 17 022 | 140 | 11/25/2015 |
|LUSC (Lung squamous cell carcinoma) | 169 | 13 589 | 70 | 11/25/2015 |
|OV (Ovarian serous cystadenocarcinoma) | 363 | 10 192 | 172 | 11/24/2014 |

### Gene network
Pathway Commons (version 6) was used as gene network and downloaded in SIF format (Pathway Commons.6.All.EXTENDED_BINARY_SIF.tsv) from http://www.pathwaycommons.org/archives/PC2/v6/.

The preprocessed *raw binary mutation matrices* as well as the adjacency matrix corresponding Pathway Commons are available in csv format in *data_NetNorM.zip*. Note that the last two columns of the mutation matrices correspond to survival information (Days_survival and Vital_Status).


```python
# This script loads the LUAD mutation dataset as well as the Pathway Commons adjacency matrix.
import zipfile

zip = zipfile.ZipFile('data_NetNorM.zip')
zip.extractall()
```

## Computing the NetNorM representation of a mutation dataset

This can be done using the *NetNorM.py* function.


```python
from NetNorM import NetNorM
print NetNorM.__doc__
```

    
        Computes the NetNorM representation of a binary mutation matrix.
    
        Parameters
        ----------
        X_raw : patient x genes pandas dataframe.
            Initial binary patients x genes mutation matrix, with where 1s
            indicate non-silent point mutations or indels in a patient-gene pair
            and 0s indicate the absence of such mutations.
        A : CSR sparse matrix
            Adjacency matrix describing the gene network used.
        A_colnames: array_like
            Column names (or equivalently row names) of the adjacency matrix A.
        k : int
            Consensus number of mutations
    
        Returns
        -------
        X_NetNorM: patient x genes pandas dataframe.
            the NetNorM representation of X_raw.
        



```python
# This script computes the NetNorM representation of the LUAD mutation dataset

# Read the csv file corresponding to the LUAD dataset
X_lab = pd.read_csv('./data_NetNorM/TCGA/data_preprocessed/LUAD/X_labwithclin_ns1_NANA_NA_NA.csv',
                    sep=',', index_col=0, header=0)

# Separates mutation profiles from the clinical data
# Most of the columns of X_lab are genes, except the last ones which contain clinical data.
# The first column which contains clinical data is 'Days_survival'.
tmp = X_lab.loc[:, 'Days_survival':]
X = X_lab.drop(list(tmp.columns), axis=1)
y = tmp.loc[:, ['Days_survival', 'Vital_Status']]
clin = tmp.drop(['Days_survival', 'Vital_Status'], axis=1)

# Read the csv file corresponding to the Pathway Commons adjacency matrix.
# We directly store the adjacency matrix in a sparse format (csr) to avoid
# wasting too much memory space. However our csv_to_csr function is quite slow.
d = csv_to_csr('./data_NetNorM/Networks/data_preprocessed/A_PC', sep=';')

# Get the columns of the mutation matrix to be the same as those of the adjacency matrix
X_raw = get_mut_cols_as_adj_cols(X, d['colnames'])

# Compute the NetNorM representation of the mutation matrix.
# Choose a parameter k
k = 295
X_NetNorM = NetNorM(X_raw, d['A'], d['colnames'], k)

print type(X_NetNorM)
print X_NetNorM.shape
```

    <class 'pandas.core.frame.DataFrame'>
    (430, 16674)


## Reproducing survival prediction results

This can be done using the *survpred.py* function.  


```python
from survpred import survpred
print survpred.__doc__
```

    
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
        


Note that the arguments *method_rep* and *method_norm* control together how the mutation matrix is preprocessed. 

|method_rep | method\_norm | corresponding preprocessing|
|:---------:|:------------:|:--------------------------:|
|raw | none | none|
|neighbours | stair | NetNorM|
|smoothing | qn | NSQN|
|smoothing | none | NS|
|smoothing\_1iter\_DAD | qn | SimpNet|

The seed *rs_folds* which determines how the dataset is split into train and test sets was set to 4, 5, 6 and 7 in the paper to obtain 20 cross-validation folds (4 times 5-fold cross-validation).


```python
# This script performs survival prediction using the NetNorM
# representation of the LUAD dataset. Survival prediction is
# performed in a cross-validation setting (5-fold cross-validation).

# Define input arguments
cancer = 'LUAD'
net = 'PC'
M_file = ('./data_NetNorM/TCGA/data_preprocessed/LUAD' +
          '/X_labwithclin_ns1_NANA_NA_NA.csv')
M_sep = ','
A_file = './data_NetNorM/Networks/data_preprocessed/A_PC'
A_sep = ';'
method_rep = 'neighbours'
method_norm = 'stair'
rs_folds = '4'
randomize = False
rs_rand = 'NA'
k_start = '50'
k_step = '10'
k_end = '375'
alpha_start = 'NA'
alpha_step = 'NA'
alpha_end = 'NA'
data_type = 'mut'
n_jobs = 1

# Perform survival prediction
res_pred = survpred(M_file, M_sep, A_file, A_sep, method_rep, method_norm, rs_folds,
                    randomize, rs_rand, k_start, k_step, k_end, alpha_start,
                    alpha_step, alpha_end, data_type, n_jobs)

# Save the result as a python object in a folder called LUAD_survpred
if not os.path.isdir(cancer + '_stratification_NMF'):
    os.makedirs(cancer + '_survpred')
pickle.dump(res_pred, open(cancer + '_survpred/survpred_' + cancer + '_' +
                           str(data_type) + '_' + net + '_' + method_rep +
                           '_' + method_norm + '_rsfolds' + str(rs_folds) + '_' +
                           str(randomize) + '_rsrand' + str(rs_rand) +
                           '_' + str(k_start) + '_' +
                           str(k_step) + '_' + str(k_end) + '_' + str(alpha_start) + '_' +
                           str(alpha_step) + '_' + str(alpha_end), 'wb'))
```

## Reproducing patient stratification results

This can be done with the *stratification_NMF.py* function (or alternatively with *stratification_ccNMF* or *stratification_KMeans.py*).


```python
from stratification_NMF import stratification_NMF
print stratification_NMF.__doc__
```

    
        Preprocess the mutation data according to the method specified, and then
        performs Non-Negative Matrix Factorisation to split patients into
        subtypes.
    
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
        ass: pandas series
            The patients' cluster assignment.
        



```python
# This script performs patient stratification using the NetNorM
# representation of the LUAD dataset.


# Define input arguments
cancer = 'LUAD'
net = 'PC'
M_file = ('./data_NetNorM/TCGA/data_preprocessed/LUAD' +
          '/X_labwithclin_ns1_NANA_NA_NA.csv')
M_sep = ','
A_file = './data_NetNorM/Networks/data_preprocessed/A_PC'
A_sep = ';'
method_rep = 'neighbours'
method_norm = 'stair'
k = 295
alpha = 'NA'
randomize = False
rs_rand = 'NA'
N = 5

# Perform patient stratification
res_strat = stratification_NMF(M_file, M_sep, A_file, A_sep,
                               method_rep, method_norm, k, alpha,
                               randomize, rs_rand, N)

# Save the result as a python object in a folder called LUAD_stratification.
if not os.path.isdir(cancer + '_stratification_NMF'):
    os.makedirs(cancer + '_stratification_NMF')
pickle.dump(res_strat, open(cancer + '_stratification_NMF/stratNMF_' +
                            cancer + '_' + net + '_' + method_rep + '_' +
                            method_norm + '_k' + str(k) + '_alpha' + str(alpha) +
                            '_randomize' + str(randomize) + '_rs_rand' +
                            str(rs_rand) + '_N' + str(N), 'wb'))
```
