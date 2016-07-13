from comput_functions import get_mut_cols_as_adj_cols
from comput_functions import get_mut_neigh
from comput_functions import stair_norm


def NetNorM(X_raw, A, A_colnames, k):
    """
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
    """

    # Get the genes of the mutation matrix to be the same as those of the
    # adjacency matrix.
    # Columns corresponding to genes that exist in the mutation matrix but not
    # in the adjacency matrix are removed from the mutation matrix.
    # Columns corresponding to genes that exist in the adjacency matrix but not
    # in the mutation matrix are added to the mutation matrix as columns filled
    # with 0s.
    X = get_mut_cols_as_adj_cols(X_raw, A_colnames)

    # For each mutation profile independently, rank genes according to their
    # mutation status first, and then according to their number of neighbours
    # (mutated genes) or number of mutated neighbours (non mutated gens).
    X = get_mut_neigh(X, A)

    # Reduce (or expand) all mutation profiles to the consensus number of k
    # mutations based on the gene ranking.
    X = stair_norm(X, k)

    return X
