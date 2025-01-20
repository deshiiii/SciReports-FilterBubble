import pytest
from divAtScale.src.helpers.semantic_helpers.generate_svd_space import SVD_builder as svd_builder
import numpy as np
from scipy.sparse import csr_matrix
import math

def assert_close(expected_value, actual_value, rel_tol=1e-9, abs_tol=1e-9):
    assert math.isclose(expected_value, actual_value, rel_tol=rel_tol, abs_tol=abs_tol), \
        f"Values are not close enough: {expected_value} vs. {actual_value}"


def test_ppmi():
    X = np.random.randint(low=0, high=10, size=(10, 10))
    X = np.triu(X) + np.triu(X, k=1).T
    X2 = csr_matrix(X)

    ppmi_M = svd_builder("","").ppmi(X2).tocsr()

    test1 = max(np.log2((X[0, 4] / X.sum()) / ((sum(X[0]) / X.sum()) * (sum(X[4]) / X.sum()))), 0)
    assert_close(ppmi_M[0, 4], test1)

    test2 = max(np.log2((X[4, 0] / X.sum()) / ((sum(X[4]) / X.sum()) * (sum(X[0]) / X.sum()))), 0)
    assert_close(ppmi_M[4, 0], test2)



retcode = pytest.main(["-x", "./"])