import pytest
from divAtScale.src.bipartite_config.src.custom_functions import comp_gamma
import math
import numpy as np

def assert_close(expected_value, actual_value, rel_tol=1e-9, abs_tol=1e-9):
    assert math.isclose(expected_value, actual_value, rel_tol=rel_tol, abs_tol=abs_tol), \
        f"Values are not close enough: {expected_value} vs. {actual_value}"


def test_gamma_inter_aff_version():
    # test for inter-affordance comparison whereby we only consider mono-partite projection between DIFFERENT affordances
    # the matrix_cutoff signifies where A the new affordance starts for both the rows and cols
    A = np.array([[0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [1, 1, 0, 0],
                  [0, 0, 0, 0]])

    matrix_cutoff = 2
    assert_close(comp_gamma(A, matrix_cutoff=matrix_cutoff), 0.75)


retcode = pytest.main(["-x", "./"])