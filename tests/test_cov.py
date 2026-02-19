import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'stats'))
from stats import construct_covariance

def compute_expected_covariance(cv, var, frac=False):
    """Compute the expected covariance matrix."""
    diff = var - cv
    expected_cov = np.zeros((len(cv), len(cv)))
    for n in range(len(var)):
        for i in range(len(cv)):
            for j in range(len(cv)):
                if frac:
                    expected_cov[i,j] += diff[n,i] * diff[n,j] / cv[i] / cv[j]
                else:
                    expected_cov[i,j] += diff[n,i] * diff[n,j]
    return expected_cov / len(var)

def test_basic_single_bin():
    """CV and one variation, single bin: fractional_cov is 0, cov is 0."""
    print("Testing basic single bin...")
    cv = np.array([100.0])
    var = np.array([[100.0]])
    cov_mat, cov_unaltered, frac_cov, frac_cov_unaltered, corr, frac_unc, iters = construct_covariance(cv, var)
    expected_cov = compute_expected_covariance(cv, var, frac=False)
    expected_frac_cov = compute_expected_covariance(cv, var, frac=True)
    assert np.allclose(cov_mat, expected_cov), f'Covariance should be {expected_cov}: {cov_mat}'
    assert np.allclose(frac_cov, expected_frac_cov), f'Fractional covariance should be {expected_frac_cov}: {frac_cov}'
    assert np.allclose(cov_unaltered, cov_mat), "Unaltered should match when no corrections"
    assert np.allclose(frac_cov_unaltered, frac_cov), "Fractional unaltered should match when no corrections"
    assert iters is None
    assert np.allclose(frac_unc, np.zeros(1)), f'Fractional uncertainty should be 0: {frac_unc}'
    print('  ...passed')

def test_basic_multiple_bins_multiple_universes():
    """Multiple bins, multiple universes: check shapes and symmetry."""
    print("Testing basic multiple bins multiple universes...")
    np.random.seed(42)
    n_bins = 5
    n_uni = 20
    cv = np.array([100.0, 200.0, 150.0, 80.0, 120.0])
    var = cv + np.random.randn(n_uni, n_bins) * 10.0
    cov_mat, cov_unaltered, frac_cov, frac_cov_unaltered, corr, frac_unc, iters = construct_covariance(cv, var)
    expected_cov = compute_expected_covariance(cv, var, frac=False)
    expected_frac_cov = compute_expected_covariance(cv, var, frac=True)
    assert cov_mat.shape == (n_bins, n_bins), f'Covariance shape should be ({n_bins}, {n_bins}): {cov_mat.shape}'
    assert frac_cov.shape == (n_bins, n_bins), f'Fractional covariance shape should be ({n_bins}, {n_bins}): {frac_cov.shape}'
    assert corr.shape == (n_bins, n_bins), f'Correlation shape should be ({n_bins}, {n_bins}): {corr.shape}'
    assert frac_unc.shape == (n_bins,), f'Fractional uncertainty shape should be ({n_bins},): {frac_unc.shape}'
    assert np.allclose(cov_mat, cov_mat.T), "Covariance should be symmetric"
    assert np.allclose(frac_cov, frac_cov.T), "Fractional cov should be symmetric"
    assert np.allclose(np.diag(corr), np.ones(n_bins)), "Correlation diagonal should be 1"
    assert np.all(frac_unc >= 0), "Fractional uncertainty should be non-negative"
    assert np.allclose(cov_mat, expected_cov), f'Covariance should be {expected_cov}: {cov_mat}'
    assert np.allclose(frac_cov, expected_frac_cov), f'Fractional covariance should be {expected_frac_cov}: {frac_cov}'
    assert np.allclose(cov_unaltered, cov_mat), "Unaltered should match when no corrections"
    assert np.allclose(frac_cov_unaltered, frac_cov), "Fractional unaltered should match when no corrections"
    assert iters is None
    print('  ...passed')

def test_var_evts_1d_promoted_to_2d():
    """Single universe passed as 1D array is treated as (1, n_bins)."""
    print("Testing var evts 1d promoted to 2d...")
    cv = np.array([10.0, 20.0])
    var_1d = np.array([11.0, 19.0])
    cov_mat, cov_unaltered, frac_cov, frac_cov_unaltered, corr, frac_unc, iters = construct_covariance(cv, var_1d)
    #expected_cov = compute_expected_covariance(cv, var_1d, frac=False)
    #expected_frac_cov = compute_expected_covariance(cv, var_1d, frac=True)
    assert cov_mat.shape == (2, 2)
    assert frac_cov.shape == (2, 2)
    #assert np.allclose(cov_mat, expected_cov), f'Covariance should be {expected_cov}: {cov_mat}'
    #assert np.allclose(frac_cov, expected_frac_cov), f'Fractional covariance should be {expected_frac_cov}: {frac_cov}'
    print('  ...passed')


def test_scale_cov():
    """scale_cov scales the covariance matrix by scale_cov**2."""
    print("Testing scale cov...")
    cv = np.array([100.0, 200.0])
    var = np.array([[105.0, 195.0], [95.0, 205.0]])
    expected_cov = compute_expected_covariance(cv, var, frac=False)
    expected_frac_cov = compute_expected_covariance(cv, var, frac=True)
    cov1, _, _, _, _, _, _ = construct_covariance(cv, var, scale_cov=1.0)
    cov2, _, _, _, _, _, _ = construct_covariance(cv, var, scale_cov=2.0)
    assert np.allclose(cov2, 4.0 * cov1), "scale_cov=2 should give 4x covariance"
    print('  ...passed')


def test_shape_mismatch_raises():
    """Mismatch between CV length and variation bin count raises ValueError."""
    print("Testing shape mismatch raises...")
    cv = np.array([10.0, 20.0, 30.0])
    var = np.array([[1.0, 2.0]])  # 2 bins vs 3
    try:
        construct_covariance(cv, var)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "do not match" in str(e) or "shape" in str(e).lower()
    print('  ...passed')


def test_zero_universes_raises():
    """Empty var_evts (0 universes) raises ValueError."""
    print("Testing zero universes raises...")
    cv = np.array([10.0, 20.0])
    var = np.array([]).reshape(0, 2)
    try:
        construct_covariance(cv, var)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "at least one universe" in str(e).lower() or "one universe" in str(e).lower()
    print('  ...passed')


def test_zero_cv_bin_nan_raises_with_assert_cov():
    """Zero in cv_evts causes inf/NaN fractional cov; with assert_cov=True we get ValueError (NaN branch)."""
    print("Testing zero cv bin nan raises with assert cov...")
    cv = np.array([100.0, 0.0, 50.0])
    var = np.array([[100.0, 1.0, 50.0], [100.0, 2.0, 50.0]])
    try:
        construct_covariance(cv, var, assert_cov=True)
        assert False, "Should have raised ValueError (NaN or inf)"
    except ValueError:
        pass
    print('  ...passed')

def test_zero_cv_bin_masked_with_assert_cov_false():
    """Zero in cv_evts with assert_cov=False: NaN/inf are masked, may still fail on negative eigenvalues."""
    print("Testing zero cv bin masked with assert cov false...")
    cv = np.array([100.0, 1.0, 50.0])  # avoid exact 0 so we don't hit inf; use tiny to get big fractional diff
    var = np.array([[100.0, 1.0, 50.0]] * 5)  # no variance in universes
    cov_mat, cov_unaltered, frac_cov, frac_cov_unaltered, corr, frac_unc, iters = construct_covariance(cv, var, assert_cov=False)
    expected_cov = compute_expected_covariance(cv, var, frac=False)
    expected_frac_cov = compute_expected_covariance(cv, var, frac=True)
    assert not np.isnan(cov_mat).any()
    assert not np.isinf(cov_mat).any()
    assert np.allclose(cov_mat, expected_cov), f'Covariance should be {expected_cov}: {cov_mat}'
    assert np.allclose(frac_cov, expected_frac_cov), f'Fractional covariance should be {expected_frac_cov}: {frac_cov}'
    print('  ...passed')

def test_fractional_cov_formula():
    """Check fractional_cov = (diff_fractional.T @ diff_fractional) / n_universes and cov = frac_cov * outer(cv,cv)."""
    print("Testing fractional cov formula...")
    cv = np.array([1.0, 2.0])
    var = np.array([[2.0, 1.0], [2.0, 1.0]])
    cov_mat, cov_unaltered, frac_cov, frac_cov_unaltered, _, frac_unc, _ = construct_covariance(cv, var, scale_cov=1.0)
    expected_frac_cov = compute_expected_covariance(cv, var, frac=True)
    assert np.allclose(frac_cov, expected_frac_cov), f'Fractional covariance should be {expected_frac_cov}: {frac_cov}'
    expected_cov = compute_expected_covariance(cv, var, frac=False)
    print(f'Expected covariance: {expected_cov}')
    assert np.allclose(cov_mat, expected_cov), f'Covariance should be {expected_cov}: {cov_mat}'
    print('  ...passed')

if __name__ == "__main__":
    test_basic_single_bin()
    test_basic_multiple_bins_multiple_universes()
    test_var_evts_1d_promoted_to_2d()
    test_scale_cov()
    test_shape_mismatch_raises()
    test_zero_universes_raises()
    test_zero_cv_bin_nan_raises_with_assert_cov()
    test_zero_cv_bin_masked_with_assert_cov_false()
    test_fractional_cov_formula()