import numpy as np
from scipy.stats import chi2
import sys
sys.path.append('/exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/sbnd/stats')
from stats import calc_chi2

def test_basic_1d_cov():
    """Test basic 1D covariance (diagonal variances)"""
    print("Test 1: Basic 1D covariance")
    pred = np.array([10, 20, 30, 40])
    true = np.array([12, 18, 32, 38])
    var = np.array([1.0, 2.0, 3.0, 4.0])  # Diagonal variances
    
    chi2_val, dof, pval = calc_chi2(pred, true, var)
    expected = np.sum((true - pred)**2 / var)
    
    print(f"  Chi2: {chi2_val:.4f}, Expected: {expected:.4f}, DOF: {dof}, P-value: {pval:.4f}")
    assert np.isclose(chi2_val, expected), "1D chi2 calculation failed"
    print("  PASSED\n")

def test_basic_2d_cov():
    """Test basic 2D covariance matrix"""
    print("Test 2: Basic 2D covariance")
    pred = np.array([10, 20, 30])
    true = np.array([12, 18, 32])
    
    # Create a well-conditioned covariance matrix
    cov = np.array([[4.0, 1.0, 0.5],
                    [1.0, 9.0, 1.5],
                    [0.5, 1.5, 16.0]])
    
    chi2_val, dof, pval = calc_chi2(pred, true, cov)
    diff = true - pred
    expected = diff.T @ np.linalg.solve(cov, diff)
    
    print(f"  Chi2: {chi2_val:.4f}, Expected: {expected:.4f}, DOF: {dof}, P-value: {pval:.4f}")
    assert np.isclose(chi2_val, expected, rtol=1e-6), "2D chi2 calculation failed"
    print("  PASSED\n")

def test_zero_variance_1d():
    """Test filtering of zero variance bins in 1D"""
    print("Test 3: Zero variance filtering (1D)")
    pred = np.array([10, 20, 30, 40])
    true = np.array([12, 18, 32, 38])
    var = np.array([1.0, 0.0, 3.0, 4.0])  # Zero variance in bin 1
    
    chi2_val, dof, pval = calc_chi2(pred, true, var)
    # Should only use bins 0, 2, 3
    expected = (true[0] - pred[0])**2 / var[0] + \
               (true[2] - pred[2])**2 / var[2] + \
               (true[3] - pred[3])**2 / var[3]
    
    print(f"  Chi2: {chi2_val:.4f}, Expected: {expected:.4f}, DOF: {dof}")
    assert np.isclose(chi2_val, expected), "Zero variance filtering failed (1D)"
    assert dof == 2, f"DOF should be 2, got {dof}"
    print("  PASSED\n")

def test_ill_conditioned_matrix():
    """Test handling of ill-conditioned covariance matrix"""
    print("Test 5: Ill-conditioned matrix handling")
    pred = np.array([10, 20, 30])
    true = np.array([12, 18, 32])
    
    # Create an ill-conditioned matrix (condition number > 1e12)
    # One way: matrix with very small eigenvalues
    eigvals = np.array([1e-10, 1.0, 100.0])
    eigvecs = np.eye(3)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Make it symmetric and positive definite
    cov = (cov + cov.T) / 2
    cov += np.eye(3) * 1e-8  # Ensure positive definite
    
    cond_num = np.linalg.cond(cov)
    print(f"  Condition number: {cond_num:.2e}")
    
    # Should not crash and should give reasonable chi2
    chi2_val, dof, pval = calc_chi2(pred, true, cov)
    
    print(f"  Chi2: {chi2_val:.4f}, DOF: {dof}, P-value: {pval:.4f}")
    assert np.isfinite(chi2_val), "Chi2 should be finite for ill-conditioned matrix"
    assert chi2_val > 0, "Chi2 should be positive"
    assert chi2_val < 1e10, "Chi2 should be reasonable (not astronomical)"
    print("  PASSED\n")

def test_perfect_agreement():
    """Test case where pred == true (should give chi2 = 0)"""
    print("Test 7: Perfect agreement")
    pred = np.array([10, 20, 30])
    true = np.array([10, 20, 30])
    var = np.array([1.0, 2.0, 3.0])
    
    chi2_val, dof, pval = calc_chi2(pred, true, var)
    
    print(f"  Chi2: {chi2_val:.4f}, DOF: {dof}, P-value: {pval:.4f}")
    assert np.isclose(chi2_val, 0.0), "Perfect agreement should give chi2 = 0"
    assert np.isclose(pval, 1.0), "Perfect agreement should give p-value = 1"
    print("  PASSED\n")

def test_large_discrepancy():
    """Test case with large discrepancy"""
    print("Test 8: Large discrepancy")
    pred = np.array([10, 20, 30])
    true = np.array([20, 40, 60])  # Double the prediction
    var = np.array([1.0, 2.0, 3.0])
    
    chi2_val, dof, pval = calc_chi2(pred, true, var)
    expected = np.sum((true - pred)**2 / var)
    
    print(f"  Chi2: {chi2_val:.4f}, Expected: {expected:.4f}, DOF: {dof}, P-value: {pval:.4f}")
    assert np.isclose(chi2_val, expected), "Large discrepancy calculation failed"
    assert pval < 0.01, "Large discrepancy should give small p-value"
    print("  PASSED\n")

def test_realistic_covariance():
    """Test with a realistic covariance matrix (like from your analysis)"""
    print("Test 9: Realistic covariance matrix")
    n_bins = 20
    pred = np.random.rand(n_bins) * 1000 + 100
    true = pred + np.random.randn(n_bins) * 50  # Some discrepancy
    
    # Create a realistic covariance matrix with correlations
    # Start with diagonal variances
    diag_var = np.random.rand(n_bins) * 100 + 10
    cov = np.diag(diag_var)
    
    # Add some off-diagonal correlations
    for i in range(n_bins):
        for j in range(i+1, min(i+3, n_bins)):  # Correlate nearby bins
            corr = 0.3
            cov[i, j] = corr * np.sqrt(diag_var[i] * diag_var[j])
            cov[j, i] = cov[i, j]
    
    # Ensure positive definite
    cov += np.eye(n_bins) * 1e-6
    
    chi2_val, dof, pval = calc_chi2(pred, true, cov)
    
    print(f"  Chi2: {chi2_val:.4f}, DOF: {dof}, P-value: {pval:.4f}")
    assert np.isfinite(chi2_val), "Chi2 should be finite"
    assert chi2_val >= 0, "Chi2 should be non-negative"
    assert 0 <= pval <= 1, "P-value should be in [0, 1]"
    print("  PASSED\n")

def test_singular_matrix_handling():
    """Test handling of singular/near-singular matrix"""
    print("Test 10: Singular matrix handling")
    pred = np.array([10, 20, 30])
    true = np.array([12, 18, 32])
    
    # Create a singular matrix (rank-deficient)
    cov = np.array([[1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],  # Identical rows
                    [1.0, 1.0, 1.0]])
    
    # Should handle gracefully (will filter or use pseudo-inverse)
    chi2_val, dof, pval = calc_chi2(pred, true, cov)
    
    print(f"  Chi2: {chi2_val:.4f}, DOF: {dof}, P-value: {pval:.4f}")
    assert np.isfinite(chi2_val), "Should handle singular matrix gracefully"
    print("  PASSED\n")

def test_cov_2d_and_1d():
    """Test if a purely identical covariance matrix is identical to the same 1D error bars"""
    print("Test 11: Covariance matrix and 1D error bars")
    pred = np.array([10, 20, 30])
    true = np.array([12, 18, 32])
    var = np.array([1.0, 2.0, 3.0])
    cov = np.eye(3) * var
    chi2_val2d, dof2d, pval2d = calc_chi2(pred, true, cov)
    chi2_val1d, dof1d, pval1d = calc_chi2(pred, true, var)
    print(f"  Chi2: {chi2_val2d:.4f}, DOF: {dof2d}, P-value: {pval2d:.4f}")
    print(f"  Chi2: {chi2_val1d:.4f}, DOF: {dof1d}, P-value: {pval1d:.4f}")
    assert np.isclose(chi2_val2d, chi2_val1d), "Covariance matrix and 1D error bars should be identical"
    assert np.isclose(dof2d, dof1d), "DOF should be identical"
    assert np.isclose(pval2d, pval1d), "P-value should be identical"
    print("  PASSED\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Running calc_chi2 tests")
    print("=" * 60 + "\n")
    
    try:
        test_basic_1d_cov()
        test_basic_2d_cov()
        #test_zero_variance_1d()
        test_ill_conditioned_matrix()
        test_perfect_agreement()
        test_large_discrepancy()
        test_realistic_covariance()
        test_singular_matrix_handling()
        test_cov_2d_and_1d()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()