"""
Unit tests for XSec.unfold() using Wiener-SVD unfolding.

Run as a script (python test_xsec.py) or via pytest.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[2]))  # .../numuincl  -> sbnd.* imports
sys.path.insert(0, str(_here.parents[4]))  # .../cafpyana  -> analysis_village.* imports

from sbnd.cafclasses.xsec import XSec
from sbnd.stats import stats
from sbnd.general import plotters

PLOT_DIR = '/exp/sbnd/data/users/brindenc/analyze_sbnd/numu/v10_06_00_validation/pandora/plots/tests'
os.makedirs(PLOT_DIR, exist_ok=True)

N = 5
BINS_EVEN   = np.linspace(-1.0, 1.0, N + 1)
BINS_UNEVEN = np.array([-1.0, -0.5, 0.0, 0.3, 0.7, 1.0])
BIN_CENTERS_EVEN   = (BINS_EVEN[:-1]   + BINS_EVEN[1:])   / 2
BIN_CENTERS_UNEVEN = (BINS_UNEVEN[:-1] + BINS_UNEVEN[1:]) / 2

SIG_TRUTH  = np.array([1000., 2000., 3000., 4000., 5000.])
BACKGROUND = np.zeros(N)
RESPONSE_I = np.eye(N)

def _poisson_fraccov(sel):
    """Diagonal fractional covariance from Poisson counting: fraccov[i,i] = 1/sel[i]."""
    return np.diag(1.0 / np.asarray(sel))

UNFOLD_KW = dict(fractional_cov=True, C_type=2, Norm_type=0.5)

# One shared XSec object per binning; all tests call unfold() with distinct names
costheta        = XSec(xsec_unit=1, bins=BINS_EVEN,   name='costheta',        variable='costheta')
costheta_uneven = XSec(xsec_unit=1, bins=BINS_UNEVEN, name='costheta_uneven', variable='costheta')


def _event_arrays(bin_centers, counts):
    """Per-event arrays for a perfectly diagonal smearing (true == reco == bin center)."""
    counts   = np.asarray(counts, dtype=int)
    per_event = np.repeat(bin_centers, counts)
    return per_event, per_event.copy(), np.ones(len(per_event))


# ---------------------------------------------------------------------------
# Test 1: even bins, identity response provided, sel == sig_truth (perfect)
# ---------------------------------------------------------------------------
def test_1_perfect_even():
    stat_cov = _poisson_fraccov(SIG_TRUTH)
    sigma_tilde = stats.compute_sigma_tilde(RESPONSE_I, SIG_TRUTH, BACKGROUND, xsec_unit=1)
    costheta.unfold(
        stat_cov, SIG_TRUTH, SIG_TRUTH,
        response=RESPONSE_I,
        sigma_tilde=sigma_tilde,
        name='perfect_even',
        stat_cov=stat_cov,
        **UNFOLD_KW,
    )
    chi2 = costheta.unfold_results['perfect_even']['chi2']
    assert np.isclose(chi2, 0.0, atol=1e-4), f'Test 1 chi2 expected ~0, got {chi2}'
    fig, ax = costheta.plot_unfold(['perfect_even'], ['Perfect Even'],
                                   title='Test 1: Perfect Even Bins')
    plotters.save_plot('test1_perfect_even', fig=fig, folder_name=PLOT_DIR)
    print(f'Test 1 passed  chi2={chi2:.2e}')


# ---------------------------------------------------------------------------
# Test 2: uneven bins, identity response provided, sel == sig_truth (perfect)
# ---------------------------------------------------------------------------
def test_2_perfect_uneven():
    stat_cov = _poisson_fraccov(SIG_TRUTH)
    sigma_tilde = stats.compute_sigma_tilde(RESPONSE_I, SIG_TRUTH, BACKGROUND, xsec_unit=1)
    costheta_uneven.unfold(
        stat_cov, SIG_TRUTH, SIG_TRUTH,
        response=RESPONSE_I,
        sigma_tilde=sigma_tilde,
        name='perfect_uneven',
        stat_cov=stat_cov,
        **UNFOLD_KW,
    )
    chi2 = costheta_uneven.unfold_results['perfect_uneven']['chi2']
    assert np.isclose(chi2, 0.0, atol=1e-4), f'Test 2 chi2 expected ~0, got {chi2}'
    fig, ax = costheta_uneven.plot_unfold(['perfect_uneven'], ['Perfect Uneven'],
                                          title='Test 2: Perfect Uneven Bins')
    plotters.save_plot('test2_perfect_uneven', fig=fig, folder_name=PLOT_DIR)
    print(f'Test 2 passed  chi2={chi2:.2e}')


# ---------------------------------------------------------------------------
# Test 3: build identity response from event arrays, full efficiency
# ---------------------------------------------------------------------------
def test_3_build_response_even():
    stat_cov = _poisson_fraccov(SIG_TRUTH)
    sel_sig_truth, sel_sig_reco, weights = _event_arrays(BIN_CENTERS_EVEN, SIG_TRUTH)
    costheta.unfold(
        stat_cov, SIG_TRUTH, SIG_TRUTH,
        sel_background=BACKGROUND,
        sel_truth=SIG_TRUTH,
        sel_sig_reco=sel_sig_reco,
        sel_sig_truth=sel_sig_truth,
        genweights_sel_sig=weights,
        name='build_response_even',
        stat_cov=stat_cov,
        **UNFOLD_KW,
    )
    chi2 = costheta.unfold_results['build_response_even']['chi2']
    assert np.isclose(chi2, 0.0, atol=1e-4), f'Test 3 chi2 expected ~0, got {chi2}'
    fig, ax = costheta.plot_unfold(['build_response_even'], ['Build Response Even'],
                                   title='Test 3: Build Identity Response')
    plotters.save_plot('test3_build_response_even', fig=fig, folder_name=PLOT_DIR)
    print(f'Test 3 passed  chi2={chi2:.2e}')


# ---------------------------------------------------------------------------
# Test 4: build response, selected signal is 1/2 of signal (eff = 0.5)
# ---------------------------------------------------------------------------
def test_4_half_sel_even():
    half_counts = (SIG_TRUTH / 2).astype(int)
    sel_sig_truth, sel_sig_reco, weights = _event_arrays(BIN_CENTERS_EVEN, half_counts)
    sel = SIG_TRUTH * 0.5
    stat_cov = _poisson_fraccov(sel)
    costheta.unfold(
        stat_cov, sel, SIG_TRUTH,
        sel_background=BACKGROUND,
        sel_truth=sel,
        sel_sig_reco=sel_sig_reco,
        sel_sig_truth=sel_sig_truth,
        genweights_sel_sig=weights,
        name='half_sel_even',
        stat_cov=stat_cov,
        **UNFOLD_KW,
    )
    chi2 = costheta.unfold_results['half_sel_even']['chi2']
    assert np.isclose(chi2, 0.0, atol=1e-4), f'Test 4 chi2 expected ~0, got {chi2}'
    fig, ax = costheta.plot_unfold(['half_sel_even'], ['Half Sel Even'],
                                   title='Test 4: Half Selection Even')
    plotters.save_plot('test4_half_sel_even', fig=fig, folder_name=PLOT_DIR)
    print(f'Test 4 passed  chi2={chi2:.2e}')


# ---------------------------------------------------------------------------
# Test 5: same as 4 but uneven bins
# ---------------------------------------------------------------------------
def test_5_half_sel_uneven():
    half_counts = (SIG_TRUTH / 2).astype(int)
    sel_sig_truth, sel_sig_reco, weights = _event_arrays(BIN_CENTERS_UNEVEN, half_counts)
    sel = SIG_TRUTH * 0.5
    stat_cov = _poisson_fraccov(sel)
    costheta_uneven.unfold(
        stat_cov, sel, SIG_TRUTH,
        sel_background=BACKGROUND,
        sel_truth=sel,
        sel_sig_reco=sel_sig_reco,
        sel_sig_truth=sel_sig_truth,
        genweights_sel_sig=weights,
        name='half_sel_uneven',
        stat_cov=stat_cov,
        **UNFOLD_KW,
    )
    chi2 = costheta_uneven.unfold_results['half_sel_uneven']['chi2']
    assert np.isclose(chi2, 0.0, atol=1e-4), f'Test 5 chi2 expected ~0, got {chi2}'
    fig, ax = costheta_uneven.plot_unfold(['half_sel_uneven'], ['Half Sel Uneven'],
                                          title='Test 5: Half Selection Uneven')
    plotters.save_plot('test5_half_sel_uneven', fig=fig, folder_name=PLOT_DIR)
    print(f'Test 5 passed  chi2={chi2:.2e}')


# ---------------------------------------------------------------------------
# Test 6: same as 4 but sel is raised above sim selected. cov = stat + 10% syst (60% correlated)
# ---------------------------------------------------------------------------

# Systematic fraccov: 10% unc, 70% off-diagonal correlation
_SYST_FRACCOV = 0.01 * (0.4 * np.eye(N) + 0.6 * np.ones((N, N)))

def test_6_fluctuated_sel_even():
    half_counts = (SIG_TRUTH / 2).astype(int)
    sel_sig_truth, sel_sig_reco, weights = _event_arrays(BIN_CENTERS_EVEN, half_counts)
    rng = np.random.default_rng(40)
    #sel = SIG_TRUTH * 0.5 * (1.0 + rng.uniform(-0.1, 0.1, N))
    sel = SIG_TRUTH * 0.7
    #sel = np.array([SIG_TRUTH[i] * (0.7 - 0.1*i) * (1.0 + rng.uniform(-0.1, 0.1)) for i in range(N)])
    stat_cov = _poisson_fraccov(sel)
    total_cov = stat_cov + _SYST_FRACCOV
    costheta.unfold(
        total_cov, sel, SIG_TRUTH,
        sel_background=BACKGROUND,
        sel_truth=SIG_TRUTH * 0.5,
        sel_sig_reco=sel_sig_reco,
        sel_sig_truth=sel_sig_truth,
        genweights_sel_sig=weights,
        name='fluctuated_even',
        stat_cov=stat_cov,
        verbose=True,
        **UNFOLD_KW,
    )
    print('Frac unc: ', costheta.unfold_results['fluctuated_even']['fracunc'])
    chi2 = costheta.unfold_results['fluctuated_even']['chi2']
    assert chi2 > 0, f'Test 6 chi2 expected > 0 for fluctuated sel, got {chi2}'

    fig, ax = costheta.plot_unfold(['fluctuated_even'], ['Fluctuated Even'],
                                   title='Test 6: Fluctuated Sel')
    plotters.save_plot('test6_fluctuated_even', fig=fig, folder_name=PLOT_DIR)

    unfold_cov = costheta.unfold_results['fluctuated_even']['unfold_cov']
    fig_cov, ax_cov = plt.subplots()
    im = ax_cov.pcolormesh(BINS_EVEN, BINS_EVEN, unfold_cov)
    fig_cov.colorbar(im, ax=ax_cov)
    ax_cov.set_xlabel(costheta.regularized_label)
    ax_cov.set_ylabel(costheta.regularized_label)
    ax_cov.set_title('Test 6: Unfolded Covariance Matrix')
    plotters.save_plot('test6_unfold_cov', fig=fig_cov, folder_name=PLOT_DIR)

    fig_resp, ax_resp = costheta.plot_response('fluctuated_even', title='Test 6: Response Matrix')
    plotters.save_plot('test6_response', fig=fig_resp, folder_name=PLOT_DIR)

    fig_res, ax_res = costheta.plot_residuals(['fluctuated_even'], ['Fluctuated Even'],
                                              title='Test 6: Normalized Residuals')
    plotters.save_plot('test6_residuals', fig=fig_res, folder_name=PLOT_DIR)

    # Plot fraccov
    fig, ax = plt.subplots()
    im = ax.pcolormesh(BINS_EVEN, BINS_EVEN, costheta.unfold_results['fluctuated_even']['unfold_fraccov'])
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(costheta.regularized_label)
    ax.set_ylabel(costheta.regularized_label)
    ax.set_title('Test 6: Systematic Fractional Covariance Matrix')
    plotters.save_plot('test6_fraccov', fig=fig, folder_name=PLOT_DIR)

    # Plot correlation
    fig, ax = plt.subplots()
    im = ax.pcolormesh(BINS_EVEN, BINS_EVEN, costheta.unfold_results['fluctuated_even']['unfold_corr'],
        vmin=0, vmax=1, cmap='Greens')
    print('Correlation matrix: ', costheta.unfold_results['fluctuated_even']['unfold_corr'])
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(costheta.regularized_label)
    ax.set_ylabel(costheta.regularized_label)
    ax.set_title('Test 6: Systematic Correlation Matrix')
    plotters.save_plot('test6_corr', fig=fig, folder_name=PLOT_DIR)

    print(f'Test 6 passed  chi2={chi2:.4f}')


# ---------------------------------------------------------------------------
# Test 7: same setup as test 6 (decreasing sel pattern, stat + 10% syst cov).
#         Separate unfold key and plot filenames from test 6.
# ---------------------------------------------------------------------------

def test_7_decreasing_sel_fluctuated_even():
    half_counts = (SIG_TRUTH / 2).astype(int)
    sel_sig_truth, sel_sig_reco, weights = _event_arrays(BIN_CENTERS_EVEN, half_counts)
    rng = np.random.default_rng(42)
    #sel = SIG_TRUTH * 0.5 * (1.0 + rng.uniform(-0.1, 0.1, N))
    sel = np.array([SIG_TRUTH[i] * (0.7 - 0.1*i) for i in range(N)])
    stat_cov = _poisson_fraccov(sel)
    total_cov = stat_cov + _SYST_FRACCOV
    unfold_key = 'decreasing_sel_fluctuated_even'
    costheta.unfold(
        total_cov, sel, SIG_TRUTH,
        sel_background=BACKGROUND,
        sel_truth=SIG_TRUTH * 0.5,
        sel_sig_reco=sel_sig_reco,
        sel_sig_truth=sel_sig_truth,
        genweights_sel_sig=weights,
        name=unfold_key,
        stat_cov=stat_cov,
        verbose=True,
        **UNFOLD_KW,
    )
    rec = costheta.unfold_results[unfold_key]
    print('Frac unc: ', rec['fracunc'])
    chi2 = rec['chi2']
    assert chi2 > 0, f'Test 7 chi2 expected > 0 for fluctuated sel, got {chi2}'

    fig, ax = costheta.plot_unfold([unfold_key], ['Decreasing fluctuated sel'],
                                   title='Test 7: Decreasing fluctuated sel')
    plotters.save_plot('test7_decreasing_sel_fluctuated_even', fig=fig, folder_name=PLOT_DIR)

    unfold_cov = rec['unfold_cov']
    fig_cov, ax_cov = plt.subplots()
    im = ax_cov.pcolormesh(BINS_EVEN, BINS_EVEN, unfold_cov)
    fig_cov.colorbar(im, ax=ax_cov)
    ax_cov.set_xlabel(costheta.regularized_label)
    ax_cov.set_ylabel(costheta.regularized_label)
    ax_cov.set_title('Test 7: Unfolded covariance')
    plotters.save_plot('test7_unfold_cov', fig=fig_cov, folder_name=PLOT_DIR)

    fig_resp, ax_resp = costheta.plot_response(unfold_key, title='Test 7: Response matrix')
    plotters.save_plot('test7_response', fig=fig_resp, folder_name=PLOT_DIR)

    fig_res, ax_res = costheta.plot_residuals([unfold_key], ['Decreasing fluctuated sel'],
                                              title='Test 7: Normalized residuals')
    plotters.save_plot('test7_residuals', fig=fig_res, folder_name=PLOT_DIR)

    # Plot fraccov
    fig, ax = plt.subplots()
    im = ax.pcolormesh(BINS_EVEN, BINS_EVEN, rec['unfold_fraccov'])
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(costheta.regularized_label)
    ax.set_ylabel(costheta.regularized_label)
    ax.set_title('Test 7: Unfolded fractional covariance')
    plotters.save_plot('test7_fraccov', fig=fig, folder_name=PLOT_DIR)

    # Plot correlation
    fig, ax = plt.subplots()
    im = ax.pcolormesh(BINS_EVEN, BINS_EVEN, rec['unfold_corr'],
        vmin=0, vmax=1, cmap='Greens')
    print('Correlation matrix: ', rec['unfold_corr'])
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(costheta.regularized_label)
    ax.set_ylabel(costheta.regularized_label)
    ax.set_title('Test 7: Unfolded correlation')
    plotters.save_plot('test7_corr', fig=fig, folder_name=PLOT_DIR)

    print(f'Test 7 passed  chi2={chi2:.4f}')

if __name__ == '__main__':
    test_1_perfect_even()
    test_2_perfect_uneven()
    test_3_build_response_even()
    test_4_half_sel_even()
    test_5_half_sel_uneven()
    test_6_fluctuated_sel_even()
    test_7_decreasing_sel_fluctuated_even()
    print('\nAll tests passed.')
