import numpy as np
from scipy.stats import chi2

def construct_covariance(cv_evts, var_evts, scale_cov=1., rank=None, do_correct_negative_eigenvalues=False, assert_cov=True, apply_shrinkage=False):
  """
  Construct covariance matrix for a given set of event counts.
  Returns both corrected (optional rank and negative-eigenvalue correction) and
  truly unaltered (no rank, no neg correction) covariances.

  Parameters
  ----------
  cv_evts : numpy array
    Event counts for each bin in cv universe
  var_evts : array_like
    Event counts for each bin in each variation universe
  scale_cov : float
    Overall scale factor applied to the covariance matrix (default: 1.)
  rank : int, optional
    Number of largest eigenvalues to keep; if set, corrected cov is rank-reduced.
    Rank -1 means to make the covariance diagonal
  do_correct_negative_eigenvalues : bool, optional
    If True, correct negative eigenvalues on the corrected covariance (default False).
  assert_cov : bool, optional
    If True, raise on NaN/Inf in fractional cov; else mask to 0.
  apply_shrinkage : bool, optional
    If True, apply shrinkage to the covariance matrix.
  Returns
  -------
  covariance_matrix : numpy array
    Corrected covariance (rank and/or neg-eig correction if requested).
  cov_unaltered : numpy array
    Raw covariance: no rank, no negative-eigenvalue correction.
  fractional_cov : numpy array
    Fractional cov used for corrected (rank-reduced if rank set).
  fractional_cov_unaltered : numpy array
    Raw fractional cov (no rank).
  correlation : numpy array
    Correlation matrix from corrected cov.
  fractional_uncertainty : numpy array
    Fractional uncertainty from corrected cov.
  iters : int or None
    Iteration count from correct_negative_eigenvalues if used, else None.
  """
  #if apply_shrinkage and rank is not None:
  #  print('Warning: apply_shrinkage and rank are both set, ignoring rank reduction')
  cv = np.asarray(cv_evts, dtype=float)
  vars_stack = np.asarray(var_evts, dtype=float)

  if vars_stack.ndim == 1:
    vars_stack = vars_stack[np.newaxis, :]

  if vars_stack.shape[1] != cv.shape[0]:
    raise ValueError(f"Universe histograms (shape {vars_stack.shape}) do not match CV bins ({cv.shape}).")

  n_universes = vars_stack.shape[0]
  if n_universes == 0:
    raise ValueError("Need at least one universe histogram to build a covariance matrix.")

  diff = vars_stack - cv
  denom = cv
  diff_fractional = diff / denom

  # print(f'diff: {diff}')
  # print(f'diff.shape: {diff.shape}')
  # print(f'n_universes: {n_universes}')

  fractional_cov = (diff_fractional.T @ diff_fractional) / n_universes
  cov_unaltered = (diff.T @ diff) / n_universes
  if rank == -1:
    fractional_cov = np.eye(fractional_cov.shape[0]) * np.diag(fractional_cov)
    cov_unaltered = np.eye(cov_unaltered.shape[0]) * np.diag(cov_unaltered)

  if np.isnan(fractional_cov).any():
    if assert_cov:
      raise ValueError(f"Covariance matrix contains NaN values: {fractional_cov}")
    else:
      fractional_cov = np.nan_to_num(fractional_cov)
  if np.isinf(fractional_cov).any():
    if assert_cov:
      raise ValueError(f"Covariance matrix contains infinite values: {fractional_cov}")
    else:
      fractional_cov = np.where(np.isinf(fractional_cov), 0, fractional_cov)
  if (np.abs(fractional_cov) == np.finfo(np.double).max).any():
    if assert_cov:
      raise ValueError(f"Covariance matrix contains values equal to the maximum finite value: {fractional_cov}")
    else:
      fractional_cov = np.where(np.abs(fractional_cov) == np.finfo(np.double).max, 0, fractional_cov)

  # Unaltered: no rank, no neg correction
  fractional_cov_unaltered = fractional_cov.copy()
  covariance_matrix = cov_unaltered.copy()
  # Optional rank on fractional, then build absolute, then optional neg-eig correction
  # if rank is not None and rank > 0 and rank < fractional_cov.shape[0] and not apply_shrinkage:
  #   fractional_cov = keep_top_eigenvalues_cov(fractional_cov, rank)
  # if apply_shrinkage and rank is not None:
  #   fractional_cov, rho, phi = shrink_covariance_oasd(fractional_cov, n=rank)
  # eigvals, eigvecs = np.linalg.eigh(fractional_cov)
  # eigval_ratios = eigvals / np.max(eigvals)
  # tol_ratio = 1e-12
  # if (eigval_ratios < -tol_ratio).any():
  #   raise ValueError(f"Fractional covariance has significantly negative eigenvalues: {eigval_ratios[eigval_ratios < -tol_ratio]}")

  # if do_correct_negative_eigenvalues:
  #   covariance_matrix, iters = correct_negative_eigenvalues(covariance_matrix)
  # else:
  #   iters = None
  iters = None

  correlation, stds = construct_correlation_matrix(covariance_matrix)
  fractional_uncertainty = np.where(cv > 0, stds / cv, 0.)

  return covariance_matrix, cov_unaltered, fractional_cov, fractional_cov_unaltered, correlation, fractional_uncertainty, iters

def covariance_from_fraccov(fraccov, cv):
  """
  Convert a fractional covariance to an absolute covariance for a given mean.

  Use this when combining covariances built from different scales (e.g. cosmic_data
  vs selected). The relation is cov_ij = fraccov_ij * cv_i * cv_j.

  Parameters
  ----------
  fraccov : numpy array
    Fractional covariance matrix (scale-invariant)
  cv : numpy array
   CV vector for the target scale

  Returns
  -------
  cov : numpy array
    Absolute covariance on the given mean scale
  """
  cv = np.asarray(cv, dtype=float)
  fraccov = np.nan_to_num(np.asarray(fraccov, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
  return fraccov * np.outer(cv, cv)

def construct_correlation_matrix(covariance_matrix):
  """
  Construct a correlation matrix from a covariance matrix
  """
  stds = np.sqrt(np.diag(covariance_matrix))
  correlation = np.zeros_like(covariance_matrix)
  for i in range(len(stds)):
    for j in range(len(stds)):
      correlation[i,j] = covariance_matrix[i,j] / (stds[i] * stds[j])
  return correlation,stds

def get_fractional_uncertainty(cv,covariance_matrix):
  """
  Get the fractional uncertainty from a covariance matrix and CV
  """
  stds = np.sqrt(np.diag(covariance_matrix))
  fractional_uncertainty = stds / cv
  #Set nan values to 0
  fractional_uncertainty = np.nan_to_num(fractional_uncertainty)
  #Set infinite values to 0
  fractional_uncertainty = np.where(np.isinf(fractional_uncertainty), 0, fractional_uncertainty)
  return fractional_uncertainty

def get_total_unc(cv,unc,allow_negative=False):
  """
  Get the total uncertainty from the fractional uncertainty and CV.
  We weight each fractional uncertainty by the CV value.

  Parameters
  ----------
  cv : array-like
    The CV values
  unc : array-like
    The fractional uncertainty values
  allow_negative : bool
    Whether to allow the total variance to be negative
  Returns
  -------
  total_uncertainty : float
    The total uncertainty
  """
  if not allow_negative:
    if np.any(unc < 0):
      raise ValueError('Fractional uncertainty is negative')
  cv_sum = np.nansum(cv) if hasattr(cv, '__len__') else cv
  total_uncertainty = np.sqrt(np.nansum(unc**2 * cv/cv_sum))
  return total_uncertainty

def calc_mean_hist(hist_vals, hist_edges):
    # Calculate bin midpoints
    bin_midpoints = (hist_edges[:-1] + hist_edges[1:]) / 2
    # Calculate mean
    mean = np.average(bin_midpoints, weights=hist_vals)
    return mean

def calc_mode_hist(hist_vals, hist_edges):
    # Find the index of the bin with the highest value (mode)
    mode_index = np.argmax(hist_vals)
    # Calculate the midpoint of the mode bin
    mode = (hist_edges[mode_index] + hist_edges[mode_index + 1]) / 2
    return mode

def build_matrix(pred_labels, true_labels, n_classes = 5):
    assert pred_labels.shape == true_labels.shape
    hist = np.zeros((n_classes,n_classes))
    for i,t in enumerate(true_labels): #Get true labels
        if t == 5: continue #skip kaons for now
        p = pred_labels[i] #Get associated predicted label
        if p == -1 or t == -1 or np.isnan(p) or np.isnan(t): continue #Dummy labels
        hist[int(p),int(t)] += 1
    return hist

def get_smear_matrix(true_var, reco_var, bins, weights=None):
    """
    Compute smearing matrix from true vs reco 2D histogram.
    
    Parameters
    ----------
    true_var : array-like
        True variable values
    reco_var : array-like
        Reconstructed variable values
    bins : array-like
        Bin edges
    weights : array-like, optional
        Weights for histogram
        
    Returns
    -------
    reco_vs_true : ndarray
        Smearing matrix (reco_vs_true.T is the response matrix shape)
    """
    reco_vs_true, _, _ = np.histogram2d(true_var, reco_var, bins=bins, weights=weights)
    return reco_vs_true


def compute_efficiency(sig_truth, sel_truth):
    """
    Compute efficiency from truth histograms.
    
    Parameters
    ----------
    sig_truth : array-like
        Signal truth histogram
    sel_truth : array-like
        Selected truth histogram
        
    Returns
    -------
    eff_truth : ndarray
        Efficiency histogram
    """
    # Efficiency = selected_truth / signal_truth
    eff_truth = np.where(sig_truth > 0, sel_truth / sig_truth, 0.0)
    # Handle division by zero
    eff_truth = np.where(sig_truth > 0, eff_truth, 0.0)
    return eff_truth


def compute_sigma_tilde(response, sig_truth, sel_background_reco, xsec_unit):
    """
    Compute sigma_tilde from response matrix.
    
    Parameters
    ----------
    response : ndarray
        Response matrix
    sig_truth : array-like
        Signal distribution in truth space
    sel_background_reco : array-like
        Selected background in reco space
    xsec_unit : float
        Cross section unit conversion factor
        
    Returns
    -------
    sigma_tilde : ndarray
        Sigma tilde histogram
    """
    sigma_tilde = xsec_unit * (
        response @ sig_truth + sel_background_reco
    )
    return sigma_tilde


def convert_smearing_to_response(smearing,eff):
  """
  Convert a smearing matrix to a response matrix by convolving 
  with the efficiency array and normalizing by the truth column

  Parameters
  ----------
  smearing : array-like (N,N)
    The smearing matrix
  eff : array-like (N,)
    The efficiency array

  Returns
  -------
  response : array-like (N,N)
    The response matrix
  """
  denom = np.sum(smearing.T,axis=0)
  response = np.divide(smearing.T*eff,denom,out=np.zeros_like(smearing,dtype=float),where=denom!=0)
  return response

def shrink_covariance_oasd(S, n=None, mean_known=False):
    """
    Oracle Approximating Shrinkage with Diagonal target (OASD) estimator.
    
    Shrinks a sample covariance matrix toward its diagonal to improve estimation
    in high-dimensional settings (p > n). This method is particularly useful when
    the diagonal elements of the true covariance matrix exhibit substantial variation.
    
    Based on: "Oracle Approximating Shrinkage with Diagonal Target" paper.
    https://www.elibrary.imf.org/view/journals/001/2023/257/article-A001-en.xml#:~:text=We%20use%20a%20simulation%20to,Sch%C3%A4fer%20and%20Strimmer%20(2005).

    Parameters
    ----------
    S : array_like, shape (p, p)
        Sample covariance matrix
    n : int, optional
        Sample size. If None, inferred from S (assumes n-1 degrees of freedom)
    mean_known : bool, default False
        Whether the mean is known. If True, uses n+1 in denominator (Theorem 3).
        If False, uses n in denominator (Theorem 2, general case).
    
    Returns
    -------
    S_shrunk : ndarray, shape (p, p)
        Shrunk covariance matrix: (1 - rho)*S + rho*diag(S)
    rho : float
        Shrinkage parameter (between 0 and 1)
    phi : float
        The phi parameter used in the calculation
    
    Notes
    -----
    The shrinkage estimator is:
        S_OASD = (1 - rho)*S + rho*diag(S)
    
    where rho is computed as:
        rho_OASD = min(1/(n*phi), 1)  [mean unknown, general case]
        rho_OASD = min(1/((n+1)*phi), 1)  [mean known, special case]
    
    and phi is:
        phi = (tr(S^2) - tr(diag(S)^2)) / (tr(S^2) + tr(S)^2 - 2*tr(diag(S)^2))
    
    The resulting matrix is guaranteed to be positive definite.
    """
    S = np.asarray(S, dtype=float)
    
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square matrix")
    
    p = S.shape[0]
    
    # Infer n if not provided (assumes n-1 degrees of freedom in sample covariance)
    if n is None:
        # This is a heuristic - in practice, n should be provided
        # We can't reliably infer it from S alone
        raise ValueError("n (sample size) must be provided")
    
    # Compute diagonal matrix
    diag_S = np.diag(np.diag(S))
    
    # Compute traces needed for phi
    tr_S2 = np.trace(S @ S)  # tr(S^2)
    tr_S = np.trace(S)  # tr(S)
    tr_diag_S2 = np.trace(diag_S @ diag_S)  # tr(diag(S)^2)
    
    # Compute phi parameter (equation 7)
    numerator = tr_S2 - tr_diag_S2
    denominator = tr_S2 + tr_S**2 - 2*tr_diag_S2
    
    if denominator == 0:
        # Edge case: if denominator is zero, set phi to a small value
        phi = 1e-10
    else:
        phi = numerator / denominator
    
    # Ensure phi is in valid range [0, 1)
    phi = np.clip(phi, 0.0, 1.0 - 1e-10)
    
    # Compute shrinkage parameter rho (equation 11 for general case, 16 for known mean)
    if mean_known:
        rho = min(1.0 / ((n + 1) * phi), 1.0)
    else:
        rho = min(1.0 / (n * phi), 1.0)
    
    # Ensure rho is in (0, 1]
    rho = np.clip(rho, 1e-10, 1.0)
    
    # Apply shrinkage (equation 12 or 17)
    S_shrunk = (1.0 - rho) * S + rho * diag_S
    
    # Ensure result is symmetric (should be by construction, but just in case)
    S_shrunk = (S_shrunk + S_shrunk.T) / 2.0
    
    return S_shrunk, rho, phi

def _cov_eigenvalue_diagnostics(cov):
    """Eigenvalues and condition number for a covariance matrix. Use to check ill-conditioning."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)
    #sort eigvecs by eigvals
    eigvecs = eigvecs[:, np.argsort(eigvals)]
    max_eig, min_eig = eigvals[-1], eigvals[0]
    cond = max_eig / min_eig if min_eig > 0 else np.inf
    return eigvals, eigvecs, cond, min_eig, max_eig


def keep_top_eigenvalues_cov(cov, k, tol=1e-10, max_iter=100, diagnose=False):
    """
    Build a reduced covariance matrix using only the k largest eigenvalues (and their
    eigenvectors). Small-eigenvalue modes are zeroed so the inverse is stable.
    Optionally refines iteratively.

    Parameters
    ----------
    cov : array_like, shape (n, n)
        Full covariance matrix.
    k : int
        Number of largest eigenvalues to keep (1 <= k <= n).
    tol : float, optional
        Relative tolerance: among the top k eigenvalues, any below tol * max(kept)
        are set to zero. Use 0 or None to keep all k. Default 1e-12.
    max_iter : int, optional
        Max refinement iterations: rebuild, re-eig, re-apply top-k and tol, repeat
        until cov_reduced stabilizes or max_iter. Default 1 (no refinement).
    diagnose : bool, default False
        If True, print eigenvalue/condition diagnostics for full cov.

    Returns
    -------
    cov_reduced : np.ndarray, shape (n, n)
        Reduced covariance: same shape as cov, only top eigenmodes above tol contribute.
        Rank <= k. Use calc_chi2(..., cov_reduced) with pinv (default) for singular cov.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if k < 1 or k > n:
        raise ValueError(f'k must be in [1, n]; got k={k}, n={n}')
    if tol is not None and tol < 0:
        raise ValueError('tol must be >= 0 or None')
    tol_use = float(tol) if tol is not None else 0.0

    cov_reduced = cov.copy()
    for it in range(max_iter):
        eigvals, eigvecs = np.linalg.eigh(cov_reduced)
        idx = np.argsort(eigvals)[::-1]
        dropped_eigvals = eigvals[idx[k:]]
        top_eigvals = eigvals[idx[:k]].copy()
        top_eigvecs = eigvecs[:, idx[:k]]

        # Zero out kept eigenvalues below relative tolerance
        if tol_use > 0:
            max_kept = np.max(top_eigvals)
            if max_kept > 0:
                top_eigvals = np.where(top_eigvals >= tol_use * max_kept, top_eigvals, 0.0)
        
        cov_new = (top_eigvecs * top_eigvals) @ top_eigvecs.T
        cov_new = 0.5 * (cov_new + cov_new.T)
        if it > 0 and np.allclose(cov_reduced, cov_new, rtol=tol, atol=tol):
            cov_reduced = cov_new
            break
        elif it == max_iter - 1:
            eigvals, eigvecs, cond, min_eig, max_eig = _cov_eigenvalue_diagnostics(cov_new) 
            print(f'eigvals: {eigvals}')
            print(f'eigvecs: {eigvecs}')
            print(f'cond: {cond}')
            print(f'min_eig: {min_eig}')
            print(f'max_eig: {max_eig}')
            raise Exception(f"Failed to reduce covariance matrix to rank {k} in {max_iter} iterations")

        cov_reduced = cov_new

    if diagnose:
        eigvals_final, _ = np.linalg.eigh(cov_reduced)
        nz = np.sum(eigvals_final > tol_use * np.max(eigvals_final))
        print(f'  effective rank: {nz}')
        print(f'  effective pinv_rcond: {np.max(dropped_eigvals)/np.max(top_eigvals):.2e}')
        print(f'  dropped eigvalues: {dropped_eigvals}')
        print(f'  kept eigvalues (after tol): {top_eigvals}')
        print(f'  max deviation from input cov: {np.max(np.abs(cov_reduced - cov)):.2e}')
    return cov_reduced

def correct_negative_eigenvalues(cov, scale_from_max=0., max_iter=1000, tol=0.):
    """
    Iteratively replace negative eigenvalues with scale_from_max * max_positive_eigenvalue,
    reconstruct the matrix, and repeat until all eigenvalues are positive (or max_iter).
    """
    cov_corrected = np.asarray(cov, dtype=float).copy()
    iters = 0
    for _ in range(max_iter):
        eigvals, eigvecs = np.linalg.eigh(cov_corrected)
        negative_eigs = eigvals < tol
        if not np.any(negative_eigs):
            break
        iters += 1
        positive_eigs = eigvals[~negative_eigs]
        max_pos = np.max(positive_eigs) if len(positive_eigs) > 0 else np.abs(np.min(eigvals)) * 1e-10
        replacement = max(scale_from_max * max_pos, tol)
        eigvals[negative_eigs] = replacement
        cov_corrected = (eigvecs * eigvals) @ eigvecs.T
    return cov_corrected, iters

def calc_chi2(pred, true, cov, n=None, apply_shrinkage=False, pinv_rcond=1e-12, diagnose=False, filter_min=-np.inf, rank=None):
    """
    Calculate the chi2 between two arrays using a covariance matrix.

    If full-cov chi2 is huge (e.g. 100k) while diagonal chi2 is modest (e.g. 90), the
    covariance is ill-conditioned: tiny eigenvalues become huge in inv(cov) and amplify
    residual components in those directions. Use pinv_rcond to regularize (pseudo-inverse
    zeros out modes with eigenvalue < rcond * max_eigenvalue).

    Parameters
    ----------
    pred : array_like
        Predicted values
    true : array_like
        True/observed values
    cov : array_like
        Covariance matrix (2D) or diagonal variances (1D)
    n : int, optional
        Sample size. If None and shrinkage is needed, inferred from sum of `true`
        (assumes `true` represents event counts in histogram bins).
    apply_shrinkage : bool, default True
        If True, automatically apply OASD shrinkage if the covariance matrix
        has negative eigenvalues or is ill-conditioned.
    pinv_rcond : float or None, default 1e-6
        If not None, use pseudo-inverse of cov with this rcond (relative to max
        singular value) instead of raw inverse. Stabilizes chi2 when cov is
        ill-conditioned. Set to None to use np.linalg.inv(cov) (old behavior).
    diagnose : bool, default False
        If True, print eigenvalue/condition diagnostics for full cov.
    filter_min : float, default -np.inf
        Filter bins with pred or true values less than filter_min.
    rank : int, optional
        Rank of the covariance matrix. If None, use the full rank.

    Returns
    -------
    chi2 : float
        Chi-squared value
    dof : int
        Degrees of freedom
    pval : float
        P-value
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    cov = np.asarray(cov)
    
    assert len(pred) == len(true), f'Pred and true must have the same length: {len(pred)} != {len(true)}'
    
    # Filter bins with pred or true values less than filter_min
    valid = (pred >= filter_min) & (true >= filter_min)
    pred = pred[valid]
    true = true[valid]
    
    diff = true - pred
    dof = len(pred) - 1
    
    if cov.ndim == 1:
        # Diagonal covariance (just variances)
        cov = cov[valid]
        assert len(cov) == len(pred), f'Cov must have the same length as pred: {len(cov)} != {len(pred)}'

        chi2_vals = diff**2 / cov
        # print(f'chi2_vals = {chi2_vals}')
        # print(f'cov = {cov}')
        # print(f'diff = {diff}')
        # print(f'diff**2 = {diff**2}')
        chi2_val = np.nansum(chi2_vals)
        pval = chi2.sf(chi2_val, dof)
        return chi2_val, dof, pval
    else:
        cov = cov[valid][:,valid]
        # Full covariance matrix
        assert cov.shape[0] == cov.shape[1], f'Cov must be a square matrix: {cov.shape}'
        assert cov.shape[0] == len(pred), f'Cov must have the same number of rows as pred: {cov.shape[0]} != {len(pred)}'
        
        if rank is not None:
            cov = keep_top_eigenvalues_cov(cov, rank)
        if diagnose:
            eigvals, eigvecs, cond, min_eig, max_eig = _cov_eigenvalue_diagnostics(cov)
            print(f'cov eigenvalues: min={min_eig:.6e} max={max_eig:.6e} cond={cond:.6e}')
            for i in range(len(eigvals)):
                print(f'    eigvalue {i}: {eigvals[i]:.2e}')
                #print(f'    eigvector {i}: {eigvecs[:,i]}')
            if pinv_rcond is not None:
                cutoff = pinv_rcond * max_eig
                n_dropped = np.sum(eigvals < cutoff)
                print(f'  pinv_rcond={pinv_rcond} -> drop eig < {cutoff:.6e} ({n_dropped} modes)')

        if pinv_rcond is not None:
            inv_cov = np.linalg.pinv(cov, rcond=pinv_rcond)
        else:
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                print('WARNING - Failed to invert covariance matrix, using pseudo-inverse')
                inv_cov = np.linalg.pinv(cov, rcond=pinv_rcond/10.)
        # print(f'inv_cov = {inv_cov}')
        # print(f'inv_cov diagonal: {np.diag(inv_cov)}')
        # print(f'diff**2 = {diff**2}')
        # print(f'diff = {diff}')
        # print(f'inv_cov @ diff**2 = {inv_cov @ diff**2}')
        chi2_val = diff.T @ inv_cov @ diff
        #print(f'chi2_vals = {chi2_vals}')
        #print(f'inv_cov = {inv_cov}')
        #print(f'diff = {diff}')
        pval = chi2.sf(chi2_val, dof)
        return chi2_val, dof, pval