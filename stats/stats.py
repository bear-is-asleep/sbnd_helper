import numpy as np

def construct_covariance(cv_evts,var_evts,scale_cov=1.):
  """
  Construct covariance matrix for a given set of event counts
  
  Parameters
  ----------
  cv_evts : numpy array
    Event counts for each bin in cv universe
  var_evts : array_like
    Event counts for each bin in each variation universe
  scale_cov : float
    Overall scale factor applied to the covariance matrix (default: 1.)
  Returns
  -------
  cov : numpy array
    Covariance matrix
  """
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
  denom = cv + 1e-10
  diff_fractional = diff / denom

  fractional_cov = diff_fractional.T @ diff_fractional / n_universes
  #fractional_cov = 0.5 * (fractional_cov + fractional_cov.T)

  if np.isnan(fractional_cov).any():
    raise ValueError("Covariance matrix contains NaN values.")
  if np.isinf(fractional_cov).any():
    raise ValueError("Covariance matrix contains infinite values.")

  eigvals, eigvecs = np.linalg.eigh(fractional_cov)
  eigval_ratios = eigvals / np.max(eigvals)
  tol_ratio = 1e-6
  if (eigval_ratios < -tol_ratio).any():
    raise ValueError(f"Fractional covariance has significantly negative eigenvalues: {eigval_ratios[eigval_ratios < -tol_ratio]}")
  eigvals = np.clip(eigvals, 0.0, None)
  fractional_cov = (eigvecs * eigvals) @ eigvecs.T

  covariance_matrix = fractional_cov * np.outer(cv, cv) * (scale_cov ** 2)

  return covariance_matrix

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

def calc_chi2(pred, true, cov):
    """
    Calculate the chi2 between two arrays using a covariance matrix.
    
    Parameters
    ----------
    pred : array_like
        Predicted values
    true : array_like  
        True/observed values
    cov : array_like
        Covariance matrix (2D) or diagonal variances (1D)
    
    Returns
    -------
    chi2 : float
        Chi-squared value
    dof : int
        Degrees of freedom
    """
    diff = pred - true
    dof = len(pred) - 1
    if cov.ndim == 1:
        # Diagonal covariance (just variances)
        return np.sum(diff**2 / cov), dof
    else:
        # Full covariance matrix
        inv_cov = np.linalg.inv(cov)
        chi2 = diff @ inv_cov @ diff
        return chi2, dof