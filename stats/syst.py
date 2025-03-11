import numpy as np

def get_weighted_hist(var,weights,bins):
    """
    Calculate histograms for the given variable and weights, and compute the
    central value (CV) histogram along with the upper and lower quantile histograms.

    Parameters
    ----------
    var : array_like
        Input array of values to be binned.
    weights : array_like
        2D array where each row contains a set of weights corresponding to `var`.
    bins : array_like
        Array of bin edges.

    Returns
    -------
    edges : ndarray
        The bin edges, with one more edge than there are bins.
    nom_hist : ndarray
        The nominal histogram computed without weights.
    cv_hist : ndarray
        The central value histogram computed as the mean of the weighted histograms.
    upper_hist : ndarray
        The upper quantile histogram.
    lower_hist : ndarray
        The lower quantile histogram.
    """
    #Get histograms
    nom_hist, edges = np.histogram(var,bins=bins)
    if weights is None:
        return edges, nom_hist
    weighted_hists = np.zeros((weights.shape[0],len(nom_hist)))
    for i,w in enumerate(weights):
        weighted_hists[i], _ = np.histogram(var,bins=bins,weights=w) #weights not supported by numba
    #Get CV hist and errors
    cv_hist = np.mean(weighted_hists,axis=0)
    upper_hist = abs(np.quantile(weighted_hists,0.68,axis=0)-cv_hist) #1 sigma
    lower_hist = abs(np.quantile(weighted_hists,0.32,axis=0)-cv_hist) #-1 sigma
    return edges, nom_hist, cv_hist, upper_hist, lower_hist, weighted_hists

#TODO: implement ability to reference a different CAF
def combine_weighted_hists(caf,weight_names,var_key,bins,reference_caf=None):
    """
    Combine weighted histograms from different weights and universes to get a single histogram with errors.
    
    Parameters
    ----------
    caf : CAF
        The CAF object containing the weights.
    weight_names : list
        List of weight names to combine.
    var_key : str
        Key of the variable to get histogram of.
    bins : array_like   
        Array of bin edges.
    reference_caf : CAF, optional   
        Reference CAF object to get the variable data from. Default is None.
    
    Returns
    -------
    edges : ndarray
        The bin edges, with one more edge than there are bins.
    cv_hist : ndarray
        The central value histogram computed as the mean of the weighted histograms.
    upper_hist : ndarray
        The upper quantile histogram.
    lower_hist : ndarray
        The lower quantile histogram.
    weighted_hist : ndarray
        The weighted histograms for each universe.
    
    """
    if reference_caf is None:
        reference_caf = caf.copy()
    else:
        raise NotImplementedError("Referencing a different CAF is not yet implemented.")
    #Initialize histogram
    weighted_hists = [None]*len(weight_names)
    #Get binned variable data
    var_col = reference_caf.get_key(var_key) #convert to tuple format
    var = reference_caf.data[var_col].values.flatten()
    #Get reweighted distributions
    for i,weight_name in enumerate(weight_names):
        weights = caf.data[weight_name].values.T
        edges, _, _, _, _, weighted_hist = get_weighted_hist(var,weights,bins)
        weighted_hists[i] = weighted_hist
    #Get summary statistics
    weighted_hist = np.mean(np.array(weighted_hists),axis=1) #average along all universes
    cv_hist = np.mean(weighted_hist,axis=0) #average along all weights
    upper_hist = abs(np.quantile(weighted_hist,0.68,axis=0)-cv_hist) #1 sigma
    lower_hist = abs(np.quantile(weighted_hist,0.32,axis=0)-cv_hist) #-1 sigma
    
    return edges,cv_hist,upper_hist,lower_hist,weighted_hist