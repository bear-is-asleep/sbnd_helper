"""
Compute responses between reco and true bins
"""
from tqdm import tqdm
import numpy as np

def calc_response_efficiency(ngens,nsels,smearing_matrix,sel_bin):
    """
    Calculate effective responses from smearing matrix and bins
    based on https://github.com/sjgardiner/stv-analysis-new/blob/eb182611dd92496e39d2ecd770056a73eab6e4b0/MCC8ForwardFolder.hh#L62
    
    Parameters:
    - ngens (np.array): total signal events (N_b)
    - nsels (np.array): selected signal events (N_b)
    - smearing_matrix (np.array): smearing matrix (N_b,N_b)
    
    Returns:
    - eff (float): efficiency
    """
    #assert len(reco_bins) == len(true_bins), 'reco and true bins must be the same length'
    assert len(ngens) == len(nsels), 'ngen and nsel must be the same length'
    num,den = 0.,0.
    for i,ngen in enumerate(ngens):
        nsel = nsels[i]
        num += smearing_matrix[sel_bin][i] * nsel
        den += smearing_matrix[sel_bin][i] * ngen
    return num/den

# def build_response_efficiency_table(partgrp,cuts,xbinning_key,ybinning_key,xtrue_binning_key,ytrue_binning_key,xbins,ybins,x_key,y_key,true_x_key,true_y_key):
#     """
#     Calculate efficiency matrix for non uniform binnings. 
#     Assume xbins and ybins are unstructured 2d arrays that we will bin over
#     Used for double differential xsec.
    
#     Parameters:
#     - partgrp (ParticleGroup): contains dataframe to split
#     - cuts (list): list of cuts to apply
#     - xbinning_key (str): x variable binning key
#     - ybinning_key (str): y variable binning key
#     - xtrue_binning_key (str): true x variable binning key
#     - ytrue_binning_key (str): true y variable binning key
#     - xbins (list of lists): x bins, not necessarily rectangular
#     - ybins (list of lists): y bins, not necessarily rectangular
#     - x_key (str): reco x variable key
#     - y_key (str): reco y variable key
#     - true_x_key (str): true x variable key
#     - true_y_key (str): true y variable key
    
#     Returns:
#     - effs (np.array): efficiencies same shape as xbins and ybins
#     """
#     #Validity checks
#     assert len(xbins) == len(ybins), 'xbins and ybins must be the same length'
#     #for _xbins,_ybins in zip(xbins,ybins):
#     #    assert len(_xbins) == len(_ybins), 'xbins and ybins must be the same length for each row'
#     #Initialize column names
#     xbinning_col = partgrp.get_key(xbinning_key)
#     ybinning_col = partgrp.get_key(ybinning_key)
#     xtrue_binning_col = partgrp.get_key(xtrue_binning_key)
#     ytrue_binning_col = partgrp.get_key(ytrue_binning_key)
    
#     effs = [None]*len(xbins)
#     for i,_xbins in enumerate(_xbins):
#         _ybins = ybins[i]
#         #Assign binnings
#         _partgrp = partgrp.copy()
#         _partgrp.assign_bins(_xbins,x_key,assign_key=xbinning_key)
#         _partgrp.assign_bins(_ybins,y_key,assign_key=ybinning_key)
#         _partgrp.assign_bins(_xbins,true_x_key,assign_key=xtrue_binning_key)
#         _partgrp.assign_bins(_ybins,true_y_key,assign_key=ytrue_binning_key)
#         #Calculate smearing matrices
#         xsmearing_matrix = calc_smearing_matrix(_partgrp.data[xbinning_col],_partgrp.data[xtrue_binning_col])
#         ysmearing_matrix = calc_smearing_matrix(_partgrp.data[ybinning_col],_partgrp.data[ytrue_binning_col])
#         #Get number of generated events
#         _partgrp_numucc = _partgrp.copy()
#         _partgrp_numucc.data = _partgrp_numucc.data[(_partgrp_numucc.data.truth.event_type == 1).values]
#         xngen = _partgrp_numucc.get_binned_numevents(xtrue_binning_key)
#         yngen = _partgrp_numucc.get_binned_numevents(ytrue_binning_key)
#         #Apply cuts and get number of generated events
#         _partgrp_numucc_cut = _partgrp_numucc.copy()
#         xnsel = _partgrp_numucc_cut.get_binned_numevents(xtrue_binning_key)
#         ynsel = _partgrp_numucc_cut.get_binned_numevents(ytrue_binning_key)
#         #Calculate efficiencies
#         xeffs = np.zeros(len(_xbins))
#         yeffs = np.zeros(len(_ybins))
#         for j in range(len(_xbins)):
#             xeffs[j] = calc_response_efficiency(xngen,xnsel,xsmearing_matrix,j)
        
        
        
        

def build_response_efficiency_array(partgrp,cuts,binning_key,true_binning_key,smearing_matrix):
    """
    Calculate efficiency for each reco bins
    Assume binning has already been done
    Build this to precompute and potentially vectorize operations
    
    Parameters:
    - partgrp (ParticleGroup): contains dataframe to split
    - cuts (list): list of cuts to apply
    - binning_key (str): reco binning key
    - true_binning_key (str): true binning key
    - smearing_matrix (np.array): smearing matrix (N_b,N_b)
    
    Returns:
    - effs (np.array): efficiencies (N_b)
    """
    assert partgrp.check_key(binning_key), 'binning key not found'
    assert partgrp.check_key(true_binning_key), 'true binning key not found'
    
    reco_binning = np.arange(0,max(partgrp.data[binning_key])+1,1)
    true_binning = np.arange(0,max(partgrp.data[true_binning_key])+1,1)
    assert len(reco_binning) == len(true_binning), 'reco and true bins must be the same length'
    
    effs = np.zeros(len(reco_binning))
    for i in tqdm(range(len(reco_binning))):
        #Filter to reco bin
        _partgrp = partgrp.copy()
        _partgrp.data = _partgrp.data[(_partgrp.data[binning_key] == i).values]
        
        #Extract ngen and nsel
        ngen = _partgrp.get_binned_numevents(true_binning_key)
        partgrp_cut = _partgrp.copy()
        for cut in cuts:
            partgrp_cut.apply_cut(cut)
        nsel = partgrp_cut.get_binned_numevents(true_binning_key)
        effs[i] = calc_response_efficiency(ngen,nsel,smearing_matrix,i)
    return effs #efficiency of each reco bin

def calc_smearing_matrix(reco_bins,true_bins,weights=None,normalize=True):
    """
    Calculate smearing matrix given list of reco_bins and true_bins
    based on https://github.com/sjgardiner/stv-analysis-new/blob/eb182611dd92496e39d2ecd770056a73eab6e4b0/internal_note/smear_matrix.C#L102
    
    Parameters:
    - reco_bins (np.array): reco bins (N)
    - true_bins (np.array): true bins (N)
    - weights (np.array): weights for each event (N)
    - normalize (bool): normalize by reco bins
    
    Returns:
    - smearing_matrix (np.array): smearing matrix (N_b,N_b)
    """
    assert len(reco_bins) == len(true_bins), 'reco and true bins must be the same length'
    tnbins = max(true_bins)
    rnbins = max(reco_bins)
    assert tnbins == rnbins, 'number of reco and true bins must be the same'
    if weights is None:
        weights = np.ones(reco_bins.shape)
    bins = np.arange(0,rnbins+2,1)
    smearing_matrix = np.histogram2d(reco_bins,true_bins,bins=bins,weights=weights)[0]
    #Normalize by reco bins, or row
    if normalize:
        smearing_matrix = np.array([row/row.sum() for row in smearing_matrix])
    
    return smearing_matrix
    
    

    
    
    