"""
Compute the total xsec, single dxsec, and double dxsec
"""
from ..detector.definitions import *
from ..flux.constants import *
import numpy as np
from tqdm import tqdm
from . import response

def get_d2xsec_dxdy_dict(partgrp, x_key, y_key, true_x_key, true_y_key, xbins, ybins, xbinning_key, ybinning_key, true_xbinning_key,true_ybinning_key, xsmearing_matrix, ysmearing_matrix,cuts,nt=NUMBER_TARGETS_FV,phi=NUMU_INTEGRATED_FLUX,
                         verbose=False):
    """
    Calculate the double differential cross section for particle groups within specified bins.

    Parameters:
    - partgrp (ParticleGroup): The particle group instance containing event data.
    - x_key (str): The key for the x variable in the data.
    - y_key (str): The key for the y variable in the data.
    - true_x_key (str): The key for the true x variable in the data.
    - true_y_key (str): The key for the true y variable in the data.
    - xbins (list): The bin edges for the x variable.
    - ybins (list): The bin edges for the y variable. If 1D, it will be expanded to match xbins length.
    - xbinning_key (str): The key used to assign x binning in the data.
    - ybinning_key (str): The key used to assign y binning in the data.
    - true_xbinning_key (str): The key used to assign true x binning in the data.
    - true_ybinning_key (str): The key used to assign true y binning in the data.
    - xsmearing_matrix (2D array): The smearing matrix for unfolding the measured to the true distribution, for x key
    - ysmearing_matrix (2D array): The smearing matrix for unfolding the measured to the true distribution, for y key
    - cuts (list): List of cuts to apply to the data for event selection.
    - nt (float): Number of scattering targets
    - phi (float): Integrated flux (integrated over face surface area)
    - verbose (bool): Whether to print out progress information.

    Returns:
    - d2x_dxdy_dict (dict): A dictionary with keys as bin index tuples and values as dictionaries containing bin information and cross section data.
    """
    if verbose:
        print(f'x_key: {x_key}')
        print(f'y_key: {y_key}')
        print(f'true_x_key: {true_x_key}')
        print(f'true_y_key: {true_y_key}')
        print(f'xbins: {xbins}')
        print(f'ybins: {ybins}')
        print(f'xbinning_key: {xbinning_key}')
        print(f'ybinning_key: {ybinning_key}')
        print(f'true_xbinning_key: {true_xbinning_key}')
        print(f'true_ybinning_key: {true_ybinning_key}')
        print(f'cuts: {cuts}')
        print(f'nt: {nt}')
        print(f'phi: {phi}')
        


    #Convert bins to multiindex format
    #x_col = partgrp.get_key(x_key)
    #y_col = partgrp.get_key(y_key)
    xbinning_col = partgrp.get_key(xbinning_key)
    ybinning_col = partgrp.get_key(ybinning_key)

    #xbins = xbins
    #ybins = ybins
    d2x_dxdy_dict = {} #Store info per bin

    # Fill out ybins to 2D binning if only one dimension provided
    #if len(np.shape(ybins)) == 1:
    #    ybins = [ybins] * len(xbins)

    #TODO: Allow for different x binnings
    # Assign xbinning 
    if not partgrp.check_key(xbinning_key): #only assign binning if not already done
        partgrp.assign_bins(xbins, x_key, assign_key=xbinning_key)
        partgrp.assign_bins(xbins, true_x_key, assign_key=true_xbinning_key)

    for i, x in enumerate(xbins[:-1]):
        _ybins = ybins[i]
        print(f'xbin ({x},{xbins[i+1]}]')
        #Make copy and filter to xbin
        _pgrp = partgrp.copy()
        _pgrp.data = _pgrp.data[(_pgrp.data[xbinning_col] == i).values]
        _pgrp.assign_bins(_ybins, y_key,assign_key=ybinning_key)
        _pgrp.assign_bins(_ybins, true_y_key, assign_key=ybinning_key)
        for j, y in enumerate(_ybins[:-1]):
            print(f'ybin ({y},{_ybins[j+1]}]')
            dx = xbins[i+1] - x
            dy = _ybins[j+1] - y

            #Get events in bin
            _pgrp_y = _pgrp.copy()
            _pgrp_y.data = _pgrp_y.data[(_pgrp_y.data[ybinning_col] == j).values]
            
            # Get efficiency and number of generated events
            pur, eff, _ = _pgrp_y.get_pur_eff_f1(cuts)
            ngen_x = _pgrp_y.get_binned_numevents(true_xbinning_key)
            ngen_y = _pgrp_y.get_binned_numevents(true_ybinning_key)

            # Get n_i, b_i
            _pgrp_cut = _pgrp_y.copy()
            for cut in cuts:
                _pgrp_cut.apply_cut(cut)
            n_i = _pgrp_cut.data.genweight.sum() #total events
            b_i = _pgrp_cut.data.genweight[_pgrp_cut.data.truth.event_type != 0].sum()
            
            # Get number of selected events in truth bins
            nsel_x = _pgrp_y.get_binned_numevents(xbinning_key)
            nsel_y = _pgrp_y.get_binned_numevents(ybinning_key)
            
            #Get MC response efficiency
            eff_x = response.calc_response_efficiency(ngen_x,nsel_x,xsmearing_matrix,sel_bin=i)
            eff_y = response.calc_response_efficiency(ngen_y,nsel_y,ysmearing_matrix,sel_bin=j)

            # Get d2dx_dxdy
            d2x_dxdy = (n_i - b_i) / (dx * dy * eff_x * eff_y * nt * phi)

            # Get unc
            stat_unc = 1 / np.sqrt(n_i + b_i)
            syst_unc = 0. #TODO: this

            # Store results in dictionary
            d2x_dxdy_dict[(i, j)] = {
                'dx': dx,
                'dy': dy,
                'eff': eff[-1],
                'pur': pur[-1],
                'eff_x': eff_x,
                'eff_y': eff_y,
                'n_i': n_i,
                'b_i': b_i,
                'phi': phi,
                'nt': nt,
                'stat_unc': stat_unc,
                'syst_unc': syst_unc,
                'd2x_dxdy': d2x_dxdy,
                'ybins': _ybins,
                'xbins': xbins,
            }
    #Store info about process
    meta_dict = {
        'cuts': cuts,
        'phi': phi,
        'nt': nt,
        'x_key': x_key,
        'y_key': y_key,
        'xbinning_key': xbinning_key,
        'ybinning_key': ybinning_key,
        'xsmearing_matrix': xsmearing_matrix,
        'ysmearing_matrix': ysmearing_matrix,
    }

    return d2x_dxdy_dict,meta_dict

