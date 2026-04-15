import numpy as np
import matplotlib.pyplot as plt
from sbnd.plotlibrary import makeplot
from sbnd.general import plotters
from sbnd.stats.stats import (
    construct_covariance, construct_correlation_matrix, covariance_from_fraccov,
    get_fractional_uncertainty, get_total_unc, convert_smearing_to_response,
    get_smear_matrix, compute_efficiency, compute_sigma_tilde,
    _cov_eigenvalue_diagnostics, calc_chi2, fraccov_from_covariance
)
from tqdm import tqdm
import os
import copy
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from sbnd.numu.numu_constants import *
from sbnd.cafclasses.binning import Binning2D

class Systematics:
    """Class for handling systematic uncertainty processing and computation."""
    
    @staticmethod
    def _get_dict_template():
        """Get the template dictionary structure for systematics."""
        return {
            'cols': None,
            'col_names': None,
            'type': None,
            'name': None,
            'variation': None,
            'label': None,
            'description': '',
            'rank': None,
            'sigma_tilde': [],
            'sel': [],
            'response': [],
            'cv_response': None,
            'cv_sel': None,
            'cv_sigma_tilde': None,
            'xsec_cov': None,
            'xsec_cov_unaltered': None,
            'xsec_fraccov_unaltered': None,
            'xsec_cond': None,
            'xsec_eigvals': None,
            'xsec_iters': None,
            'event_cov': None,
            'event_cov_unaltered': None,
            'event_fraccov_unaltered': None,
            'event_cond': None,
            'event_eigvals': None,
            'event_iters': None,
            'xsec_corr': None,
            'event_corr': None,
            'xsec_fraccov': None,
            'event_fraccov': None,
            'xsec_fracunc': None,
            'event_fracunc': None,
            'event_totalunc': None,
            'xsec_totalunc': None,
            'order': None,
            'color': None,
        }
    
    def __init__(self, variable_name, bins,
                 reco_sel, reco_sel_background,
                 genweights_sig, genweights_sel, genweights_sel_background,
                 xsec_unit=None, true_sig=None, true_sel=None, true_sel_background=None,
                 keys=None,data=None,genweights_data=None,**kwargs):
        """
        Initialize Systematics class.
        
        Parameters
        ----------
        variable_name : str
            Name of the variable (e.g., 'costheta', 'momentum', 'differential')
        bins : array-like
            Bin edges for this variable
        reco_sel : array-like
            Reconstructed variable values for selected signal
        reco_sel_background : array-like
            Reconstructed variable values for selected background
        genweights_sig : array-like
            Generator weights for signal
        genweights_sel : array-like
            Generator weights for selected signal
        genweights_sel_background : array-like
            Generator weights for selected background
        xsec_unit : float, optional
            Cross section unit conversion factor. If None, smearing matrix will not be computed.
        true_sig : array-like or None
            True variable values for signal. If None, smearing matrix will not be computed.
        true_sel : array-like or None
            True variable values for selected signal. If None, smearing matrix will not be computed.
        true_sel_background : array-like or None
            True variable values for selected background. If None, smearing matrix will not be computed.
        keys : array-like, optional
            Keys from truth level to identify systematics (can be set later)
        data : array-like or None
            Data variable values. If None, data will not be used.
        genweights_data : array-like or None
            Generator weights for data. If None, data will not be used.
        **kwargs: dict
            Keyword arguments to pass to get_sys_keydict
        """
        # Store variable name and bins
        self.variable_name = variable_name
        self.bins = np.array(bins)
        self.xsec_unit = xsec_unit
        
        # Store distributions for later use (handle None)
        self._true_sig = np.array(true_sig) if true_sig is not None else None
        self._true_sel = np.array(true_sel) if true_sel is not None else None
        self._true_sel_background = np.array(true_sel_background) if true_sel_background is not None else None
        self._reco_sel = np.array(reco_sel)
        self._reco_sel_background = np.array(reco_sel_background)
        self._genweights_sig = np.array(genweights_sig)
        self._genweights_sel = np.array(genweights_sel)
        self._genweights_sel_background = np.array(genweights_sel_background)
        self._genweights_data = np.array(genweights_data) if genweights_data is not None else None
        self._data = np.array(data) if data is not None else None
        self.xlabel = self._get_default_xlabel()
        # Conditional assertions
        if self._true_sig is not None:
            assert len(self._true_sig) == len(self._genweights_sig), f'True signal and generator weights have different lengths: {len(self._true_sig)} != {len(self._genweights_sig)}'
        else:
            assert len(self._reco_sel) == len(self._genweights_sel), f'Reconstructed selected and generator weights have different lengths: {len(self._reco_sel)} != {len(self._genweights_sel)}'
        if self._true_sel is not None:
            assert len(self._true_sel) == len(self._genweights_sel) == len(self._reco_sel), f'True selected, reconstructed selected, and generator weights have different lengths: {len(self._true_sel)} != {len(self._genweights_sel)} != {len(self._reco_sel)}'
        else:
            assert len(self._reco_sel) == len(self._genweights_sel), f'Reconstructed selected and generator weights have different lengths: {len(self._reco_sel)} != {len(self._genweights_sel)}'
        if self._true_sel_background is not None:
            assert len(self._true_sel_background) == len(self._genweights_sel_background) == len(self._reco_sel_background), f'True selected background, reconstructed selected background, and generator weights have different lengths: {len(self._true_sel_background)} != {len(self._genweights_sel_background)} != {len(self._reco_sel_background)}'
        else:
            assert len(self._reco_sel_background) == len(self._genweights_sel_background), f'Reconstructed selected background and generator weights have different lengths: {len(self._reco_sel_background)} != {len(self._genweights_sel_background)}'
        if self._genweights_data is not None and self._data is not None:
            assert len(self._genweights_data) == len(self._data), f'Generator weights and data have different lengths: {len(self._genweights_data)} != {len(self._data)}'
        # Inline histogram computation (previously compute_histograms)
        # Signal truth histogram
        if self._true_sig is not None:
            sig_truth, _ = np.histogram(
                self._true_sig, bins=self.bins, weights=self._genweights_sig
            )
        else:
            sig_truth = None
        
        # Selected truth histogram
        if self._true_sel is not None:
            sel_truth, _ = np.histogram(
                self._true_sel, bins=self.bins, weights=self._genweights_sel
            )
        else:
            sel_truth = None
        
        # Selected background truth histogram
        if self._true_sel_background is not None:
            sel_background_truth, _ = np.histogram(
                self._true_sel_background, bins=self.bins, weights=self._genweights_sel_background
            )
        else:
            sel_background_truth = None
        
        # Selected reco histogram
        sel, _ = np.histogram(
            self._reco_sel, bins=self.bins, weights=self._genweights_sel
        )
        
        # Selected background reco histogram
        sel_background, _ = np.histogram(
            self._reco_sel_background, bins=self.bins, weights=self._genweights_sel_background
        )
        
        # Data histogram
        if self._data is not None:
            sel_data, _ = np.histogram(self._data, bins=self.bins, weights=self._genweights_data)
        else:
            sel_data = None
        self.sig_truth = sig_truth
        self.sel_truth = sel_truth
        self.sel_background_truth = sel_background_truth
        self.sel = sel
        self.sel_background = sel_background
        self.sel_data = sel_data
        assert len(self.sel) == len(self.sel_background), f'Selected and selected background have different lengths: {len(self.sel)} != {len(self.sel_background)}'
        
        # Compute efficiency if we have the required histograms
        if self.sig_truth is not None and self.sel_truth is not None:
            self.eff_truth = compute_efficiency(self.sig_truth, self.sel_truth)
        else:
            self.eff_truth = None

        # Compute eff_reco (efficiency over reco distribution)
        if self.sig_truth is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                self.eff_reco = np.where(self.sig_truth > 0, self.sel / self.sig_truth, 0.0)
        else:
            self.eff_reco = None
        
        # Compute smearing matrix, response matrix, and sigma_tilde only if we have true_sel and xsec_unit
        if self.sig_truth is not None and self.eff_truth is not None and self.xsec_unit is not None:
            self.smearing = get_smear_matrix(
                self._true_sel, self._reco_sel, self.bins,
                weights=self._genweights_sel
            )
            
            self.response = convert_smearing_to_response(self.smearing, self.eff_truth)
            
            self.sigma_tilde = compute_sigma_tilde(
                self.response, self.sig_truth, self.sel_background, self.xsec_unit
            )
        else:
            self.smearing = None
            self.response = None
            self.sigma_tilde = None
        
        self.systematics = {}
        if keys is not None:
            self._initialize_from_keys(keys, **kwargs)

    def _get_default_xlabel(self):
        """
        Default reconstructed-axis label for this variable.
        """
        if self.variable_name == 'costheta':
            return r'Reconstructed $\cos\theta_{\mu}$'
        if self.variable_name == 'momentum':
            return r'Reconstructed $p_{\mu}$ (GeV)'
        if self.variable_name == 'differential':
            return r'Reconstructed 2D Bin ID'
        if self.variable_name == 'opt0':
            return r'Reconstructed $1/Opt0$ Score'
        return f'{self.variable_name}'
    
    def add_syst(self, key, metadata=None):
        """Add a new systematic to the systematics dictionary."""
        dict_template = Systematics._get_dict_template()
        self.systematics[key] = dict_template.copy()
        if metadata is not None:
            for field in ('cols', 'col_names', 'type', 'name', 'variation', 'order'):
                self.systematics[key][field] = metadata.get(field)
            self.systematics[key]['label'] = metadata.get('label', metadata.get('name'))
            self.systematics[key]['description'] = metadata.get('description', '')

    def _initialize_from_keys(self, keys, **kwargs):
        """Build systematics entries from raw keys via get_sys_keydict."""
        sys_key_info = self.get_sys_keydict(keys, **kwargs)
        for key, metadata in sys_key_info.items():
            self.add_syst(key, metadata=metadata)

    def __repr__(self):
        return f'Systematics(variable={self.variable_name}, keys={list(self.systematics.keys())})'

    @property
    def types(self):
        """Return sorted unique systematic type names."""
        return sorted({
            sdict.get('type')
            for sdict in self.systematics.values()
            if sdict.get('type') not in (None, '')
        })

    def get_sys_keydict(self,keys,pattern=None,stype='RW'):
        """
        We're going to use the fact that the systematic are in the level of truth. '...'
        Return a dictionary with the base key as the dict key and the values are the 
        indices of the systematic in the data frame

        Parameters
        ----------
        keys: list
            List of keys to be processed
        pattern: list
            List of patterns to match
        stype: str
            Type of systematic
                RW : Reweightable (weights in the data frame)
                Det: Detector systematic (no weights in the data frame)

        Returns
        -------
        sys_col_dict: dict
            Dictionary with the base key as the dict key and the values are the 
            (1) list of tuples of the systematic keys and (2) the names of the systematic
        """
        def format_name(key):
            #Drop all the plesentaries
            key = key.replace('multisigma_', '')
            key = key.replace('multisim_', '')
            key = key.replace('nsigma_', '')
            key = key.replace('GENIEReWeight_SBN_v1_','')
            key = key.replace('GENIEReWeight_', '')
            key = key.replace('_Flux', '_flux')
            key = key.replace('reinteractions_', '')
            key = key.replace('_Geant4', '_g4')
            key = key.replace('SBNNuSyst_', '')
            key = key.replace('MECq0q3InterpWeighting_', '')
            key = key.replace('SuSAv2To', '')
            key = key.replace('q0binned_', '')
            key = key.replace('q0bin', '')
            return key
        def assign_type(key,stype='RW'):
            if stype == 'RW':
                if 'genie' in key.lower() or 'sbnnusyst' in key.lower() or 'susav2' in key.lower():
                    return 'xsec'
                elif 'flux' in key.lower():
                    return 'flux'
                elif 'geant4' in key.lower():
                    return 'g4'
                elif 'stat' in key.lower():
                    return 'stat'
                else:
                    print(f'Unknown systematic type: {key}')
                    return 'unknown'
            elif stype == 'Det':
                if 'pmt' in key.lower() or 'pds' in key.lower():
                    return 'pds'
                elif 'sce' in key.lower():
                    return 'sce'
                elif 'wiremod' in key.lower():
                    return 'tpc'
                elif 'alpha' in key.lower() or 'beta' in key.lower() or 'R_' in key or 'c_cal_frac' in key.lower():
                    return 'calo'
                else:
                    print(f'Unknown systematic type: {key}')
                    return 'unknown'
            elif stype == 'Cosmic':
                return 'cosmic'
            else:
                print(f'Unknown systematic type: {stype}')
                return 'unknown'
        def assign_variation(key,stype='RW'):
            if stype == 'RW':
                if 'multisigma' in key.lower():
                    return 'multisigma'
                elif 'multisim' in key.lower() or 'flux' in key.lower() or 'geant4' in key.lower():
                    return 'multisim'
                else:
                    return 'multisim'
            elif stype == 'Det' or stype == 'Cosmic':
                return 'unisim'
            else:
                print(f'Unknown systematic type: {stype}')
                return 'unknown'
        def assign_candidate(candidate):
            if candidate == 'stat':
                return 'stat_rw' #Change this to stat_rw or something if combining other stat unc
            return candidate
        sys_col_dict = {}
        cnt = -1
        for k in keys:
            #assert k[0] == 'truth', f'Systematics are not at truth level: {k}'
            candidate = k[0] if stype == 'RW' else k
            passes = False
            if pattern is not None:
                for p in pattern:
                    if p in candidate:
                        passes = True
                        break
            else:
                passes = True #No pattern
            if not passes:
                continue
                    #print(candidate)
            candidate = assign_candidate(candidate)
            if candidate not in sys_col_dict:
                formatted_name = format_name(candidate)
                sys_col_dict[candidate] = {'cols' : [], 'col_names' : [], 'type' : assign_type(candidate,stype), 'name' : formatted_name, 'label' : formatted_name, 'description' : '', 'variation' : assign_variation(candidate,stype)}
            cnt+=1
            if stype == 'RW':
                sys_col_dict[candidate]['cols'].append(tuple(['truth']+list(k)))
                sys_col_dict[candidate]['col_names'].append('_'.join(list(k)).rstrip('_'))
        return sys_col_dict
    
    def process_systematics(self, mc_signal_data, mc_sel_signal_data, mc_sel_background_data,
                           progress_bar=False,sys_names=None):
        """
        Process all systematic universes and compute histograms, efficiencies, 
        smearing/response matrices, and sigma_tilde values.
        
        Parameters
        ----------
        mc_signal_data : data frame
            MC signal slice with data attribute containing systematic columns
        mc_sel_signal_data : data frame
            MC selected signal slice with data attribute containing systematic columns
        mc_sel_background_data : data frame
            MC selected background slice with data attribute containing systematic columns
        progress_bar : bool
            Whether to show progress bar (requires tqdm)
        sys_names : list of str, optional
            Names of the systematics to process. If None, process all systematics
        """
        # Set items to iterate over
        if sys_names is None:
            iterator = self.systematics.items()
            keys_to_delete = []
        else:
            keys_to_delete = [k for k in self.systematics.keys() if k not in sys_names]
            iterator = [(k,v) for k,v in self.systematics.items() if k in sys_names]
            #Print any missing systematics from sys_names
            for k in sys_names:
                if k not in self.systematics:
                    print(f' - WARNING: {k} not found in systematics dictionary')
        if progress_bar:
            iterator = tqdm(iterator, unit=' goomba')
        #print(f'Processing ({list(self.systematics.keys())})')
        for key, sys_dict in iterator:
            cols = sys_dict['cols']
            col_names = sys_dict['col_names']
            if cols is None:
                print(f'WARNING: {key} has no cols')
                continue
            # Reset lists for this systematic
            sys_dict['sigma_tilde'] = []
            sys_dict['sel'] = []
            sys_dict['sel_background'] = []
            sys_dict['eff_truth'] = []
            sys_dict['response'] = []
            sys_dict['cv_response'] = None
            is_multisigma = sys_dict['variation'] == 'multisigma'
            has_xsec = (self._true_sig is not None and self._true_sel is not None
                        and self._true_sel_background is not None and self.xsec_unit is not None)

            ref_sig_truth = self.sig_truth
            ref_eff_truth = self.eff_truth

            # For multisigma, reorder so the CV column is processed first
            iter_cols, iter_names = cols, col_names
            if is_multisigma:
                cv_idx = [i for i, cn in enumerate(col_names) if '_cv' in cn]
                other_idx = [i for i, cn in enumerate(col_names) if '_cv' not in cn]
                order = cv_idx + other_idx
                iter_cols = [cols[i] for i in order]
                iter_names = [col_names[i] for i in order]

            for col, col_name in zip(iter_cols, iter_names):
                is_cv = '_cv' in col_name and is_multisigma
                if col not in mc_sel_signal_data.keys() or col not in mc_sel_background_data.keys() or col not in mc_signal_data.keys():
                    continue
                rw_sel_signal = mc_sel_signal_data[col].copy()
                rw_sel_background = mc_sel_background_data[col].copy()
                rw_signal = mc_signal_data[col].copy()
                rw_sel_signal[(abs(rw_sel_signal) == np.inf) | (np.isnan(rw_sel_signal))] = 1
                rw_sel_background[(abs(rw_sel_background) == np.inf) | (np.isnan(rw_sel_background))] = 1
                rw_signal[(abs(rw_signal) == np.inf) | (np.isnan(rw_signal))] = 1
                rw_sel_signal = np.clip(rw_sel_signal, 0, 10)
                rw_sel_background = np.clip(rw_sel_background, 0, 10)
                rw_signal = np.clip(rw_signal, 0, 10)
                _sel, _ = np.histogram(
                    self._reco_sel, bins=self.bins,
                    weights=self._genweights_sel * rw_sel_signal
                )
                _sel_background, _ = np.histogram(
                    self._reco_sel_background, bins=self.bins,
                    weights=self._genweights_sel_background * rw_sel_background
                )

                if is_cv:
                    sys_dict['cv_sel'] = _sel + _sel_background
                else:
                    sys_dict['sel'].append(_sel + _sel_background)
                    sys_dict['sel_background'].append(_sel_background)

                if has_xsec:
                    _sig_truth, _ = np.histogram(
                        self._true_sig, bins=self.bins,
                        weights=self._genweights_sig * rw_signal
                    )
                    _sel_truth, _ = np.histogram(
                        self._true_sel, bins=self.bins,
                        weights=self._genweights_sel * rw_sel_signal
                    )
                    _smearing = get_smear_matrix(
                        self._true_sel, self._reco_sel, self.bins,
                        weights=self._genweights_sel * rw_sel_signal
                    )
                    _eff_truth = compute_efficiency(_sig_truth, _sel_truth)
                    if is_cv:
                        ref_sig_truth = _sig_truth
                        ref_eff_truth = _eff_truth
                        _response = convert_smearing_to_response(_smearing, _eff_truth)
                        sys_dict['cv_sigma_tilde'] = compute_sigma_tilde(
                            _response, ref_sig_truth, _sel_background, self.xsec_unit)
                        sys_dict['cv_response'] = _response
                    else:
                        if sys_dict['type'] == 'xsec':
                            _response = convert_smearing_to_response(_smearing, _eff_truth)
                            sys_dict['eff_truth'].append(_eff_truth)
                        else:
                            ref_eff_truth = compute_efficiency(ref_sig_truth, _sel_truth)
                            _response = convert_smearing_to_response(_smearing, ref_eff_truth)
                            sys_dict['eff_truth'].append(ref_eff_truth)
                        _sigma_tilde = compute_sigma_tilde(
                            _response, ref_sig_truth, _sel_background, self.xsec_unit
                        )
                        sys_dict['sigma_tilde'].append(_sigma_tilde)
                        sys_dict['response'].append(_response)
                elif not is_cv:
                    sys_dict['sigma_tilde'].append(None)
                    sys_dict['eff_truth'].append(None)
                    sys_dict['response'].append(None)

            sys_dict['rank'] = len(sys_dict['sel'])
            if len(sys_dict['sel']) == 0:
                keys_to_delete.append(key)
                print(f'WARNING: {key} has no usable universes, deleting')
                continue
        for key in keys_to_delete:
            self.systematics.pop(key)
    
    def process_det_systematics(self,
                                reco_sel_vars, reco_sel_background_vars,genweights_sel_vars,
                                genweights_sel_background_vars,
                                true_sig_vars=None, true_sel_vars=None, true_sel_background_vars=None,
                                sys_names=None,progress_bar=False, use_cv_background=False):
        """
        Process detector systematic variation (unisim, no weights).
        
        Parameters
        ----------
        reco_sel_vars : list of array-like
            Reco variable for selected (variation)
        reco_sel_background_vars : list of array-like
            Reco variable for selected background (variation)x
        genweights_sel_vars : list of array-like
            Generator weights for selected (variation)
        genweights_sel_background_vars : list of array-like
            Generator weights for selected background (variation)
        true_sig_vars : list of array-like, optional
            True variable for signal (variation). If None, sigma_tilde will not be computed.
        true_sel_vars : list of array-like, optional
            True variable for selected (variation). If None, sigma_tilde will not be computed.
        true_sel_background_vars : list of array-like, optional
            True variable for selected background (variation). If None, sigma_tilde will not be computed.
        sys_names : list of str, optional
            Names of the systematics. If None, use the keys of systematics dictionary
        progress_bar : bool
            Whether to show progress bar (requires tqdm)
        use_cv_background : bool
            Whether to use the CV background for the systematic. If True, the background will be the CV background. If False, the background will be the selected background.
        """
        #print(f'len(reco_sel_vars): {len(reco_sel_vars)}')
        #TODO: Handle case where sys_name is not in the systematics dictionary
        assert len(reco_sel_vars) == len(reco_sel_background_vars) == len(genweights_sel_vars) == len(genweights_sel_background_vars), f'Reconstructed, reconstructed background, generator weights for selected, and generator weights for selected background have different lengths: {len(reco_sel_vars)} != {len(reco_sel_background_vars)} != {len(genweights_sel_vars)} != {len(genweights_sel_background_vars)}'
        if sys_names is None:
            # Check lengths for non-None parameters
            if true_sig_vars is not None and true_sel_vars is not None and true_sel_background_vars is not None:
                assert len(true_sig_vars) == len(true_sel_vars) == len(true_sel_background_vars) == len(reco_sel_vars), f'True signal, selected, selected background have different lengths from reconstructed: {len(true_sig_vars)} != {len(true_sel_vars)} != {len(true_sel_background_vars)} != {len(reco_sel_vars)}'
            sys_names = list(self.systematics.keys())
            provided = True
        else:
            for sys_name in sys_names:
                if sys_name not in self.systematics:
                    raise ValueError(f'Systematic {sys_name} not found in systematics dictionary')
            provided = True
        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(enumerate(sys_names), unit=' goomba')
            except ImportError:
                iterator = enumerate(sys_names)
        else:
            iterator = enumerate(sys_names)
        #It's important that the keys are in the same order as the lists passed into this function
        for i,sys_name in iterator:
            #TODO: Handle case where sys_name is not in the systematics dictionary
            if provided:
                sys_dict = self.systematics[sys_name]
            
            #Clear the lists for this systematic
            sys_dict['sigma_tilde'] = []
            sys_dict['sel'] = []
            sys_dict['response'] = []
            sys_dict['cv_response'] = None
            sys_dict['rank'] = 1 #Only single sigma variation so rank is 1
            #print(f'i: {i}, sys_name: {sys_name}, len(reco_sel_vars): {len(reco_sel_vars)}')

            reco_sel_var = reco_sel_vars[i]
            reco_sel_background_var = reco_sel_background_vars[i]
            genweights_sel_var = genweights_sel_vars[i]
            genweights_sel_background_var = genweights_sel_background_vars[i]
            
            # Get true variables if provided
            true_sig_var = true_sig_vars[i] if true_sig_vars is not None else None
            true_sel_var = true_sel_vars[i] if true_sel_vars is not None else None
            true_sel_background_var = true_sel_background_vars[i] if true_sel_background_vars is not None else None

            #Make histograms of the variables
            _sel, _ = np.histogram(
                reco_sel_var, bins=self.bins,
                weights=genweights_sel_var
            )
            if use_cv_background:
                _sel_background = self.sel_background
            else:
                _sel_background, _ = np.histogram(
                    reco_sel_background_var, bins=self.bins,
                    weights=genweights_sel_background_var
                )

            # Store sel results
            sys_dict['sel'].append(_sel + _sel_background)

            # Compute sigma_tilde only if we have the required attributes and true variables
            if (true_sel_var is not None and self.eff_truth is not None and 
                self.sel_truth is not None and self.xsec_unit is not None):
                # Compute smearing matrix for variation
                _var_smearing = get_smear_matrix(
                    true_sel_var, reco_sel_var, self.bins, weights=genweights_sel_var
                )


                _sel_truth, _ = np.histogram(
                    true_sel_var, bins=self.bins,
                    weights=genweights_sel_var
                )
                # Use CV efficiency (not variation-specific) for response matrix conversion
                _eff_truth = compute_efficiency(self.sig_truth, _sel_truth)
                _var_response = convert_smearing_to_response(
                    _var_smearing, _eff_truth
                )
                
                # Compute sigma_tilde for variation
                _var_sigma_tilde = compute_sigma_tilde(
                    _var_response, self.sig_truth, _sel_background, self.xsec_unit
                )
                
                # Store results
                sys_dict['sigma_tilde'].append(_var_sigma_tilde)
                sys_dict['response'].append(_var_response)
            else:
                # If we don't have the required attributes, append None
                sys_dict['sigma_tilde'].append(None)
                sys_dict['response'].append(None)
    
    def process_flat_systematic(self, sys_key, frac_unc):
        """
        Process a flat systematic with a given fractional uncertainty.
        Creates two symmetric variations (+/-) that will produce a flat covariance.
        
        Parameters
        ----------
        sys_key : str
            Key for the systematic in self.systematics dictionary
        frac_unc : float
            Overall fractional uncertainty (e.g., 0.05 for 5%)
        """
        if sys_key not in self.systematics:
            dict_template = Systematics._get_dict_template()
            sys_dict = dict_template.copy()
            sys_dict['type'] = sys_key
            sys_dict['name'] = sys_key
            sys_dict['label'] = sys_key
            sys_dict['description'] = ''
            sys_dict['variation'] = 'flat'
            sys_dict['order'] = None
            self.systematics[sys_key] = sys_dict
        else:
            sys_dict = self.systematics[sys_key]
        
        # Reset lists for this systematic
        sys_dict['sigma_tilde'] = []
        sys_dict['sel'] = []
        sys_dict['response'] = []
        sys_dict['cv_response'] = None
        # Create two symmetric variations: +frac_unc and -frac_unc
        for sign in [+1, -1]:
            # Apply to sel (reco event rate)
            _sel_variation = (self.sel + self.sel_background) * (1 + sign * frac_unc)
            sys_dict['sel'].append(_sel_variation)
            
            # Apply to sigma_tilde (xsec) only if it exists
            if self.sigma_tilde is not None:
                _sigma_tilde_variation = self.sigma_tilde * (1 + sign * frac_unc)
                sys_dict['sigma_tilde'].append(_sigma_tilde_variation)
            else:
                sys_dict['sigma_tilde'].append(None)
            if self.response is not None:
                sys_dict['response'].append(np.copy(self.response))
            else:
                sys_dict['response'].append(None)
    
    def process_stat_systematics(self, sys_key='stat'):
        """
        Process statistical (Poisson) systematic uncertainty.
        Creates two symmetric variations (+/-) based on Poisson uncertainty sqrt(N).
        
        Parameters
        ----------
        sys_key : str
            Key for the systematic in self.systematics dictionary
        """
        if sys_key not in self.systematics:
            dict_template = Systematics._get_dict_template()
            sys_dict = dict_template.copy()
            sys_dict['type'] = 'stat'
            sys_dict['name'] = sys_key
            sys_dict['label'] = sys_key
            sys_dict['description'] = ''
            sys_dict['variation'] = 'stat'
            sys_dict['order'] = None
            self.systematics[sys_key] = sys_dict
        else:
            sys_dict = self.systematics[sys_key]
        
        # Reset lists for this systematic
        sys_dict['sigma_tilde'] = []
        sys_dict['sel'] = []
        sys_dict['response'] = []
        sys_dict['cv_response'] = None
        sys_dict['rank'] = -1 #Make sure the covarariance is diagonal
        
        # Create two symmetric variations: +sqrt(N) and -sqrt(N)
        for sign in [+1, -1]:
            # Apply to sel (reco event rate)
            sel_total = self.sel + self.sel_background
            # Poisson uncertainty is sqrt(N)
            sel_unc = np.sqrt(sel_total)
            with np.errstate(divide='ignore', invalid='ignore'):
                frac_unc = np.where(sel_total > 0, sel_unc / sel_total, 0)
            _sel_variation = sel_total + sign * sel_unc
            # Ensure non-negative
            _sel_variation = np.maximum(_sel_variation, 0)
            sys_dict['sel'].append(_sel_variation)
            
            # Apply to sigma_tilde (xsec) only if it exists
            if self.sigma_tilde is not None:
                # Get fractional uncertainty from selection, then apply to sigma_tilde
                sigma_tilde_unc = frac_unc * self.sigma_tilde
                sys_dict['sigma_tilde'].append(self.sigma_tilde + sign * sigma_tilde_unc)
            else:
                sys_dict['sigma_tilde'].append(None)
            if self.response is not None:
                sys_dict['response'].append(np.copy(self.response))
            else:
                sys_dict['response'].append(None)
    
    def compute_covariances(self, keys=None, check=False, do_correct_negative_eigenvalues=False, compute_xsec_cov=True, **kwargs):
        """
        Compute covariance matrices, correlations, and uncertainties for each systematic.

        Parameters
        ----------
        keys : list, optional
            List of keys to compute covariance matrices for. If None, compute for all keys
        check : bool, optional
            If True, compute the eigenvalues and store the difference between the largest and smallest eigenvalues
            This is a measure of the condition number of the covariance matrix
        do_correct_negative_eigenvalues : bool, optional
            If True, correct negative eigenvalues of the covariance matrix
        compute_xsec_cov : bool, optional
            If True, compute xsec (sigma_tilde) covariances. If False, xsec cov fields are left None.
        **kwargs : dict
            Additional keyword arguments to pass to the construct_covariance function
        """
        if keys is None:
            keys = list(self.systematics.keys())
        else:
            #Check that keys are in the systematics dictionary
            for key in keys:
                if key not in self.systematics:
                    raise ValueError(f'Key {key} not found in systematics dictionary')
        #print(f'Processing covariances for variable <{self.variable_name}> with keys:\n{keys}')
        for key in keys:
            sys_dict = self.systematics[key]
            if len(sys_dict['sel']) == 0:
                raise ValueError(f'No sel events for systematic {key} for {self.variable_name}')
            # First do event covariance
            event_cv = sys_dict['cv_sel'] if sys_dict.get('cv_sel') is not None else self.sel + self.sel_background
            _cov, _cov_unaltered, _frac, _frac_unaltered, _corr, _fracunc, _event_iters = construct_covariance(
                event_cv, sys_dict['sel'],
                rank=sys_dict['rank'],
                do_correct_negative_eigenvalues=do_correct_negative_eigenvalues,
                assert_cov=False,
                **kwargs
            )
            sys_dict['event_cov'] = _cov
            sys_dict['event_cov_unaltered'] = _cov_unaltered
            sys_dict['event_fraccov'] = _frac
            sys_dict['event_fraccov_unaltered'] = _frac_unaltered
            sys_dict['event_corr'] = _corr
            sys_dict['event_fracunc'] = _fracunc
            if _event_iters is not None:
                sys_dict['event_iters'] = _event_iters
            sys_dict['event_totalunc'] = get_total_unc(event_cv, sys_dict['event_fracunc'])
            if check:
                eigvals, eigvecs, cond, min_eig, max_eig = _cov_eigenvalue_diagnostics(_cov)
                sys_dict['event_cond'] = cond
                sys_dict['event_eigvals'] = eigvals

            # Xsec (only compute if sigma_tilde exists and compute_xsec_cov; use type-specific CV when available)
            if compute_xsec_cov and self.sigma_tilde is not None and sys_dict['sigma_tilde'] is not None:
                if len(sys_dict['sigma_tilde']) == 0:
                    raise ValueError(f'No sigma_tilde events for systematic {key} for {self.variable_name} (but we expect them)')
                sigma_tilde_list = sys_dict['sigma_tilde']
                xsec_cv = sys_dict['cv_sigma_tilde'] if sys_dict.get('cv_sigma_tilde') is not None else self.sigma_tilde
                _xsec_cov, _xsec_cov_unaltered, _xsec_frac, _xsec_frac_unaltered, _xsec_corr, _xsec_fracunc, _xsec_iters = construct_covariance(
                    xsec_cv, sigma_tilde_list,
                    rank=sys_dict['rank'],
                    do_correct_negative_eigenvalues=do_correct_negative_eigenvalues,
                    assert_cov=False,
                    **kwargs
                )
                sys_dict['xsec_cov'] = _xsec_cov
                sys_dict['xsec_cov_unaltered'] = _xsec_cov_unaltered
                sys_dict['xsec_fraccov'] = _xsec_frac
                sys_dict['xsec_fraccov_unaltered'] = _xsec_frac_unaltered
                sys_dict['xsec_corr'] = _xsec_corr
                sys_dict['xsec_fracunc'] = _xsec_fracunc
                if _xsec_iters is not None:
                    sys_dict['xsec_iters'] = _xsec_iters
                sys_dict['xsec_totalunc'] = get_total_unc(xsec_cv, sys_dict['xsec_fracunc'])
                if check:
                    eigvals, eigvecs, cond, min_eig, max_eig = _cov_eigenvalue_diagnostics(_xsec_cov)
                    sys_dict['xsec_cond'] = cond
                    sys_dict['xsec_eigvals'] = eigvals
            else:
                sys_dict['xsec_cov'] = None
                sys_dict['xsec_fraccov'] = None
                sys_dict['xsec_fraccov_unaltered'] = None
                sys_dict['xsec_corr'] = None
                sys_dict['xsec_fracunc'] = None
                sys_dict['xsec_totalunc'] = None
            if 'cosmic' in key and compute_xsec_cov:
                #Special treatment for cosmic uncertainty, just use the event uncertainty, scaled by xsec unit
                print(f'sel_background: {self.sel_background}, sel: {self.sel}, xsec_unit: {self.xsec_unit}')
                sys_dict['xsec_cov'] = covariance_from_fraccov(
                    sys_dict['event_fraccov'], (self.sel_background + self.sel) * self.xsec_unit
                )
                sys_dict['xsec_fraccov'] = sys_dict['event_fraccov']
                sys_dict['xsec_fraccov_unaltered'] = sys_dict['event_fraccov_unaltered']
                sys_dict['xsec_corr'] = sys_dict['event_corr']
                sys_dict['xsec_fracunc'] = sys_dict['event_fracunc']
                sys_dict['xsec_totalunc'] = sys_dict['event_totalunc']
                if check:
                    eigvals, eigvecs, cond, min_eig, max_eig = _cov_eigenvalue_diagnostics(_xsec_cov)
                    sys_dict['xsec_cond'] = cond
                    sys_dict['xsec_eigvals'] = eigvals
            
    def compute_inverse_covariances(self, keys=None, **kwargs):
        """
        Compute inverse covariance matrices for each systematic.
        """
        if keys is None:
            keys = list(self.systematics.keys())
        for key in keys:
            sys_dict = self.systematics[key]
            if sys_dict['event_cov'] is not None:
                sys_dict['event_inv_cov'] = np.linalg.pinv(sys_dict['event_cov'], **kwargs)
            else:
                sys_dict['event_inv_cov'] = None
            if sys_dict['xsec_cov'] is not None:
                sys_dict['xsec_inv_cov'] = np.linalg.pinv(sys_dict['xsec_cov'], **kwargs)
            else:
                sys_dict['xsec_inv_cov'] = None
    def get_keys(self,key=None,pattern=None):
        """Check `key` for a given `pattern` in the systematics dictionary.
        
        Parameters
        ----------
        key : str
            The key to check
        pattern : str
            The pattern to check for
        
        Returns
        -------
        keys : list
            list of systematic keys that match the pattern
        """
        keys = []
        for k,sdict in self.systematics.items():
            if key is not None and pattern is not None:
                if pattern in sdict[key]:
                    keys.append(k)
            elif pattern is not None: #directly check the key k
                if pattern in k:
                    keys.append(k)
            elif key is not None: #directly check the key sdict[key]
                if key in sdict.keys():
                    keys.append(k)
        return keys
    
    def combine(self, other, other_override=False, store_other=False, other_name='other', rescale_covs=True, rescale_fraccovs=False):
        """
        Combine another Systematics object into this one.
        Non-overlapping keys from other are added to this object.
        
        Parameters
        ----------
        other : Systematics
            Another Systematics object to combine with this one
        other_override : bool, optional
            If True, override existing keys in other with keys from self.systematics
        store_other : bool, optional
            If True, store the other object in self.systematics as other_name
        other_name : str, optional
            The name to store the other object in self.systematics
        rescale_covs : bool, optional
            If True, rescale the covariance matrices to the selected and background of this object
            This is necessary because the covariance matrices are computed on the selected and background of the other object
        rescale_fraccovs : bool, optional
            If True, rescale the fractional covariance matrices to the covariance and selected/background of this object
            Also rescale the fractional uncertainties
        """
        # Check compatibility
        if not np.array_equal(self.bins, other.bins):
            raise ValueError(f'Cannot combine Systematics objects: bins differ')
        if self.variable_name != other.variable_name:
            raise ValueError(f'Cannot combine Systematics objects: variable names differ ({self.variable_name} vs {other.variable_name})')
        if rescale_covs and rescale_fraccovs:
            raise ValueError(f'Both rescale_covs and rescale_fraccovs cannot be True')
        # if self.xsec_unit != other.xsec_unit:
        #     raise ValueError(f'Cannot combine Systematics objects: xsec_unit differs ({self.xsec_unit} vs {other.xsec_unit})')
        
        # Add non-overlapping keys from other to self
        for key, sys_dict in other.systematics.items():
            if key not in self.systematics:
                #print(f'Adding key {key} to self.systematics')
                # Deep copy the systematic dictionary
                self.systematics[key] = copy.deepcopy(sys_dict)
                valid = self.systematics[key]['event_cov'] is not None
                if rescale_covs and valid:
                    self.systematics[key]['event_cov'] = covariance_from_fraccov(
                        self.systematics[key]['event_fraccov'], self.sel + self.sel_background
                    )
                    valid_xsec = self.systematics[key]['xsec_cov'] is not None
                    if self.sigma_tilde is not None and valid_xsec:
                        self.systematics[key]['xsec_cov'] = covariance_from_fraccov(
                            self.systematics[key]['xsec_fraccov'], self.sigma_tilde
                        )
                if rescale_fraccovs and valid:
                    self.systematics[key]['event_fraccov'] = fraccov_from_covariance(
                        self.systematics[key]['event_cov'], self.sel + self.sel_background
                    )
                    self.systematics[key]['event_fracunc'] = get_fractional_uncertainty(self.sel + self.sel_background, self.systematics[key]['event_cov'])
                    self.systematics[key]['event_totalunc'] = get_total_unc(self.sel + self.sel_background, self.systematics[key]['event_fracunc'])
                    valid_xsec = self.systematics[key]['xsec_cov'] is not None
                    if self.sigma_tilde is not None and valid_xsec:
                        self.systematics[key]['xsec_fraccov'] = fraccov_from_covariance(
                            self.systematics[key]['xsec_cov'], self.sigma_tilde
                        )
                        self.systematics[key]['xsec_fracunc'] = get_fractional_uncertainty(self.sigma_tilde, self.systematics[key]['xsec_cov'])
                        self.systematics[key]['xsec_totalunc'] = get_total_unc(self.sigma_tilde, self.systematics[key]['xsec_fracunc'])
            # If key exists in both, skip it (non-overlapping only)
            else:
                if other_override:
                    #print(f'Key {key} already exists in self.systematics and other_override is False, skipping')
                    pass
                else:
                    self.systematics[key] = copy.deepcopy(sys_dict)
                    valid = self.systematics[key]['event_cov'] is not None
                    if rescale_covs and valid:
                        self.systematics[key]['event_cov'] = covariance_from_fraccov(
                            self.systematics[key]['event_fraccov'], self.sel + self.sel_background
                        )
                        valid_xsec = self.systematics[key]['xsec_cov'] is not None
                        if self.sigma_tilde is not None and valid_xsec:
                            self.systematics[key]['xsec_cov'] = covariance_from_fraccov(
                                self.systematics[key]['xsec_fraccov'], self.sigma_tilde
                            )
                    if rescale_fraccovs and valid:
                        self.systematics[key]['event_fraccov'] = fraccov_from_covariance(
                            self.systematics[key]['event_cov'], self.sel + self.sel_background
                        )
                        self.systematics[key]['event_fracunc'] = get_fractional_uncertainty(self.sel + self.sel_background, self.systematics[key]['event_cov'])
                        self.systematics[key]['event_totalunc'] = get_total_unc(self.sel + self.sel_background, self.systematics[key]['event_fracunc'])
                        valid_xsec = self.systematics[key]['xsec_cov'] is not None
                        if self.sigma_tilde is not None and valid_xsec:
                            self.systematics[key]['xsec_fraccov'] = fraccov_from_covariance(
                                self.systematics[key]['xsec_cov'], self.sigma_tilde)
                            self.systematics[key]['xsec_fracunc'] = get_fractional_uncertainty(self.sigma_tilde, self.systematics[key]['xsec_cov'])
                            self.systematics[key]['xsec_totalunc'] = get_total_unc(self.sigma_tilde, self.systematics[key]['xsec_fracunc'])
        
        if store_other:
            sdict = Systematics._get_dict_template()
            sdict['name'] = other_name
            sdict['sel'] = other.sel
            sdict['sel_background'] = other.sel_background
            sdict['sigma_tilde'] = other.sigma_tilde
            sdict['variation'] = 'self'
            self.systematics[other_name] = sdict

        #print(f'self.systematics: {self.systematics.keys()}')
    def _calc_chi2(self,keys=None,exclude_keys=None,include_summary=False,check_myself=False,use_diag=False,cov_key='event_cov',**kwargs):
        """
        Calculate the chi2 for set of keys.

        Parameters
        ----------
        keys : list, optional
            List of keys to calculate chi2 for. If None, calculate for all keys
        exclude_keys : list, optional
            List of keys to exclude from the calculation
        include_summary : bool, optional
            If True, include summary keys in the calculation
        **kwargs : dict
            Additional keyword arguments to pass to the calc_chi2 function
        
        Returns
        -------
        chi2_dict : dict
            Dictionary with - keys used, chi2 value, dof, p-value, covariance matrix, pred, true
        """
        assert self.sel_data is not None, 'No data is provided, provide it in initialization'
        assert self.sel is not None and self.sel_background is not None, 'No sel or sel_background is provided, provide it in initialization'
        if keys is None:
            keys = list(self.systematics.keys())
        if exclude_keys is not None:
            keys = [k for k in keys if k not in exclude_keys]
        if not include_summary:
            keys = [k for k in keys if self.systematics[k]['variation'] != 'summary' and self.systematics[k]['variation'] != 'self']
        chi2_dict = {}
        # First get the combined covariance matrix
        n_bins = len(self.bins) - 1
        cov = np.zeros((n_bins, n_bins))
        for key in keys:
            sys_dict = self.systematics[key]
            _cov = sys_dict[cov_key]
            if _cov is not None:
                cov += _cov
            else:
                continue
                #print(f'WARNING: No covariance matrix for {key}')
        # Then get the total chi2
        pred = self.sel+self.sel_background
        true = self.sel_data
        if use_diag:
            cov = np.eye(cov.shape[0])*np.diag(cov)
        chi2, dof, p_value = calc_chi2(pred, true, cov, **kwargs)
        #Form chi2_dict
        chi2_dict['keys'] = keys
        chi2_dict['chi2'] = chi2
        chi2_dict['dof'] = dof
        chi2_dict['pvalue'] = p_value
        chi2_dict['cov'] = cov
        chi2_dict['pred'] = pred
        chi2_dict['true'] = true
        #Check the covariance matrix wrt the total covariance matrix
        if check_myself:
            print(f'   ')
        return chi2_dict

    def _get_cv_source(self, sys_type, sys_name):
        """Get the correct CV source for a given systematic type.
        
        Parameters
        ----------
        sys_type : str
            The type of systematic (e.g., 'xsec', 'flux', 'geant4', 'cosmic', 'det')
        sys_name : str
            The systematic name. Used for special-case CV routing.
        
        Returns
        -------
        dict or None
            The CV source dictionary if found, None otherwise
        """
        # If 'geant4_syst' exists and this is a RW systematic (not detector), use slim CV
        # If 'cosmic_data' exists and this is a cosmic systematic, use cosmic CV
        # Detector systematics are typically 'Det' type, while slim are 'RW' type
        if 'geant4_syst' in self.systematics and sys_type in ['xsec', 'flux', 'g4']:
            return self.systematics['geant4_syst']
        elif 'cosmic_data' in self.systematics and sys_type == 'cosmic':
            return self.systematics['cosmic_data']
        elif 'det_data' in self.systematics and sys_type in ['pds','tpc','sce','calo']:
            return self.systematics['det_data']
        elif sys_type == 'stat' and ('lowe' in sys_name) and ('lowe' in self.systematics):
            return self.systematics['lowe']
        return None
    
    def _get_cv_for_type(self, sys_type, cv_key='sel', sys_dict=None):
        """Get the correct CV array for a given systematic type and CV key.
        
        Parameters
        ----------
        sys_type : str
            The type of systematic
        cv_key : str
            Either 'sel' or 'sigma_tilde'
        sys_dict : dict, optional
            Per-systematic dict. If it contains cv_sel / cv_sigma_tilde, those
            take priority over type-based lookup.
        
        Returns
        -------
        np.ndarray or None
            The CV array, or None if not available
        """
        if sys_dict is not None:
            if cv_key == 'sigma_tilde' and sys_dict.get('cv_sigma_tilde') is not None:
                #print(f'Using cv_sigma_tilde for {sys_dict["name"]}')
                return sys_dict['cv_sigma_tilde']
            elif cv_key == 'sel' and sys_dict.get('cv_sel') is not None:
                return sys_dict['cv_sel']

        sys_name = ''
        if sys_dict is not None and sys_dict.get('name') is not None:
            sys_name = sys_dict.get('name')
        cv_source = self._get_cv_source(sys_type, sys_name)
        
        if cv_source is not None:
            if cv_key == 'sigma_tilde' and cv_source.get('sigma_tilde') is not None:
                # Check if cv_source has array (CV) not list (variations)
                if isinstance(cv_source['sigma_tilde'], np.ndarray):
                    return cv_source['sigma_tilde']
            elif cv_key == 'sel' and cv_source.get('sel') is not None:
                if isinstance(cv_source['sel'], np.ndarray):
                    cv_bg = cv_source.get('sel_background', self.sel_background)
                    if isinstance(cv_bg, np.ndarray):
                        return cv_source['sel'] + cv_bg
                    else:
                        return cv_source['sel'] + self.sel_background
        
        # Default to self CV
        if cv_key == 'sigma_tilde':
            return self.sigma_tilde
        elif cv_key == 'sel':
            return self.sel + self.sel_background
        return None
    
    def combine_summaries(self,summary_keys=['total']):
        """Combine systematics into summary groups.
        
        Parameters
        ----------
        summary_keys : list
            List of summary keys to combine. Each summary key should be a type
        """
        
        # Create summary dictionaries
        dict_template = Systematics._get_dict_template()
        
        for sk in summary_keys:
            self.systematics[sk] = dict_template.copy()
            self.systematics[sk]['cols'] = None
            self.systematics[sk]['col_names'] = None
            self.systematics[sk]['type'] = sk
            self.systematics[sk]['name'] = sk
            self.systematics[sk]['label'] = sk
            self.systematics[sk]['description'] = ''
            self.systematics[sk]['variation'] = 'summary'
            self.systematics[sk]['order'] = None
            
            # Initialize covariance matrices
            n_bins = len(self.bins) - 1

            self.systematics[sk]['event_cov'] = np.zeros((n_bins, n_bins))
            self.systematics[sk]['event_cov_unaltered'] = np.zeros((n_bins, n_bins))
            if self.sigma_tilde is not None:
                self.systematics[sk]['xsec_cov'] = np.zeros((n_bins, n_bins))
                self.systematics[sk]['xsec_cov_unaltered'] = np.zeros((n_bins, n_bins))
                self.systematics[sk]['xsec_fraccov'] = np.zeros((n_bins, n_bins))
            else:
                self.systematics[sk]['xsec_cov'] = None
                self.systematics[sk]['xsec_cov_unaltered'] = None
                self.systematics[sk]['xsec_fraccov'] = None
            self.systematics[sk]['event_fraccov'] = np.zeros((n_bins, n_bins))
        
        #Print unique types
        #print(f'Combine summaries for variable: {self.variable_name}')
        #print(f'Unique types: {list(set([sys_dict["type"] for sys_dict in self.systematics.values()]))}')
        #print(f'Summary keys: {summary_keys}')

        # Sum covariance matrices (event and xsec cov rescaled to selected scale so different-scale
        # systematics combine correctly)
        target_cv = self.sel + self.sel_background
        target_xsec_cv = self.sigma_tilde
        for key, sys_dict in self.systematics.items():
            if sys_dict['variation'] == 'summary' or sys_dict['variation'] == 'self':
                continue

            for sk in summary_keys:
                if (sys_dict['type'] == sk) or (sk == 'total'):
                    if sys_dict['event_fraccov'] is not None:
                        self.systematics[sk]['event_fraccov'] += sys_dict['event_fraccov']
                    if sys_dict['event_cov'] is not None:
                        self.systematics[sk]['event_cov'] += sys_dict['event_cov']
                    if sys_dict['event_cov_unaltered'] is not None:
                        self.systematics[sk]['event_cov_unaltered'] += sys_dict['event_cov_unaltered']
                    if sys_dict['xsec_fraccov'] is not None:
                        if self.systematics[sk]['xsec_fraccov'] is None:
                            print(self.systematics)
                            raise ValueError(f'xsec_fraccov is None for {sk}')
                        self.systematics[sk]['xsec_fraccov'] += sys_dict['xsec_fraccov']
                    if sys_dict['xsec_cov'] is not None:
                        self.systematics[sk]['xsec_cov'] += sys_dict['xsec_cov']
                    if sys_dict['xsec_cov_unaltered'] is not None:
                        self.systematics[sk]['xsec_cov_unaltered'] += sys_dict['xsec_cov_unaltered']
        # Compute fractional uncertainties and correlations from combined covariances
        # (combined event and xsec cov are on selected scale, so use target_cv / target_xsec_cv)
        for sk in summary_keys:
            event_cv = target_cv
            xsec_cv = target_xsec_cv

            # Event fractional uncertainty
            if event_cv is not None:
                self.systematics[sk]['event_fracunc'] = get_fractional_uncertainty(
                    event_cv, self.systematics[sk]['event_cov']
                )
            
            # Xsec fractional uncertainty (only if sigma_tilde exists)
            if xsec_cv is not None and self.systematics[sk]['xsec_cov'] is not None:
                self.systematics[sk]['xsec_fracunc'] = get_fractional_uncertainty(
                    xsec_cv, self.systematics[sk]['xsec_cov']
                )
            
            # Event correlation
            self.systematics[sk]['event_corr'], _ = construct_correlation_matrix(
                self.systematics[sk]['event_cov']
            )
            
            # Xsec correlation (only if xsec_cov exists)
            if self.systematics[sk]['xsec_cov'] is not None:
                self.systematics[sk]['xsec_corr'], _ = construct_correlation_matrix(
                    self.systematics[sk]['xsec_cov']
                )
            
            # Event total uncertainty
            if event_cv is not None and self.systematics[sk]['event_fracunc'] is not None:
                self.systematics[sk]['event_totalunc'] = get_total_unc(
                    event_cv, self.systematics[sk]['event_fracunc']
                )
            
            # Xsec total uncertainty (only if sigma_tilde exists)
            if xsec_cv is not None and self.systematics[sk]['xsec_fracunc'] is not None:
                self.systematics[sk]['xsec_totalunc'] = get_total_unc(
                    xsec_cv, self.systematics[sk]['xsec_fracunc']
                )
    
    def generate_description_template(self, save_dir=None, descriptions_path=None,
                                       exclude_keys=None, include_keys=None,
                                       key_order=None):
        """
        Generate a rich template dictionary for systematics with metadata.
        
        Parameters
        ----------
        save_dir : str, optional
            Directory to save the template JSON file.
        descriptions_path : str, optional
            Path to a JSON file mapping systematic names to description strings.
            Descriptions found in this file override any already stored on the object.
        exclude_keys : list, optional
            Keys to exclude from the template.
        include_keys : list, optional
            If provided, only include these keys.
        key_order : list, optional
            Ordered list of keys. Each key's ``order`` field is set to its
            position in this list. Keys not in the list keep their default
            order (appended after the explicit ones).
            
        Returns
        -------
        dict
            Keyed by systematic key. Values are dicts with: name, variation,
            description, event_rate_unc, xsec_unc, order.
        """
        desc_lookup = {}
        if descriptions_path is not None:
            with open(descriptions_path, 'r') as f:
                desc_lookup = json.load(f)

        order_lookup = {}
        if key_order is not None:
            for i, k in enumerate(key_order):
                order_lookup[k] = i

        template = {}
        next_order = len(order_lookup)
        seen_names = {}
        for key, sys_dict in self.systematics.items():
            name = sys_dict.get('name', key)
            seen_names.setdefault(name, []).append(key)

        next_order = len(order_lookup)
        for key, sys_dict in self.systematics.items():
            if sys_dict.get('variation') in ('self', None):
                continue
            if exclude_keys and key in exclude_keys:
                continue
            if include_keys and key not in include_keys:
                continue

            name = sys_dict.get('name', key)
            stype = sys_dict.get('type', '')
            if len(seen_names.get(name, [])) > 1 and stype:
                display_name = f'{name} ({stype})'
            else:
                display_name = name
            description = desc_lookup.get(name, desc_lookup.get(key, sys_dict.get('description', '')))

            event_unc = sys_dict.get('event_totalunc', None)
            xsec_unc = sys_dict.get('xsec_totalunc', None)

            if key in order_lookup:
                order = order_lookup[key]
            elif sys_dict.get('order') is not None:
                order = sys_dict['order']
            else:
                order = next_order
                next_order += 1

            template[key] = {
                'name': display_name,
                'variation': sys_dict.get('variation', ''),
                'description': description,
                'event_rate_unc': float(event_unc) if event_unc is not None and not hasattr(event_unc, '__len__') else event_unc,
                'xsec_unc': float(xsec_unc) if xsec_unc is not None and not hasattr(xsec_unc, '__len__') else xsec_unc,
                'order': order,
            }
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            template_path = os.path.join(save_dir, 'description_template.json')
            with open(template_path, 'w') as f:
                json.dump(template, f, indent=2, default=str)
            print(f'Description template saved to {template_path}')
        
        return template
    
    def _per_bin_event_rate_unc(self, sys_dict, base_key, use_fracunc, default_cv):
        """
        Per-bin fractional uncertainty or sqrt(diag fraccov) for event or xsec rate plots.

        Parameters
        ----------
        sys_dict : dict
            Single systematic entry from self.systematics.
        base_key : str
            'event' or 'xsec'.
        use_fracunc : bool
            If True, return ``{base_key}_fracunc``; else convert from ``{base_key}_fraccov``.
        default_cv : np.ndarray
            Fallback CV for covariance_from_fraccov when _get_cv_for_type returns None.

        Returns
        -------
        np.ndarray or None
            Per-bin uncertainty, or None if missing.
        """
        if use_fracunc:
            return sys_dict.get(f'{base_key}_fracunc')
        cv_key = 'sigma_tilde' if base_key == 'xsec' else 'sel'
        target_cv = self._get_cv_for_type(sys_dict.get('type'), cv_key=cv_key, sys_dict=sys_dict)
        if target_cv is None:
            target_cv = default_cv
        return np.sqrt(np.diag(covariance_from_fraccov(sys_dict.get(f'{base_key}_fraccov'), target_cv)))

    _COLORABLE_VARIATIONS = frozenset({'multisim', 'multisigma', 'unisim', 'summary', 'unknown'})

    def set_colors(self):
        """
        Assign plot colors for physics systematics and summary rows (Glasbey via colorcet).
        Summary key ``total`` is always black; does not consume a palette slot.
        Other variation types (stat, flat, self) are left with ``color`` None.
        """
        import colorcet as cc
        import seaborn as sns

        eligible = sorted(
            k
            for k, d in self.systematics.items()
            if d.get('variation') in Systematics._COLORABLE_VARIATIONS
        )
        total_is_summary = (
            'total' in self.systematics
            and self.systematics['total'].get('variation') == 'summary'
        )
        palette_keys = sorted(k for k in eligible if not (total_is_summary and k == 'total'))
        if total_is_summary:
            self.systematics['total']['color'] = 'red'
        if len(palette_keys) == 0:
            return
        palette = sns.color_palette(cc.glasbey, n_colors=len(palette_keys))
        for i, k in enumerate(palette_keys):
            self.systematics[k]['color'] = palette[i]
    
    def plot_event_rate_errs(self, base_key, exclude_keys=[], include_keys=[],
                            include_types=[], exclude_types=[], include_variations=[], exclude_variations=[],
                            xlabel='', max_uncs=None, fig=None, ax=None, sort=False, use_fracunc=True):
        """
        Plot the event rate uncertainties for a given base key.
        
        Parameters
        ----------
        base_key : str
            The base key to plot (e.g., 'event', 'xsec')
        exclude_keys : list
            A list of keys to exclude from the plot
        include_keys : list
            List of keys to include, if empty use all keys (except exclude keys)
        include_types : list
            List of systematic types to include, if empty use all types
        exclude_types : list
            List of systematic types to exclude
        include_variations : list
            List of systematic variations to include, if empty use all variations
        exclude_variations : list
            List of systematic variations to exclude
        xlabel : str
            The x-axis label
        max_uncs : int, optional
            The maximum number of uncertainties to plot
        fig : matplotlib.figure.Figure, optional
            Figure to use (creates new if None)
        ax : matplotlib.axes.Axes, optional
            Axes to use (creates new if None)
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure
        ax : matplotlib.axes.Axes
            The axes
        """
        # Use a sensible default xlabel if caller did not provide one
        if not xlabel:
            xlabel = self.xlabel

        # Filter to valid keys for this plot only (do not mutate self.systematics)
        valid_keys = []
        for key, sys_dict in self.systematics.items():
            if sys_dict.get(f'{base_key}_totalunc') is not None:
                valid_keys.append(key)
            #else:
            #    print(f'WARNING: {key} for {self.variable_name} {base_key} has no total uncertainty, skipping...')
        systematics_to_plot = {k: v for k, v in self.systematics.items() if k in valid_keys}
        if sort:
            # Sort systematics by total uncertainty (largest first)
            sorted_items = sorted(
                systematics_to_plot.items(),
                key=lambda x: x[1].get(f'{base_key}_totalunc', 0),
                reverse=True
            )
        else:
            sorted_items = systematics_to_plot.items()
        if len(sorted_items) == 0:
            print(f'WARNING: No systematics ({self.variable_name}) found for {base_key} skipping event rate error plot...')
            return [fig,ax,None]
        
        if max_uncs is None:
            max_uncs = len(sorted_items)
        
        if len(include_keys) == 0:
            include_keys = [item[0] for item in sorted_items]
            if max_uncs == len(sorted_items):
                max_uncs = len(include_keys)
        else:
            max_uncs = len(include_keys)
        
        unc_colors = plotters.get_colors('gist_rainbow', max_uncs + 1)
        #if exclude_keys is not None:
        #    print(f'Using keys: {include_keys}')
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        if base_key == 'event':
            default_cv = self.sel + self.sel_background
        elif base_key == 'xsec':
            default_cv = self.sigma_tilde
        else:
            raise ValueError(f'Invalid base key: {base_key}')
        cnt = 0
        for i, (sys_name, sys_dict) in enumerate(sorted_items):
            if cnt >= max_uncs:
                break
            if sys_name in exclude_keys or sys_name not in include_keys:
                continue
            sys_type = sys_dict.get('type')
            if sys_type in exclude_types:
                continue
            if len(include_types) != 0 and sys_type not in include_types:
                continue
            if sys_dict.get('variation') in exclude_variations:
                continue
            if len(include_variations) != 0 and sys_dict.get('variation') not in include_variations:
                continue
            cnt += 1
            c = sys_dict.get('color')
            if c is None:
                c = unc_colors[cnt]
            total_unc = sys_dict.get(f'{base_key}_totalunc', 0)
            unc = self._per_bin_event_rate_unc(sys_dict, base_key, use_fracunc, default_cv)
            
            if unc is None:
                continue
            label = f"{sys_dict['label']} ({total_unc*100:.1f}%)"
            makeplot.plot_hist_edges(
                self.bins, unc, None, ax=ax, label=label, color=c
            )
        
        # Redo max_unc
        max_uncs = max(1, min(max_uncs, len(ax.get_lines())))

        if use_fracunc:
            ylabel_type = ' Uncertainty'
        else:
            ylabel_type = r' $\sigma$'
        if 'xsec' in base_key:
            ax.set_ylabel(r'Cross Section '+ylabel_type)
        else:
            ax.set_ylabel(r'Event Rate '+ylabel_type)
        if self.variable_name == 'momentum':
            ax.set_xlim(0, MAX_PMOM)
        if self.variable_name == 'opt0':
            ax.set_xscale('log')

        ax.set_xlabel(xlabel)
        ax.legend(ncol=int(np.ceil(max_uncs/18)),loc='upper left',bbox_to_anchor=(1.02, 1.0))
        
        return [fig,ax,None]
    
    def plot_event_rate_errs_2d(
        self,
        binning2d,
        base_key,
        exclude_keys=[],
        include_keys=[],
        include_types=[],
        exclude_types=[],
        include_variations=[],
        exclude_variations=[],
        xlabel='',
        max_uncs=None,
        fig=None,
        axs=None,
        ncols=3,
        sort=False,
        use_fracunc=True,
        panel_label_prefix='',
        legend=True,
    ):
        """
        Plot per-bin event or xsec rate uncertainties on a cosθ by momentum panel grid.

        Same filtering and per-bin uncertainty logic as plot_event_rate_errs, but each
        cosθ slice is drawn as a step histogram vs momentum using the Binning2D geometry.

        Parameters
        ----------
        binning2d : Binning2D
            Must match how differential bins were defined: same diff_costheta_bins,
            diff_momentum_bins_2d, and keep_null as used for this Systematics instance
            so len(differential_centers) matches len(per-bin unc arrays).
        base_key : str
            'event' or 'xsec'.
        exclude_keys, include_keys, include_types, exclude_types,
        include_variations, exclude_variations, max_uncs, sort, use_fracunc
            Same as plot_event_rate_errs.
        xlabel : str
            Momentum axis label; default is reconstructed p_mu if empty.
        fig, axs : optional
            Figure and axes grid; if both None, a grid with ncols columns is created.
        ncols : int
            Number of columns in the panel grid (default 3).
        panel_label_prefix : str
            Prepended to each panel cosθ range label.
        legend : bool
            If True, draw the legend on ``axs[0, 2]`` when the grid has at least three columns
            (handles come from the first cosθ panel, where step labels are set).

        Returns
        -------
        list
            [fig, axs, None] for consistency with plot_event_rate_errs.
        """
        if not isinstance(binning2d, Binning2D):
            raise TypeError('binning2d must be a Binning2D instance')

        # Default momentum label (1D plot uses self.xlabel which is often bin ID for differential)
        if not xlabel:
            xlabel = r'Reconstructed $p_{\mu}$ [GeV/c]'

        valid_keys = []
        for key, sys_dict in self.systematics.items():
            if sys_dict.get(f'{base_key}_totalunc') is not None:
                valid_keys.append(key)
        systematics_to_plot = {k: v for k, v in self.systematics.items() if k in valid_keys}
        if sort:
            sorted_items = sorted(
                systematics_to_plot.items(),
                key=lambda x: x[1].get(f'{base_key}_totalunc', 0),
                reverse=True,
            )
        else:
            sorted_items = systematics_to_plot.items()
        if len(sorted_items) == 0:
            print(
                f'WARNING: No systematics ({self.variable_name}) found for {base_key} '
                'skipping 2d event rate error plot...'
            )
            return [fig, axs, None]

        if max_uncs is None:
            max_uncs = len(sorted_items)

        if len(include_keys) == 0:
            include_keys = [item[0] for item in sorted_items]
            if max_uncs == len(sorted_items):
                max_uncs = len(include_keys)
        else:
            max_uncs = len(include_keys)

        unc_colors = plotters.get_colors('gist_rainbow', max_uncs + 1)
        n_ct_bins = binning2d.n_costheta_bins

        if fig is None and axs is None:
            nrows = int(np.ceil(n_ct_bins / ncols))
            fig, axs = plt.subplots(figsize=(12, 8), nrows=nrows, ncols=ncols)
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        axs_flat = axs.flatten()
        if len(axs_flat) < n_ct_bins:
            raise ValueError(
                f'Not enough axes for costheta bins: need {n_ct_bins}, got {len(axs_flat)}'
            )

        if base_key == 'event':
            default_cv = self.sel + self.sel_background
        elif base_key == 'xsec':
            default_cv = self.sigma_tilde
        else:
            raise ValueError(f'Invalid base key: {base_key}')

        cnt = 0
        for i, (sys_name, sys_dict) in enumerate(sorted_items):
            if cnt >= max_uncs:
                break
            if sys_name in exclude_keys or sys_name not in include_keys:
                continue
            sys_type = sys_dict.get('type')
            if sys_type in exclude_types:
                continue
            if len(include_types) != 0 and sys_type not in include_types:
                continue
            if sys_dict.get('variation') in exclude_variations:
                continue
            if len(include_variations) != 0 and sys_dict.get('variation') not in include_variations:
                continue
            cnt += 1
            c = sys_dict.get('color')
            if c is None:
                c = unc_colors[cnt]
            total_unc = sys_dict.get(f'{base_key}_totalunc', 0)
            unc = self._per_bin_event_rate_unc(sys_dict, base_key, use_fracunc, default_cv)
            if unc is None:
                continue
            unc = np.asarray(unc).ravel()
            n_expect = len(binning2d.differential_centers)
            if len(unc) != n_expect:
                raise ValueError(
                    f'Per-bin unc length {len(unc)} != Binning2D.differential_centers length {n_expect}. '
                    'Pass a Binning2D consistent with this Systematics differential binning (including keep_null).'
                )
            hist_dict, _ = binning2d.bin_differential_dict_binned(unc, bin_by='costheta')
            label = f"{sys_dict['label']} ({total_unc*100:.1f}%)"
            for ct_idx in range(n_ct_bins):
                if ct_idx not in hist_dict:
                    continue
                values, mom_edges = hist_dict[ct_idx][0], hist_dict[ct_idx][1]
                if values is None or len(values) == 0:
                    continue
                makeplot.plot_hist_edges(
                    mom_edges,
                    values,
                    None,
                    ax=axs_flat[ct_idx],
                    label=label if ct_idx == 0 else None,
                    color=c,
                )

        for ct_idx in range(n_ct_bins):
            ct_lo = binning2d.diff_costheta_bins[ct_idx]
            ct_hi = binning2d.diff_costheta_bins[ct_idx + 1]
            panel_label = (
                f'{panel_label_prefix}{ct_lo:.2f} <' + r'$\cos\theta_\mu$' + f' < {ct_hi:.2f}'
            )
            plotters.add_label(
                axs_flat[ct_idx],
                panel_label,
                where='centerright',
                fontsize=10,
                color='black',
                alpha=0.9,
            )
            axs_flat[ct_idx].set_xlim(0, MAX_PMOM)

        max_uncs = max(1, min(max_uncs, len(axs_flat[0].get_lines())))

        if use_fracunc:
            ylabel_type = ' Uncertainty'
        else:
            ylabel_type = r' $\sigma$'
        if 'xsec' in base_key:
            ylabel_text = r'Cross Section ' + ylabel_type
        else:
            ylabel_text = r'Event Rate ' + ylabel_type

        if axs.ndim == 2 and axs.shape[0] > 1:
            axs[1, 0].set_ylabel(ylabel_text)
        else:
            axs_flat[0].set_ylabel(ylabel_text)

        if axs.ndim == 2:
            bottom_ax = axs[-1, axs.shape[1] // 2]
        else:
            bottom_ax = axs_flat[-1]
        bottom_ax.set_xlabel(xlabel)

        if legend:
            # Lines and labels live on axs_flat[0]; place legend on axs[0,2] with explicit handles
            handles, leg_labels = axs_flat[0].get_legend_handles_labels()
            if handles:
                ncol = int(np.ceil(max_uncs / 18))
                axs[0, 2].legend(
                    handles,
                    leg_labels,
                    ncol=ncol,
                    loc='upper left',
                    bbox_to_anchor=(1.02, 1.0),
                )
            

        return [fig, axs, None]

    def plot_all_covariance_matrices(self, plot_dir=None, save_plots=True, progress_bar=False, suffix='', keys=None, **kwargs):
        """
        Plot all covariance matrices.
        
        Parameters
        ----------
        plot_dir : str, optional
            Directory to save plots
        save_plots : bool
            Whether to save plots
        keys : list, optional
            List of keys to plot, if None plot all keys
        """
        if keys is None:
            systematics_to_plot = self.systematics
        else:
            for key in keys:
                if key not in self.systematics:
                    raise ValueError(f"Systematic key '{key}' not found")
            systematics_to_plot = {k: v for k, v in self.systematics.items() if k in keys}
        if progress_bar:
            pbar = tqdm(systematics_to_plot.items(), unit=' goomba')
        else:
            pbar = systematics_to_plot.items()
        for key, _ in pbar:
            _ = self.plot_covariance_matrices(key, plot_dir, save_plots, suffix, **kwargs)
    
    def plot_covariance_matrices(self, sys_key, plot_dir=None, save_plots=True, suffix='', include_xsec_cov=True, 
        include_event_cov=True, histtypes=None):
        """
        Plot covariance, fractional covariance, and correlation matrices.
        
        Parameters
        ----------
        sys_key : str
            The systematic key to plot
        plot_dir : str, optional
            Directory to save plots
        save_plots : bool
            Whether to save plots
        include_xsec_cov : bool, optional
            Whether to include xsec covariance
        include_event_cov : bool, optional
            Whether to include event covariance
        histtypes : list, optional
            List of histogram types to plot, if None plot all types
        Returns
        -------
        figs : dict
            Dictionary of figures for each matrix type
        """
        if sys_key not in self.systematics:
            raise ValueError(f"Systematic key '{sys_key}' not found")
        
        sys_dict = self.systematics[sys_key]
        figs = {}
        
        hist2dkeys = [k for k in sys_dict.keys() if 'corr' in k or 'cov' in k]
        
        for plotkey in hist2dkeys:
            #Condition to plot only certain histogram types
            if histtypes is not None:
                for histtype in histtypes:
                    if histtype not in plotkey:
                        continue
            # Condition to plot the xsec or event covariance
            if not include_xsec_cov and plotkey.startswith('xsec'):
                continue
            if not include_event_cov and plotkey.startswith('event'):
                continue
            hist2d = sys_dict[plotkey]

            if hist2d is None:
                continue
                
            if not isinstance(hist2d, np.ndarray):
                print(f'WARNING: {plotkey} for {sys_key} / {self.variable_name} is not a numpy array, skipping...')
                print(f'Type: {type(hist2d)}')
                continue
            if hist2d.ndim != 2 or 0 in hist2d.shape:
                print(f'WARNING: {plotkey} for {sys_key} / {self.variable_name} is not a 2D array, skipping...')
                print(f'Shape: {hist2d.shape}')
                continue


            #Get hist2d, but replace Nan or Inf with 0
            hist2d = np.nan_to_num(hist2d)
            hist2d = np.where(np.isinf(hist2d), 0, hist2d)
            hist2d = np.where(np.abs(hist2d) == np.finfo(np.double).max, 0, hist2d)
            #Skip if all values are 0
            if np.all(hist2d == 0):
                continue
            # Compute valid vmin/vmax for colorbar to avoid NaN/Inf limits
            vmin = np.nanmin(hist2d)
            vmax = np.nanmax(hist2d)
            # Handle edge case where all values are the same or array is empty
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin = 0
                vmax = 1
            
            # Determine matrix type and colormap
            if 'fraccov' in plotkey:
                cmap = 'viridis'
                subfolder = 'fraccov'
            elif 'inv_cov' in plotkey:
                cmap = 'summer'
                subfolder = 'inv_cov'
            elif 'cov' in plotkey:
                cmap = 'inferno'
                subfolder = 'cov'
            elif 'corr' in plotkey:
                cmap = 'cividis'
                subfolder = 'corr'
                vmin, vmax = -1, 1
            else:
                continue
            
            if 'unaltered' in plotkey:
                continue
            
            # Determine variable and bins based on variable_name
            set_axlims = self.variable_name == 'momentum'
            axlabel = self.xlabel
            
            bins = self.bins
            
            title = f"{sys_dict['name']} {self.variable_name} {plotkey.replace('_', ' ')}"
            variation = sys_dict["variation"]
            if variation == None:
                name = title.replace(' ', '_')
            else:
                name = title.replace(' ', '_')+f'_{variation}'
            
            fig, ax = plt.subplots()
            ax.set_title(title)

            im = ax.pcolormesh(bins, bins, hist2d, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel(axlabel)
            ax.set_ylabel(axlabel)
            fig.colorbar(im, ax=ax)
            if set_axlims:
                ax.set_xlim(0, MAX_PMOM)
                ax.set_ylim(0, MAX_PMOM)
            
            if save_plots:
                if plot_dir is not None:
                    plotters.save_plot(fname=f'{name}{suffix}', fig=fig, folder_name=f'{plot_dir}/{subfolder}')
                else:
                    raise ValueError('plot_dir is None and save_plots is True')
                plt.close(fig)
            else:
                plt.show()
            
            figs[plotkey] = [fig,ax,im]
        
        return figs
    
    def plot_all_distributions(self, keys=None, exclude_keys=[], include_keys=[], suffix='', **kwargs):
        """
        Plot all distributions for all systematics.
        """
        if keys is None:
            keys = self.systematics.keys()
        else:
            for key in keys:
                if key not in self.systematics:
                    raise ValueError(f"Systematic key '{key}' not found")
        figs = {}
        for sys_key in keys:
            if sys_key in exclude_keys:
                continue
            if self.systematics[sys_key].get('variation') == 'self':
                continue
            figs[sys_key] = self.plot_distributions(sys_key, suffix=suffix, **kwargs)
        return figs
    
    def plot_distributions(self, sys_key, plot_key='sel', plot_dir=None, save_plots=True,
                           scale_by_xsec_unit=False, suffix=''):
        """
        Plot sigma_tilde or sel distributions for CV and all universe variations.
        
        Parameters
        ----------
        sys_key : str
            The systematic key to plot
        plot_key : str, optional
            The key to plot, either 'sigma_tilde' or 'sel'
        plot_dir : str, optional
            Directory to save plots
        save_plots : bool
            Whether to save plots
        scale_by_xsec_unit : bool, optional
            Undo scaling by xsec unit to plot the true xsec.
            Default is False.
        
        Returns
        -------
        figs : dict
            Dictionary of figures for each variable
        """
        if sys_key not in self.systematics:
            raise ValueError(f"Systematic key '{sys_key}' not found")
        
        sys_dict = self.systematics[sys_key]
        figs = {}
        
        # Check for sigma_tilde key
        if plot_key not in sys_dict or sys_dict[plot_key] is None or len(sys_dict[plot_key]) == 0:
            return figs
        
        set_xlim = self.variable_name == 'momentum'
        xscale = 'linear'
        if self.variable_name == 'opt0':
            xscale = 'log'
        axlabel = self.xlabel
        
        bins = self.bins
        hist_var = self._get_cv_for_type(sys_dict.get('type'), cv_key=plot_key, sys_dict=sys_dict)
        if hist_var is not None:
            hist_var = hist_var.copy()
        else:
            return figs
        if scale_by_xsec_unit:
            hist_var /= self.xsec_unit

        if plot_key == 'sigma_tilde':
            y_label = r'$\tilde{\sigma}$'
        elif plot_key == 'sel':
            y_label = r'Selected Candidates'
        else:
            raise ValueError(f'Invalid plot_key: {plot_key}')
        
        title = f"{sys_dict['name']} {plot_key.replace('_', ' ')} {self.variable_name}"
        name = title.replace(' ', '_')+f'_{sys_dict["variation"]}'
        
        fig, ax = plt.subplots()
        # Filter out None values for alpha calculation
        valid_vars = [v for v in sys_dict[plot_key] if v is not None]
        if len(valid_vars) > 0:
            # Use fixed alpha for variations to avoid stacking artifacts
            var_alpha = max(0.3, 1./len(valid_vars))
        else:
            var_alpha = 0.3
        if len(np.shape(sys_dict[plot_key])) == 1:
            var_plots = [sys_dict[plot_key]]
        else:
            var_plots = sys_dict[plot_key]
        for i,var in enumerate(var_plots):
            var = var.copy()
            if var is None:
                continue
            if not isinstance(var, np.ndarray):
                continue
                #raise ValueError(f'var is not a numpy array: {var} for {sys_key} {plot_key}')
            if i == 0:
                label = 'Variation(s)'
            else:
                label = None
            # Scale by xsec unit if desired
            if scale_by_xsec_unit:
                var /= self.xsec_unit
            makeplot.plot_hist_edges(
                bins, var, None, ax=ax, color='blue', label=label,
                alpha=var_alpha, linewidth=1
            )
        makeplot.plot_hist_edges(
            bins, hist_var, None, ax=ax, color='black', label='CV', alpha=1., linewidth=2
        )
        ax.legend()
        ax.set_xlabel(axlabel)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        if set_xlim:
            ax.set_xlim(0, MAX_PMOM)
        ax.set_xscale(xscale)
        if save_plots:
            if plot_dir is not None:
                plotters.save_plot(fname=f'{name}{suffix}', fig=fig, folder_name=plot_dir)
            else:
                raise ValueError('plot_dir is None and save_plots is True')
            plt.close(fig)
        else:
            plt.show()
        
        figs[plot_key] = [fig,ax,None]
        
        return figs

    def copy(self):
        """
        Copy the systematics object
        
        Returns
        -------
        Systematics
            A deep copy of the systematics object
        """
        # Create a new instance with copied attributes
        new_obj = Systematics.__new__(Systematics)
        
        # Deep copy all numpy arrays and basic attributes
        new_obj.variable_name = self.variable_name
        new_obj.bins = np.copy(self.bins)
        new_obj.xsec_unit = self.xsec_unit
        
        new_obj.sigma_tilde = np.copy(self.sigma_tilde) if self.sigma_tilde is not None else None
        new_obj.sel = np.copy(self.sel)
        new_obj.sel_background = np.copy(self.sel_background)
        new_obj.sel_data = np.copy(self.sel_data) if self.sel_data is not None else None
        new_obj.sel_truth = np.copy(self.sel_truth) if self.sel_truth is not None else None
        new_obj.sel_background_truth = np.copy(self.sel_background_truth) if self.sel_background_truth is not None else None
        new_obj.sig_truth = np.copy(self.sig_truth) if self.sig_truth is not None else None
        new_obj.response = np.copy(self.response) if self.response is not None else None
        new_obj.smearing = np.copy(self.smearing) if self.smearing is not None else None
        new_obj.eff_truth = np.copy(self.eff_truth) if self.eff_truth is not None else None
        new_obj.eff_reco = np.copy(self.eff_reco) if self.eff_reco is not None else None
        
        # Copy stored distributions
        new_obj._true_sig = np.copy(self._true_sig) if self._true_sig is not None else None
        new_obj._true_sel = np.copy(self._true_sel) if self._true_sel is not None else None
        new_obj._true_sel_background = np.copy(self._true_sel_background) if self._true_sel_background is not None else None
        new_obj._reco_sel = np.copy(self._reco_sel)
        new_obj._reco_sel_background = np.copy(self._reco_sel_background)
        new_obj._genweights_sig = np.copy(self._genweights_sig)
        new_obj._genweights_sel = np.copy(self._genweights_sel)
        new_obj._genweights_sel_background = np.copy(self._genweights_sel_background)
        
        # Deep copy the systematics dict (contains nested dicts with numpy arrays and lists)
        new_obj.systematics = copy.deepcopy(self.systematics)
        
        return new_obj

    def save(self, save_dir, save_keys=None, metadata_dir='metadata'):
        """
        Save all the systematics to folders depending on the key.
        The file name depends on covariance, fractional uncertainty, correlation, etc.

        Parameters
        ----------
        save_dir : str
            Directory to save the systematics
        save_keys : list, optional
            List of keys to save, if None save all the variable quantities
        metadata_dir : str, optional
            Directory to save the metadata to
        """
        if save_keys is None:
            #Save all the variable quantities by default
            if len(self.systematics) > 0:
                _sdict = self.systematics[list(self.systematics.keys())[0]]
                save_keys = list(_sdict.keys())
            else:
                # Use template keys if no systematics yet
                save_keys = list(Systematics._get_dict_template().keys())
        print(f'Saving {len(save_keys)} keys to {save_dir}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metadata needed for reconstruction
        metadata_dir = os.path.join(save_dir, metadata_dir)
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Save variable bins
        np.savetxt(os.path.join(metadata_dir, f'{self.variable_name}_bins.csv'), self.bins)
        
        # Save CV data (only if not None)
        np.savetxt(os.path.join(metadata_dir, f'{self.variable_name}_sel.csv'), self.sel)
        np.savetxt(os.path.join(metadata_dir, f'{self.variable_name}_sel_background.csv'), self.sel_background)
        if self.sel_data is not None:
            np.savetxt(os.path.join(metadata_dir, f'{self.variable_name}_sel_data.csv'), self.sel_data)
        
        if self.sigma_tilde is not None:
            np.savetxt(os.path.join(metadata_dir, f'{self.variable_name}_sigma_tilde.csv'), self.sigma_tilde)
        if self.sel_truth is not None:
            np.savetxt(os.path.join(metadata_dir, f'{self.variable_name}_sel_truth.csv'), self.sel_truth)
        if self.sel_background_truth is not None:
            np.savetxt(os.path.join(metadata_dir, f'{self.variable_name}_sel_background_truth.csv'), self.sel_background_truth)
        if self.sig_truth is not None:
            np.savetxt(os.path.join(metadata_dir, f'{self.variable_name}_sig_truth.csv'), self.sig_truth)
        if self.response is not None:
            np.savetxt(os.path.join(metadata_dir, f'{self.variable_name}_response.csv'), self.response)
        if self.smearing is not None:
            np.savetxt(os.path.join(metadata_dir, f'{self.variable_name}_smearing.csv'), self.smearing)
        if self.eff_truth is not None:
            np.savetxt(os.path.join(metadata_dir, f'{self.variable_name}_eff_truth.csv'), self.eff_truth)
        if self.eff_reco is not None:
            np.savetxt(os.path.join(metadata_dir, f'{self.variable_name}_eff_reco.csv'), self.eff_reco)
        if self.sel_data is not None:
            np.savetxt(os.path.join(metadata_dir, f'{self.variable_name}_sel_data.csv'), self.sel_data)

        # Save xsec_unit (only if not None)
        if self.xsec_unit is not None:
            np.savetxt(os.path.join(metadata_dir, 'xsec_unit.csv'), np.array([self.xsec_unit]))
        
        # Save sys_dict structure (convert tuples to lists for JSON serialization)
        sys_dict_serializable = {}
        for key, val in self.systematics.items():
            if val['cols'] is None:
                cols = None
            else:
                cols = [list(col) if isinstance(col, tuple) else col for col in val['cols']]
            sys_dict_serializable[key] = {
                'cols': cols,
                'col_names': val['col_names'],
                'type': val['type'],
                'name': val['name'],
                'label': val.get('label', val.get('name', key)),
                'description': val.get('description', ''),
                'variation': val['variation'],
                'order': val.get('order', None)
            }
        with open(os.path.join(metadata_dir, 'sys_dict.json'), 'w') as f:
            json.dump(sys_dict_serializable, f, indent=2)
        
        # Save systematics data (folder per dict key, not name)
        metadata_keys = {'name', 'type', 'variation', 'label', 'description',
                         'order', 'cols', 'col_names', 'rank', 'color'}
        for key, sdict in self.systematics.items():
            subfolder = key
            for k, arr_like in sdict.items():
                if k in metadata_keys:
                    continue
                if k not in save_keys:
                    continue
                if arr_like is None:
                    continue
                # Skip empty lists/arrays
                if isinstance(arr_like, (list, np.ndarray)) and len(arr_like) == 0:
                    continue
                # Skip lists that contain only None values
                if isinstance(arr_like, list) and all(item is None for item in arr_like):
                    continue
                os.makedirs(f'{save_dir}/{subfolder}', exist_ok=True)
                # Prefix filename with variable_name
                filename_prefix = f'{self.variable_name}_'
                # Per-systematic response matrices (2D cv or 3D stack): .npy only
                if k in ('response', 'cv_response'):
                    if k == 'cv_response':
                        to_save = np.asarray(arr_like)
                    else:
                        if isinstance(arr_like, list):
                            if all(x is not None for x in arr_like):
                                to_save = np.stack([np.asarray(x) for x in arr_like])
                            else:
                                to_save = np.asarray(arr_like, dtype=object)
                        else:
                            to_save = np.asarray(arr_like)
                    np.save(
                        f'{save_dir}/{subfolder}/{filename_prefix}{k}.npy',
                        to_save)
                    continue
                # This is a numpy array or list
                if isinstance(arr_like, (np.ndarray, list)) and k not in ['cols', 'col_names']:
                    #Make sure the array is a numpy array
                    arr_like = np.array(arr_like)
                    # Check if array is empty after conversion
                    if arr_like.size == 0:
                        continue
                    if arr_like.ndim == 0:
                        # Scalar numpy array - save as text
                        with open(f'{save_dir}/{subfolder}/{filename_prefix}{k}.txt', 'w') as f:
                            f.write(str(arr_like.item()))
                    else:
                        #Save the array
                        np.savetxt(f'{save_dir}/{subfolder}/{filename_prefix}{k}.csv', arr_like)
                #This is a list of strings (like a list of keys dumbby)
                elif isinstance(arr_like, list):
                    with open(f'{save_dir}/{subfolder}/{filename_prefix}{k}.txt', 'w') as f:
                        for item in arr_like:
                            f.write(f'{item}\n')
                #This is a string
                elif isinstance(arr_like, str):
                    with open(f'{save_dir}/{subfolder}/{filename_prefix}{k}.txt', 'w') as f:
                        f.write(arr_like)
                #This is a scalar (float, int, numpy scalar)
                elif isinstance(arr_like, (int, float, np.number)):
                    with open(f'{save_dir}/{subfolder}/{filename_prefix}{k}.txt', 'w') as f:
                        f.write(str(arr_like))
                else:
                    raise ValueError(f'Array {k} has type {type(arr_like)}, only numpy arrays, lists of strings, strings, and scalars are supported')
                        
    @classmethod
    def from_saved(cls, load_dir, var_name, metadata_dir='metadata_detsys',
                   ignore_keys=[], ignore_types=[], select_types=[], select_keys=[],
                   ncpus=10, lite=True, use_legacy_names=False):
        """
        Create a Systematics object from saved data.
        This is the recommended way to load saved systematics.
        Note: This loads pre-computed CV quantities. To recompute from distributions,
        use the regular __init__.
        
        Parameters
        ----------
        load_dir : str
            Directory to load the systematics from
        var_name : str
            Variable name to load
        metadata_dir : str, optional
            Directory to load the metadata from
        ignore_keys : list, optional
            List of keys (directory names) to ignore
        ignore_types : list, optional
            List of types to ignore
        select_types : list, optional
            List of types to select
        select_keys : list, optional
            If non empty, only systematic dictionary keys in this list are loaded
        ncpus : int, optional
            Number of CPUs to use for parallel processing
        lite : bool, optional
            If True, only load the metadata and necessary systematics data.
        use_legacy_names : bool, optional
            If True, match subfolders by their 'name' field (old save format).
        Returns
        -------
        Systematics
            A new Systematics instance loaded from saved data
        """
        metadata_dir = os.path.join(load_dir, metadata_dir)
        if not os.path.exists(metadata_dir):
            raise ValueError(f'Metadata directory not found in {load_dir}. This might be an old save format.')
        
        bins = np.loadtxt(os.path.join(metadata_dir, f'{var_name}_bins.csv'))
        
        # Load CV quantities (handle None/missing files)
        sel = np.loadtxt(os.path.join(metadata_dir, f'{var_name}_sel.csv'))
        sel_background = np.loadtxt(os.path.join(metadata_dir, f'{var_name}_sel_background.csv'))
        
        # Load optional CV quantities (may not exist if they were None)
        sigma_tilde_path = os.path.join(metadata_dir, f'{var_name}_sigma_tilde.csv')
        sigma_tilde = np.loadtxt(sigma_tilde_path) if os.path.exists(sigma_tilde_path) else None
        
        sel_truth_path = os.path.join(metadata_dir, f'{var_name}_sel_truth.csv')
        sel_truth = np.loadtxt(sel_truth_path) if os.path.exists(sel_truth_path) else None
        
        sel_background_truth_path = os.path.join(metadata_dir, f'{var_name}_sel_background_truth.csv')
        sel_background_truth = np.loadtxt(sel_background_truth_path) if os.path.exists(sel_background_truth_path) else None
        
        sig_truth_path = os.path.join(metadata_dir, f'{var_name}_sig_truth.csv')
        sig_truth = np.loadtxt(sig_truth_path) if os.path.exists(sig_truth_path) else None
        
        response_path = os.path.join(metadata_dir, f'{var_name}_response.csv')
        response = np.loadtxt(response_path) if os.path.exists(response_path) else None
        
        smearing_path = os.path.join(metadata_dir, f'{var_name}_smearing.csv')
        smearing = np.loadtxt(smearing_path) if os.path.exists(smearing_path) else None
        
        eff_truth_path = os.path.join(metadata_dir, f'{var_name}_eff_truth.csv')
        eff_truth = np.loadtxt(eff_truth_path) if os.path.exists(eff_truth_path) else None
        
        eff_reco_path = os.path.join(metadata_dir, f'{var_name}_eff_reco.csv')
        eff_reco = np.loadtxt(eff_reco_path) if os.path.exists(eff_reco_path) else None

        sel_data_path = os.path.join(metadata_dir, f'{var_name}_sel_data.csv')
        sel_data = np.loadtxt(sel_data_path) if os.path.exists(sel_data_path) else None
        
        # Load xsec_unit (may not exist if it was None)
        xsec_unit_path = os.path.join(metadata_dir, 'xsec_unit.csv')
        if os.path.exists(xsec_unit_path):
            xsec_unit = np.loadtxt(xsec_unit_path)
            if xsec_unit.ndim == 0:
                xsec_unit = xsec_unit.item()
            else:
                xsec_unit = xsec_unit[0]
        else:
            xsec_unit = None
        
        # Load sys_dict and reconstruct keys (in parallel over systematics entries)
        with open(os.path.join(metadata_dir, 'sys_dict.json'), 'r') as f:
            sys_dict_serializable = json.load(f)
        
        # Convert back to proper format (lists to tuples for cols)
        sys_dict = {}
        items = list(sys_dict_serializable.items())

        def _process_sys_entry(item):
            key, val = item
            # Apply key/type selection logic
            if key in ignore_keys or val['type'] in ignore_types:
                return None
            if select_types != [] and val['type'] not in select_types:
                return None
            if select_keys != [] and key not in select_keys:
                return None

            # Handle summary keys where cols might be None
            if val['cols'] is None:
                cols_as_tuples = None
            else:
                cols_as_tuples = [
                    tuple(col) if isinstance(col, list) else col
                    for col in val['cols']
                ]

            entry = {
                'cols': cols_as_tuples,
                'col_names': val['col_names'],
                'type': val['type'],
                'name': val['name'],
                'label': val.get('label', val.get('name', key)),
                'description': val.get('description', ''),
                'variation': val['variation'],
                'order': val.get('order', None)
            }
            return key, entry

        max_workers = ncpus if ncpus is not None and ncpus > 0 else None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(_process_sys_entry, items):
                if result is None:
                    continue
                key, entry = result
                sys_dict[key] = entry

        if len(sys_dict) == 0:
            raise ValueError(f'No systematics found in {load_dir} for variable {var_name}')
        # Create a minimal instance by directly setting attributes
        # Since we don't have distributions, we can't use __init__
        instance = cls.__new__(cls)
        instance.variable_name = var_name
        instance.bins = bins
        instance.xsec_unit = xsec_unit
        instance.sigma_tilde = sigma_tilde
        instance.sel = sel
        instance.sel_background = sel_background
        instance.sel_data = sel_data
        instance.sel_truth = sel_truth
        instance.sel_background_truth = sel_background_truth
        instance.sig_truth = sig_truth
        instance.response = response
        instance.smearing = smearing
        instance.eff_truth = eff_truth
        instance.eff_reco = eff_reco
        # Set dummy distributions (won't be used but needed for compatibility)
        # These are empty arrays - from_saved objects can't process new systematics
        instance._true_sig = np.array([])
        instance._true_sel = np.array([])
        instance._true_sel_background = np.array([])
        instance._reco_sel = np.array([])
        instance._reco_sel_background = np.array([])
        instance._genweights_sig = np.array([])
        instance._genweights_sel = np.array([])
        instance._genweights_sel_background = np.array([])
        
        instance.systematics = {}
        for key, metadata in sys_dict.items():
            instance.add_syst(key, metadata=metadata)
        
        # Now load the systematics data
        instance.load(load_dir, ignore_keys=ignore_keys, ignore_types=ignore_types,
                      select_types=select_types, select_keys=select_keys,
                      metadata_dir=metadata_dir, lite=lite,
                      use_legacy_names=use_legacy_names)
        
        return instance
    
    def load(self, load_dir, ignore_keys=[], ignore_types=[], select_types=[],
             select_keys=[], metadata_dir='metadata_detsys', lite=True,
             use_legacy_names=False):
        """
        Load the systematics from a directory
        Note: The systematics object must be initialized before loading.
        For loading without initialization, use from_saved() class method instead.
        
        Parameters
        ----------
        load_dir : str
            Directory to load the systematics from
        ignore_keys : list, optional
            List of keys to ignore
        ignore_types : list, optional
            List of types to ignore
        select_types : list, optional
            List of types to select
        select_keys : list, optional
            If non empty, only these systematic keys are loaded from disk
        metadata_dir : str, optional
            Directory to load the metadata from
        lite : bool, optional
            If True, only load the metadata and necessary systematics data.
            Skips heavy per-systematic files (e.g. sel, sigma_tilde, response npy).
        use_legacy_names : bool, optional
            If True, match subfolders by their 'name' field (old save format
            where folders were named by sdict['name']). Default False uses
            the dict key directly.
        """
        lite_drop_keys = ['sel','sigma_tilde','eff','cols','col_names','description','response']
        if not self.systematics:
            raise ValueError('Systematics object must be initialized before loading. Use Systematics.from_saved() instead.')
        
        name_to_key = {}
        if use_legacy_names:
            name_to_key = {sdict['name']: key for key, sdict in self.systematics.items()}

        for subfolder in tqdm(os.listdir(load_dir), desc='Loading systematics', unit=' subfolder'):
            if subfolder in ignore_keys:
                continue
            subfolder_path = os.path.join(load_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            if subfolder == metadata_dir:
                continue

            if use_legacy_names and subfolder in name_to_key:
                sys_key = name_to_key[subfolder]
            elif subfolder in self.systematics:
                sys_key = subfolder
            else:
                sys_key = subfolder
                if select_keys != [] and sys_key not in select_keys:
                    continue
                self.systematics[sys_key] = Systematics._get_dict_template().copy()
                self.systematics[sys_key]['name'] = subfolder
                self.systematics[sys_key]['label'] = subfolder
                self.systematics[sys_key]['description'] = ''
                self.systematics[sys_key]['order'] = None

            if select_keys != [] and sys_key not in select_keys:
                self.systematics.pop(sys_key, None)
                continue

            if self.systematics[sys_key]['type'] in ignore_types or (select_types != [] and self.systematics[sys_key]['type'] not in select_types):
                self.systematics.pop(sys_key)
                continue
            for file in os.listdir(subfolder_path):
                if lite and any(key in file for key in lite_drop_keys):
                    continue
                file_path = os.path.join(subfolder_path, file)
                if not os.path.isfile(file_path):
                    continue
                    
                # Check if file matches current variable_name prefix
                filename_prefix = f'{self.variable_name}_'
                if not file.startswith(filename_prefix):
                    # Skip files that don't belong to this variable name
                    continue
                    
                # Remove extension and variable_name prefix
                key_name = file
                for _suf in ('.csv', '.txt', '.npy'):
                    if key_name.endswith(_suf):
                        key_name = key_name[: -len(_suf)]
                        break
                # Strip variable_name prefix (should always match at this point)
                if key_name.startswith(filename_prefix):
                    key_name = key_name[len(filename_prefix):]
                
                if file.endswith('.csv'):
                    # Suppress UserWarning for empty data in loadtxt
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning, message='.*loadtxt: input contained no data.*')
                        arr_like = np.loadtxt(file_path)
                    # Handle 0D arrays
                    if arr_like.ndim == 0:
                        arr_like = arr_like.item()
                    # Handle 1D arrays with single element
                    elif arr_like.size == 1 and arr_like.ndim == 1:
                        arr_like = arr_like[0]
                    self.systematics[sys_key][key_name] = arr_like
                elif file.endswith('.txt'):
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    # Strip newlines and filter empty lines
                    arr_like = [line.strip() for line in lines if line.strip()]
                    # If it's a single value
                    if len(arr_like) == 1 and key_name not in ['col_names']:
                        value = arr_like[0]
                        # Check if this key should be a string (like 'name', 'type', 'variation', 'label', 'description')
                        if key_name in ['name', 'type', 'variation', 'label', 'description']:
                            arr_like = value
                        else:
                            # Try to convert to numeric scalar
                            try:
                                # Try to convert to float first
                                float_val = float(value)
                                # If it's actually an integer, convert to int
                                if float_val.is_integer():
                                    arr_like = int(float_val)
                                else:
                                    arr_like = float_val
                            except ValueError:
                                # If conversion fails, keep as string
                                arr_like = value
                    self.systematics[sys_key][key_name] = arr_like
                elif file.endswith('.npy'):
                    self.systematics[sys_key][key_name] = np.load(
                        file_path, allow_pickle=True)
                else:
                    #raise ValueError(f'File {file} has unknown extension, only .csv and .txt are supported')
                    continue
        print(f'Loaded {len(self.systematics)} systematics from {load_dir}')