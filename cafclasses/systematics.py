import numpy as np
import matplotlib.pyplot as plt
from sbnd.plotlibrary import makeplot
from sbnd.general import plotters
from sbnd.stats.stats import (
    construct_covariance, construct_correlation_matrix, 
    get_fractional_uncertainty, get_total_unc, convert_smearing_to_response,
    get_smear_matrix, compute_efficiency, compute_sigma_tilde
)
from tqdm import tqdm
import os
import copy
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed


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
            # Sigma tilde
            'sigma_tilde': [],
            # Reco event rate
            'sel': [],
            # Xsec covariance
            'xsec_cov': None,
            # Event covariance
            'event_cov': None,
            # Xsec correlation
            'xsec_corr': None,
            # Event correlation
            'event_corr': None,
            # Xsec fractional covariance
            'xsec_fraccov': None,
            # Event fractional covariance
            'event_fraccov': None,
            # Xsec fractional uncertainty
            'xsec_fracunc': None,
            # Event fractional uncertainty
            'event_fracunc': None,
            # Event total uncertainty
            'event_totalunc': None,
            # Xsec total uncertainty
            'xsec_totalunc': None
        }
    
    def __init__(self, variable_name, bins,
                 reco_sel, reco_sel_background,
                 genweights_sig, genweights_sel, genweights_sel_background,
                 xsec_unit=None, true_sig=None, true_sel=None, true_sel_background=None,
                 keys=None,**kwargs):
        """
        Initialize Systematics class.
        
        Parameters
        ----------
        variable_name : str
            Name of the variable (e.g., 'costheta', 'momentum', 'differential')
        bins : array-like
            Bin edges for this variable
        reco_sel : array-like
            Reconstructed variable values for selected
        reco_sel_background : array-like
            Reconstructed variable values for selected background
        genweights_sig : array-like
            Generator weights for signal
        genweights_sel : array-like
            Generator weights for selected
        genweights_sel_background : array-like
            Generator weights for selected background
        xsec_unit : float, optional
            Cross section unit conversion factor. If None, smearing matrix will not be computed.
        true_sig : array-like or None
            True variable values for signal. If None, smearing matrix will not be computed.
        true_sel : array-like or None
            True variable values for selected. If None, smearing matrix will not be computed.
        true_sel_background : array-like or None
            True variable values for selected background. If None, smearing matrix will not be computed.
        keys : array-like, optional
            Keys from truth level to identify systematics (can be set later)
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

        # Inline histogram computation (previously compute_histograms)
        # Signal truth histogram
        if self._true_sig is not None:
            if self._genweights_sig is not None:
                sig_truth, _ = np.histogram(
                    self._true_sig, bins=self.bins, weights=self._genweights_sig
                )
            else:
                sig_truth, _ = np.histogram(self._true_sig, bins=self.bins)
        else:
            sig_truth = None
        
        # Selected truth histogram
        if self._true_sel is not None:
            if self._genweights_sel is not None:
                sel_truth, _ = np.histogram(
                    self._true_sel, bins=self.bins, weights=self._genweights_sel
                )
            else:
                sel_truth, _ = np.histogram(self._true_sel, bins=self.bins)
        else:
            sel_truth = None
        
        # Selected background truth histogram
        if self._true_sel_background is not None:
            if self._genweights_sel_background is not None:
                sel_background_truth, _ = np.histogram(
                    self._true_sel_background, bins=self.bins, weights=self._genweights_sel_background
                )
            else:
                sel_background_truth, _ = np.histogram(self._true_sel_background, bins=self.bins)
        else:
            sel_background_truth = None
        
        # Selected reco histogram
        if self._genweights_sel is not None:
            sel, _ = np.histogram(
                self._reco_sel, bins=self.bins, weights=self._genweights_sel
            )
        else:
            sel, _ = np.histogram(self._reco_sel, bins=self.bins)
        
        # Selected background reco histogram
        if self._genweights_sel_background is not None:
            sel_background, _ = np.histogram(
                self._reco_sel_background, bins=self.bins, weights=self._genweights_sel_background
            )
        else:
            sel_background, _ = np.histogram(self._reco_sel_background, bins=self.bins)
        
        self.sig_truth = sig_truth
        self.sel_truth = sel_truth
        self.sel_background_truth = sel_background_truth
        self.sel = sel
        self.sel_background = sel_background
        if self.sig_truth is not None:
            assert len(self.sig_truth) == len(self.sel_truth) == len(self.sel_background_truth) == len(self.sel) == len(self.sel_background), f'Sig truth, sel truth, sel background, sel, and sel background have different lengths: {len(self.sig_truth)} != {len(self.sel_truth)} != {len(self.sel_background_truth)} != {len(self.sel)} != {len(self.sel_background)}'
        else:
            assert len(self.sel) == len(self.sel_background), f'Sel and sel background have different lengths: {len(self.sel)} != {len(self.sel_background)}'
        
        # Compute efficiency if we have the required histograms
        if self.sig_truth is not None and self.sel_truth is not None:
            self.eff_truth = compute_efficiency(self.sig_truth, self.sel_truth)
        else:
            self.eff_truth = None
        
        # Compute eff_reco (efficiency over reco distribution)
        if self.sig_truth is not None:
            self.eff_reco = self.sel / self.sig_truth
            # Handle division by zero
            self.eff_reco = np.where(self.sig_truth > 0, self.eff_reco, 0.0)
        else:
            self.eff_reco = None
        
        # Compute smearing matrix, response matrix, and sigma_tilde only if we have true_sel and xsec_unit
        if self._true_sel is not None and self.eff_truth is not None and self.xsec_unit is not None:
            self.smearing = get_smear_matrix(
                self._true_sel, self._reco_sel, self.bins,
                weights=self._genweights_sel
            )
            
            self.response = convert_smearing_to_response(self.smearing, self.eff_truth)
            
            self.sigma_tilde = compute_sigma_tilde(
                self.response, self.sel_truth, self.sel_background, self.xsec_unit
            )
        else:
            self.smearing = None
            self.response = None
            self.sigma_tilde = None
        
        # Initialize systematic dictionary
        if keys is not None:
            self.sys_dict = self.get_sys_keydict(keys,**kwargs)
        else:
            self.sys_dict = {}
        
        # Initialize systematic results dictionary
        self.systematics = {}
        self._initialize_systematic_dicts()
    
    def _initialize_systematic_dicts(self):
        """Initialize the systematic dictionaries with the template structure."""
        dict_template = Systematics._get_dict_template()
        
        for key in self.sys_dict:
            self.systematics[key] = dict_template.copy()
            self.systematics[key]['cols'] = self.sys_dict[key]['cols']
            self.systematics[key]['col_names'] = self.sys_dict[key]['col_names']
            self.systematics[key]['type'] = self.sys_dict[key]['type']
            self.systematics[key]['name'] = self.sys_dict[key]['name']
            # If label/description exist in sys_dict (from saved data), use them; otherwise default to name/empty
            self.systematics[key]['label'] = self.sys_dict[key].get('label', self.sys_dict[key]['name'])
            self.systematics[key]['description'] = self.sys_dict[key].get('description', '')
            self.systematics[key]['variation'] = self.sys_dict[key]['variation']
    def __repr__(self):
        return f'Systematics(variable={self.variable_name}, keys={list(self.sys_dict.keys())})'
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
            key = key.replace('GENIEReWeight_SBN_v1_', '')
            key = key.replace('_Flux', '')
            key = key.replace('reinteractions_', '')
            key = key.replace('_Geant4', '')
            return key
        def assign_type(key,stype='RW'):
            if stype == 'RW':
                if 'genie' in key.lower():
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
                elif 'alpha' in key.lower() or 'beta' in key.lower() or 'R_' in key:
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
                    return 'unisim'
            elif stype == 'Det' or stype == 'Cosmic':
                return 'unisim'
            else:
                print(f'Unknown systematic type: {stype}')
                return 'unknown'
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
            if candidate not in sys_col_dict:
                formatted_name = format_name(candidate)
                sys_col_dict[candidate] = {'cols' : [], 'col_names' : [], 'type' : assign_type(candidate,stype), 'name' : formatted_name, 'label' : formatted_name, 'description' : '', 'variation' : assign_variation(candidate,stype)}
            cnt+=1
            if stype == 'RW':
                sys_col_dict[candidate]['cols'].append(tuple(['truth']+list(k)))
                sys_col_dict[candidate]['col_names'].append('_'.join(list(k)).rstrip('_'))
        return sys_col_dict
    
    # @staticmethod
    # def process_systematics_parallel(systematics_objects, mc_signal_data, mc_sel_signal_data,
    #                                  max_workers=None, progress_bar=False):
    #     """
    #     Process multiple Systematics objects in parallel.
        
    #     Parameters
    #     ----------
    #     systematics_objects : list
    #         List of Systematics objects to process
    #     mc_signal_data : data frame
    #         MC signal slice with data attribute containing systematic columns
    #     mc_sel_signal_data : data frame
    #         MC selected signal slice with data attribute containing systematic columns
    #     max_workers : int, optional
    #         Maximum number of worker threads (default: len(systematics_objects))
    #     progress_bar : bool
    #         Whether to show progress bar (requires tqdm)
        
    #     Returns
    #     -------
    #     None
    #         Results are stored in each Systematics object
    #     """
    #     if max_workers is None:
    #         max_workers = len(systematics_objects)
        
    #     def _process_one(sys_obj):
    #         """Helper function to process a single Systematics object."""
    #         sys_obj.process_systematics(mc_signal_data, mc_sel_signal_data, progress_bar=False)
    #         return sys_obj.variable_name
        
    #     if progress_bar:
    #         with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #             futures = {executor.submit(_process_one, sys_obj): sys_obj 
    #                       for sys_obj in systematics_objects}
    #             for future in tqdm(as_completed(futures), total=len(futures), unit=' variable'):
    #                 variable_name = future.result()
    #     else:
    #         with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #             futures = [executor.submit(_process_one, sys_obj) for sys_obj in systematics_objects]
    #             for future in as_completed(futures):
    #                 future.result()
    
    def process_systematics(self, mc_signal_data, mc_sel_signal_data,
                           progress_bar=False):
        """
        Process all systematic universes and compute histograms, efficiencies, 
        smearing/response matrices, and sigma_tilde values.
        
        Parameters
        ----------
        mc_signal_data : data frame
            MC signal slice with data attribute containing systematic columns
        mc_sel_signal_data : data frame
            MC selected signal slice with data attribute containing systematic columns
        progress_bar : bool
            Whether to show progress bar (requires tqdm)
        """
        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(self.systematics.items(), unit=' goomba')
            except ImportError:
                iterator = self.systematics.items()
        else:
            iterator = self.systematics.items()
        
        for key, sys_dict in iterator:
            cols = sys_dict['cols']
            col_names = sys_dict['col_names']
            
            # Reset lists for this systematic
            sys_dict['sigma_tilde'] = []
            sys_dict['sel'] = []
            
            for col, col_name in zip(cols, col_names):
                # Skip CV for multisigma variations
                if '_cv' in col_name and sys_dict['variation'] == 'multisigma':
                    continue
                
                # Selected reco histogram
                _sel, _ = np.histogram(
                    self._reco_sel, bins=self.bins,
                    weights=self._genweights_sel * mc_sel_signal_data[col]
                )
                
                # Selected background reco histogram (no weights variation)
                _sel_background, _ = np.histogram(
                    self._reco_sel_background, bins=self.bins,
                    weights=self._genweights_sel_background
                )
                
                # Store sel results
                sys_dict['sel'].append(_sel + _sel_background)
                
                # Compute sigma_tilde only if we have true distributions and xsec_unit
                if self._true_sig is not None and self._true_sel is not None and self._true_sel_background is not None and self.xsec_unit is not None:
                    # Get the distributions for this universe
                    # Signal truth histogram
                    _sig_truth, _ = np.histogram(
                        self._true_sig, bins=self.bins,
                        weights=self._genweights_sig * mc_signal_data[col]
                    )
                    
                    # Selected truth histogram
                    _sel_truth, _ = np.histogram(
                        self._true_sel, bins=self.bins,
                        weights=self._genweights_sel * mc_sel_signal_data[col]
                    )
                    
                    # Selected background truth histogram (no weights variation)
                    _sel_background_truth, _ = np.histogram(
                        self._true_sel_background, bins=self.bins,
                        weights=self._genweights_sel_background
                    )
                    
                    # Get the efficiency
                    _eff_truth = compute_efficiency(_sig_truth, _sel_truth)
                    
                    # Get the smearing matrix
                    _smearing = get_smear_matrix(
                        self._true_sel, self._reco_sel, self.bins,
                        weights=self._genweights_sel * mc_sel_signal_data[col]
                    )
                    
                    # Use universe efficiency for GENIE xsec uncertainties, CV efficiency otherwise
                    if 'GENIE' in key:
                        _response = convert_smearing_to_response(_smearing, _eff_truth)
                    else:
                        _response = convert_smearing_to_response(_smearing, self.eff_truth)
                    
                    # Compute sigma_tilde
                    _sigma_tilde = compute_sigma_tilde(
                        _response, self.sel_truth, _sel_background, self.xsec_unit
                    )
                    
                    # Store results
                    sys_dict['sigma_tilde'].append(_sigma_tilde)
                else:
                    # If we don't have the required parameters, append None
                    sys_dict['sigma_tilde'].append(None)
    
    def process_det_systematics(self,
                                reco_sel_vars, reco_sel_background_vars,genweights_sel_vars,
                                genweights_sel_background_vars,
                                true_sig_vars=None, true_sel_vars=None, true_sel_background_vars=None,
                                sys_names=None,progress_bar=False):
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
                
                # Use CV efficiency (not variation-specific) for response matrix conversion
                _var_response = convert_smearing_to_response(
                    _var_smearing, self.eff_truth
                )
                
                # Compute sigma_tilde for variation
                _var_sigma_tilde = compute_sigma_tilde(
                    _var_response, self.sel_truth, _sel_background, self.xsec_unit
                )
                
                # Store results
                sys_dict['sigma_tilde'].append(_var_sigma_tilde)
            else:
                # If we don't have the required attributes, append None
                sys_dict['sigma_tilde'].append(None)
    
    def process_flat_systematic(self, sys_key, frac_unc, apply_to='both'):
        """
        Process a flat systematic with a given fractional uncertainty.
        Creates two symmetric variations (+/-) that will produce a flat covariance.
        
        Parameters
        ----------
        sys_key : str
            Key for the systematic in self.systematics dictionary
        frac_unc : float
            Overall fractional uncertainty (e.g., 0.05 for 5%)
        apply_to : str, optional
            What to apply the systematic to: 'sel', 'sigma_tilde', or 'both' (default)
        """
        if sys_key not in self.systematics:
            dict_template = Systematics._get_dict_template()
            sys_dict = dict_template.copy()
            sys_dict['type'] = sys_key
            sys_dict['name'] = sys_key
            sys_dict['label'] = sys_key
            sys_dict['description'] = ''
            sys_dict['variation'] = 'flat'
            self.systematics[sys_key] = sys_dict
        else:
            sys_dict = self.systematics[sys_key]
        
        # Reset lists for this systematic
        sys_dict['sigma_tilde'] = []
        sys_dict['sel'] = []
        
        # Create two symmetric variations: +frac_unc and -frac_unc
        for sign in [+1, -1]:
            # Apply to sel (reco event rate)
            if apply_to in ['sel', 'both']:
                _sel_variation = (self.sel + self.sel_background) * (1 + sign * frac_unc)
                sys_dict['sel'].append(_sel_variation)
            else:
                sys_dict['sel'].append(self.sel + self.sel_background)
            
            # Apply to sigma_tilde (xsec) only if it exists
            if apply_to in ['sigma_tilde', 'both']:
                if self.sigma_tilde is not None:
                    _sigma_tilde_variation = self.sigma_tilde * (1 + sign * frac_unc)
                    sys_dict['sigma_tilde'].append(_sigma_tilde_variation)
                else:
                    sys_dict['sigma_tilde'].append(None)
            else:
                sys_dict['sigma_tilde'].append(self.sigma_tilde)
    
    def compute_covariances(self,keys=None):
        """
        Compute covariance matrices, correlations, and uncertainties for each systematic.
        
        Parameters
        ----------
        keys : list, optional
            List of keys to compute covariance matrices for. If None, compute for all keys
        """
        if keys is None:
            keys = list(self.systematics.keys())
        else:
            #Check that keys are in the systematics dictionary
            for key in keys:
                if key not in self.systematics:
                    raise ValueError(f'Key {key} not found in systematics dictionary')
        for key in keys:
            sys_dict = self.systematics[key]
            if len(sys_dict['sel']) == 0:
                raise ValueError(f'No sel events for systematic {key} for {self.variable_name}')

            # Xsec (only compute if sigma_tilde exists)
            if self.sigma_tilde is not None and sys_dict['sigma_tilde'] is not None:
                if len(sys_dict['sigma_tilde']) == 0:
                    raise ValueError(f'No sigma_tilde events for systematic {key} for {self.variable_name} (but we expect them)')
                # Filter out None values from sigma_tilde list
                sigma_tilde_list = [s for s in sys_dict['sigma_tilde'] if s is not None]
                if len(sigma_tilde_list) > 0:
                    _xsec_cov, _xsec_frac, _xsec_corr, _xsec_fracunc = construct_covariance(
                        self.sigma_tilde, sigma_tilde_list, assert_cov=False
                    )
                    sys_dict['xsec_cov'] = _xsec_cov
                    sys_dict['xsec_fraccov'] = _xsec_frac
                    sys_dict['xsec_corr'] = _xsec_corr
                    sys_dict['xsec_fracunc'] = _xsec_fracunc
                    _xsec_total_unc = get_total_unc(
                        self.sigma_tilde, sys_dict['xsec_fracunc']
                    )
                    sys_dict['xsec_totalunc'] = _xsec_total_unc
                else:
                    sys_dict['xsec_cov'] = None
                    sys_dict['xsec_fraccov'] = None
                    sys_dict['xsec_corr'] = None
                    sys_dict['xsec_fracunc'] = None
                    sys_dict['xsec_totalunc'] = None
            else:
                sys_dict['xsec_cov'] = None
                sys_dict['xsec_fraccov'] = None
                sys_dict['xsec_corr'] = None
                sys_dict['xsec_fracunc'] = None
                sys_dict['xsec_totalunc'] = None
            
            # Event
            #print(f'key: {key}, len(self.sel): {len(self.sel)}, len(sys_dict["sel"]): {len(sys_dict["sel"])}')
            _cov, _frac, _corr, _fracunc = construct_covariance(
                self.sel+self.sel_background, sys_dict['sel'], assert_cov=False
            )
            sys_dict['event_cov'] = _cov
            sys_dict['event_fraccov'] = _frac
            sys_dict['event_corr'] = _corr
            sys_dict['event_fracunc'] = _fracunc
            _total_unc = get_total_unc(
                self.sel+self.sel_background, sys_dict['event_fracunc']
            )
            sys_dict['event_totalunc'] = _total_unc

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
    
    def combine(self, other, other_override=False, store_other=False, other_name='other'):
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
        """
        # Check compatibility
        if not np.array_equal(self.bins, other.bins):
            raise ValueError(f'Cannot combine Systematics objects: bins differ')
        if self.variable_name != other.variable_name:
            raise ValueError(f'Cannot combine Systematics objects: variable names differ ({self.variable_name} vs {other.variable_name})')
        # if self.xsec_unit != other.xsec_unit:
        #     raise ValueError(f'Cannot combine Systematics objects: xsec_unit differs ({self.xsec_unit} vs {other.xsec_unit})')
        
        # Add non-overlapping keys from other to self
        for key, sys_dict in other.systematics.items():
            if key not in self.systematics:
                # Deep copy the systematic dictionary
                self.systematics[key] = copy.deepcopy(sys_dict)
            # If key exists in both, skip it (non-overlapping only)
            else:
                if other_override:
                    #print(f'Key {key} already exists in self.systematics and other_override is False, skipping')
                    pass
                else:
                    self.systematics[key] = copy.deepcopy(sys_dict)
        
        # Also merge sys_dict if it exists
        if hasattr(other, 'sys_dict'):
            for key, val in other.sys_dict.items():
                if key not in self.sys_dict:
                    self.sys_dict[key] = copy.deepcopy(val)
        
        if store_other:
            sdict = Systematics._get_dict_template()
            sdict['name'] = other_name
            sdict['sel'] = other.sel
            sdict['sigma_tilde'] = other.sigma_tilde
            self.systematics[other_name] = sdict

        #print(f'self.systematics: {self.systematics.keys()}')
    
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
            
            # Initialize covariance matrices
            n_bins = len(self.bins) - 1

            self.systematics[sk]['event_cov'] = np.zeros((n_bins, n_bins))
            if self.sigma_tilde is not None:
                self.systematics[sk]['xsec_cov'] = np.zeros((n_bins, n_bins))
                self.systematics[sk]['xsec_fraccov'] = np.zeros((n_bins, n_bins))
            else:
                self.systematics[sk]['xsec_cov'] = None
                self.systematics[sk]['xsec_fraccov'] = None
            self.systematics[sk]['event_fraccov'] = np.zeros((n_bins, n_bins))
        
        # Sum covariance matrices
        for key, sys_dict in self.systematics.items():
            if sys_dict['type'] == sys_dict['name']:  # Skip summary keys themselves
                continue
            
            for sk in summary_keys:
                if (sys_dict['type'] == sk) or (sk == 'total') or (sk == 'det'):
                    # Event covariance
                    if sys_dict['event_cov'] is not None:
                        self.systematics[sk]['event_cov'] += sys_dict['event_cov']
                    
                    # Xsec covariance (only add if summary's xsec_cov is not None)
                    if self.systematics[sk]['xsec_cov'] is not None and sys_dict['xsec_cov'] is not None:
                        self.systematics[sk]['xsec_cov'] += sys_dict['xsec_cov']
                    
                    # Event fractional covariance
                    if sys_dict['event_fraccov'] is not None:
                        self.systematics[sk]['event_fraccov'] += sys_dict['event_fraccov']
                    
                    # Xsec fractional covariance (only add if summary's xsec_fraccov is not None)
                    if self.systematics[sk]['xsec_fraccov'] is not None and sys_dict['xsec_fraccov'] is not None:
                        self.systematics[sk]['xsec_fraccov'] += sys_dict['xsec_fraccov']
        
        # Compute fractional uncertainties and correlations from combined covariances
        for sk in summary_keys:
            # Event fractional uncertainty
            self.systematics[sk]['event_fracunc'] = get_fractional_uncertainty(
                self.sel+self.sel_background, self.systematics[sk]['event_cov']
            )
            
            # Xsec fractional uncertainty (only if sigma_tilde exists)
            if self.sigma_tilde is not None and self.systematics[sk]['xsec_cov'] is not None:
                self.systematics[sk]['xsec_fracunc'] = get_fractional_uncertainty(
                    self.sigma_tilde, self.systematics[sk]['xsec_cov']
                )
            else:
                self.systematics[sk]['xsec_fracunc'] = None
            
            # Event correlation
            self.systematics[sk]['event_corr'], _ = construct_correlation_matrix(
                self.systematics[sk]['event_cov']
            )
            
            # Xsec correlation (only if xsec_cov exists)
            if self.systematics[sk]['xsec_cov'] is not None:
                self.systematics[sk]['xsec_corr'], _ = construct_correlation_matrix(
                    self.systematics[sk]['xsec_cov']
                )
            else:
                self.systematics[sk]['xsec_corr'] = None
            
            # Event total uncertainty
            self.systematics[sk]['event_totalunc'] = get_total_unc(
                self.sel+self.sel_background, self.systematics[sk]['event_fracunc']
            )
            
            # Xsec total uncertainty (only if sigma_tilde exists)
            if self.sigma_tilde is not None and self.systematics[sk]['xsec_fracunc'] is not None:
                self.systematics[sk]['xsec_totalunc'] = get_total_unc(
                    self.sigma_tilde, self.systematics[sk]['xsec_fracunc']
                )
            else:
                self.systematics[sk]['xsec_totalunc'] = None
    
    def add_description(self, description_dict):
        """
        Add labels and descriptions to systematics from a dictionary.
        
        Parameters
        ----------
        description_dict : dict
            Dictionary with keys matching systematic keys. Each value should be
            a dict with 'label' and/or 'description' keys.
            
        Raises
        ------
        KeyError
            If a key in description_dict is not found in self.systematics
        """
        for key, desc_data in description_dict.items():
            if key not in self.systematics:
                raise KeyError(f"Key '{key}' not found in systematics dictionary. Available keys: {list(self.systematics.keys())}")
            
            if not isinstance(desc_data, dict):
                raise TypeError(f'Value for key "{key}" must be a dictionary with "label" and/or "description" keys')
            
            if 'label' in desc_data:
                self.systematics[key]['label'] = desc_data['label']
            if 'description' in desc_data:
                self.systematics[key]['description'] = desc_data['description']
    
    def generate_description_template(self, save_dir=None):
        """
        Generate a template dictionary for adding descriptions to systematics.
        
        Parameters
        ----------
        save_dir : str, optional
            Directory to save the template JSON file. If None, don't save.
            
        Returns
        -------
        dict
            Dictionary with keys matching systematic keys. Each value is a dict
            with 'label' (set to current label/name) and 'description' (empty string).
        """
        template = {}
        for key, sys_dict in self.systematics.items():
            template[key] = {
                'label': sys_dict.get('label', sys_dict.get('name', key)),
                'description': sys_dict.get('description', '')
            }
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            template_path = os.path.join(save_dir, 'description_template.json')
            with open(template_path, 'w') as f:
                json.dump(template, f, indent=2)
            print(f'Description template saved to {template_path}')
        
        return template
    
    def plot_event_rate_errs(self, base_key, exclude_keys=[], include_keys=[],
                            xlabel='', max_uncs=None, fig=None, ax=None, sort=False):
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
        #First, filter systematics to only include valid keys (where the total unc is not None)
        valid_keys = []
        for key,sys_dict in self.systematics.items():
            if sys_dict.get(f'{base_key}_totalunc') is not None:
                valid_keys.append(key)
        self.systematics = {k:v for k,v in self.systematics.items() if k in valid_keys}
        if sort:
            # Sort systematics by total uncertainty (largest first)
            sorted_items = sorted(
                self.systematics.items(),
                key=lambda x: x[1].get(f'{base_key}_totalunc', 0),
                reverse=True
            )
        else:
            sorted_items = self.systematics.items()
        assert len(sorted_items) > 0, f'No systematics found for {base_key}'
        
        if max_uncs is None:
            max_uncs = len(sorted_items)
        
        if len(include_keys) == 0:
            include_keys = [item[0] for item in sorted_items]
            if max_uncs == len(sorted_items):
                max_uncs = len(include_keys)
        else:
            max_uncs = len(include_keys)
        
        unc_colors = plotters.get_colors('gist_rainbow', max_uncs + 1)
        
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        
        cnt = 0
        for i, (sys_name, sys_dict) in enumerate(sorted_items):
            if cnt >= max_uncs:
                break
            if sys_name in exclude_keys or sys_name not in include_keys:
                continue
            
            cnt += 1
            c = unc_colors[cnt]
            total_unc = sys_dict.get(f'{base_key}_totalunc', 0)
            frac_unc = sys_dict.get(f'{base_key}_fracunc')

            #print(sys_name,sys_dict['label'],total_unc,frac_unc)
            
            if frac_unc is None:
                continue
            
            label = f"{sys_dict['label']} ({total_unc*100:.1f}%)"
            #color looks bad, skip for now
            makeplot.plot_hist_edges(
                self.bins, frac_unc, None, ax=ax, label=label#, color=c
            )
        
        ax.legend(ncol=int(np.ceil(max_uncs/24)), bbox_to_anchor=(1.05, 1.05))
        ax.set_xlabel(xlabel)
        if 'xsec' in base_key:
            ax.set_ylabel(r'Cross Section Uncertainty')
        else:
            ax.set_ylabel(r'Event Rate Uncertainty')
        if self.variable_name == 'momentum':
            ax.set_xlim(0, 4)
        if self.variable_name == 'opt0':
            ax.set_xscale('log')
        
        return [fig,ax,None]
    
    def plot_all_covariance_matrices(self, plot_dir=None, save_plots=True, progress_bar=False):
        """
        Plot all covariance matrices.
        
        Parameters
        ----------
        plot_dir : str, optional
            Directory to save plots
        save_plots : bool
            Whether to save plots
        """
        if progress_bar:
            pbar = tqdm(self.systematics.items(), unit=' goomba')
        else:
            pbar = self.systematics.items()
        for key, _ in pbar:
            _ = self.plot_covariance_matrices(key, plot_dir, save_plots)
    
    def plot_covariance_matrices(self, sys_key, plot_dir=None, save_plots=True):
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
            if sys_dict[plotkey] is None:
                continue
            
            # Determine matrix type and colormap
            if 'fraccov' in plotkey:
                cmap = 'viridis'
                subfolder = 'fraccov'
            elif 'cov' in plotkey:
                cmap = 'inferno'
                subfolder = 'cov'
            elif 'corr' in plotkey:
                cmap = 'cividis'
                subfolder = 'corr'
            else:
                continue
            
            # Determine variable and bins based on variable_name
            set_axlims = False
            if self.variable_name == 'costheta':
                axlabel = r'Reconstructed $\cos\theta_{\mu}$'
            elif self.variable_name == 'momentum':
                axlabel = r'Reconstructed $p_{\mu}$ (GeV)'
                set_axlims = True
            elif self.variable_name == 'differential':
                axlabel = r'Reconstructed 2D Bin ID'
            else:
                axlabel = f'Reconstructed {self.variable_name}'
            
            bins = self.bins
            
            title = f"{sys_dict['name']} {self.variable_name} {plotkey.replace('_', ' ')}"
            name = title.replace(' ', '_')+f'_{sys_dict["variation"]}'
            
            fig, ax = plt.subplots()
            ax.set_title(title)
            im = ax.pcolormesh(bins, bins, sys_dict[plotkey], cmap=cmap)
            ax.set_xlabel(axlabel)
            ax.set_ylabel(axlabel)
            #If the matrix contains NaN or Inf, don't plot the colorbar
            if not np.isfinite(sys_dict[plotkey]).all() or not np.isnan(bins).all():
                fig.colorbar(im, ax=ax)
            #else:
            #    fig.colorbar(im, ax=ax,vmin=np.nanmin(sys_dict[plotkey]),vmax=np.nanmax(sys_dict[plotkey]))
            if set_axlims:
                ax.set_xlim(0, 4.)
                ax.set_ylim(0, 4.)
            
            if save_plots:
                if plot_dir is not None:
                    plotters.save_plot(name, fig=fig, folder_name=f'{plot_dir}/{subfolder}')
                else:
                    raise ValueError('plot_dir is None and save_plots is True')
                plt.close(fig)
            else:
                plt.show(fig=fig)
            
            figs[plotkey] = [fig,ax,im]
        
        return figs
    
    def plot_all_distributions(self, keys=None, exclude_keys=[], include_keys=[], **kwargs):
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
            figs[sys_key] = self.plot_distributions(sys_key, **kwargs)
        return figs
    
    def plot_distributions(self, sys_key, plot_key='sel', plot_dir=None, save_plots=True):
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
        
        set_xlim = False
        xscale = 'linear'
        if self.variable_name == 'costheta':
            axlabel = r'Reconstructed $\cos\theta_{\mu}$'
        elif self.variable_name == 'momentum':
            axlabel = r'Reconstructed $p_{\mu}$ (GeV)'
            set_xlim = True
        elif self.variable_name == 'differential':
            axlabel = r'Reconstructed 2D Bin ID'
        elif self.variable_name == 'opt0':
            axlabel = r'Reconstructed $1/Opt0$ Score'
            xscale = 'log'
        else:
            axlabel = f'{self.variable_name}'
        
        bins = self.bins
        if plot_key == 'sigma_tilde':
            hist_var = self.sigma_tilde
            y_label = r'$\tilde{\sigma}$'
        elif plot_key == 'sel':
            hist_var = self.sel + self.sel_background
            y_label = r'Selected Candidates'
        else:
            raise ValueError(f'Invalid plot_key: {plot_key}')
        if hist_var is None:
            return figs
        title = f"{sys_dict['name']} {plot_key.replace('_', ' ')} {self.variable_name}"
        name = title.replace(' ', '_')+f'_{sys_dict["variation"]}'
        
        fig, ax = plt.subplots()
        makeplot.plot_hist_edges(
            bins, hist_var, None, ax=ax, color='black', label='CV', alpha=1.
        )
        for i,var in enumerate(sys_dict[plot_key]):
            if var is None:
                continue
            if i == 0:
                label = 'Variation(s)'
            else:
                label = None
            makeplot.plot_hist_edges(
                bins, var, None, ax=ax, color='blue', label=label,
                alpha=1./len(sys_dict[plot_key])
            )
        ax.legend()
        ax.set_xlabel(axlabel)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        if set_xlim:
            ax.set_xlim(0, 4.)
        ax.set_xscale(xscale)
        if save_plots:
            if plot_dir is not None:
                plotters.save_plot(name, fig=fig, folder_name=plot_dir)
            else:
                raise ValueError('plot_dir is None and save_plots is True')
            plt.close(fig)
        else:
            plt.show(fig=fig)
        
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
        
        # Deep copy the sys_dict
        new_obj.sys_dict = copy.deepcopy(self.sys_dict)
        
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
                'variation': val['variation']
            }
        with open(os.path.join(metadata_dir, 'sys_dict.json'), 'w') as f:
            json.dump(sys_dict_serializable, f, indent=2)
        
        # Save systematics data
        for key,sdict in self.systematics.items():
            subfolder = sdict['name']
            for k,arr_like in sdict.items():
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
    def from_saved(cls, load_dir, var_name,metadata_dir='metadata',ignore_keys=[],ignore_types=[],select_types=[]):
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
        
        # Load sys_dict and reconstruct keys
        with open(os.path.join(metadata_dir, 'sys_dict.json'), 'r') as f:
            sys_dict_serializable = json.load(f)
        
        # Convert back to proper format (lists to tuples for cols)
        sys_dict = {}
        reconstructed_keys = []
        for key, val in sys_dict_serializable.items():
            if key in ignore_keys or val['type'] in ignore_types:
                #print(f'Ignoring {key} of type {val["type"]} because it is in ignore_keys: {ignore_keys}')
                continue
            if select_types != [] and val['type'] not in select_types:
                #print(f'Ignoring {key} of type {val["type"]} because it is not in select_types: {select_types}')
                continue
            # Handle summary keys where cols might be None
            if val['cols'] is None:
                cols_as_tuples = None
            else:
                cols_as_tuples = [tuple(col) if isinstance(col, list) else col for col in val['cols']]
            sys_dict[key] = {
                'cols': cols_as_tuples,
                'col_names': val['col_names'],
                'type': val['type'],
                'name': val['name'],
                'label': val.get('label', val.get('name', key)),
                'description': val.get('description', ''),
                'variation': val['variation']
            }
            # Reconstruct original keys from cols (remove 'truth' prefix)
            # cols are tuples like ('truth', 'GENIE', ...), keys are ('GENIE', ...)
            # Skip this for summary keys where cols is None
            if cols_as_tuples is not None:
                for col in cols_as_tuples:
                    if isinstance(col, (list, tuple)) and len(col) > 0:
                        col_list = list(col)
                        # Remove 'truth' prefix if present
                        if col_list[0] == 'truth':
                            reconstructed_keys.append(tuple(col_list[1:]))
                        else:
                            reconstructed_keys.append(tuple(col_list))
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
        
        instance.sys_dict = sys_dict
        instance.systematics = {}
        instance._initialize_systematic_dicts()
        
        # Now load the systematics data
        instance.load(load_dir,ignore_keys=ignore_keys,ignore_types=ignore_types,select_types=select_types,metadata_dir=metadata_dir)
        
        return instance
    
    def load(self, load_dir,ignore_keys=[],ignore_types=[],select_types=[],metadata_dir='metadata'):
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
        metadata_dir : str, optional
            Directory to load the metadata from
        """
        if not self.systematics:
            raise ValueError('Systematics object must be initialized before loading. Use Systematics.from_saved() instead.')
        
        # Create a mapping from name to systematic key
        name_to_key = {sdict['name']: key for key, sdict in self.systematics.items()}
        
        for subfolder in tqdm(os.listdir(load_dir),desc='Loading systematics',unit=' subfolder'):
            if subfolder in ignore_keys:
            #print(f'Ignoring {subfolder} because it is in ignore_keys: {ignore_keys}')
                continue
            subfolder_path = os.path.join(load_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            
            # Skip metadata directory
            if subfolder == metadata_dir:
                continue
            
            
            # Find the systematic key - try by name first, then by key directly
            if subfolder in name_to_key:
                sys_key = name_to_key[subfolder]
            elif subfolder in self.systematics:
                # For summary systematics where name == key
                sys_key = subfolder
            else:
                # Create it if it doesn't exist (e.g., summary systematics loaded via from_saved)
                sys_key = subfolder
                self.systematics[sys_key] = Systematics._get_dict_template().copy()
                self.systematics[sys_key]['name'] = subfolder
                self.systematics[sys_key]['label'] = subfolder
                self.systematics[sys_key]['description'] = ''
            

            if self.systematics[sys_key]['type'] in ignore_types or (select_types != [] and self.systematics[sys_key]['type'] not in select_types):
                self.systematics.pop(sys_key)
                continue
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                if not os.path.isfile(file_path):
                    continue
                    
                # Check if file matches current variable_name prefix
                filename_prefix = f'{self.variable_name}_'
                if not file.startswith(filename_prefix):
                    # Skip files that don't belong to this variable name
                    continue
                    
                # Remove extension and variable_name prefix
                key_name = file.replace('.csv', '').replace('.txt', '')
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
                else:
                    #raise ValueError(f'File {file} has unknown extension, only .csv and .txt are supported')
                    continue
        print(f'Loaded {len(self.systematics)} systematics from {load_dir}')