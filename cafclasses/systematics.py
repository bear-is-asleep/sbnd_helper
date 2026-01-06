import numpy as np
import matplotlib.pyplot as plt
from sbnd.plotlibrary import makeplot
from sbnd.general import plotters
from sbnd.stats.stats import (
    construct_covariance, construct_correlation_matrix, 
    get_fractional_uncertainty, get_total_unc, convert_smearing_to_response
)


def _get_smear_matrix(true_var, reco_var, bins, weights=None):
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
    if weights is not None:
        reco_vs_true, _, _ = np.histogram2d(true_var, reco_var, bins=bins, weights=weights)
    else:
        reco_vs_true, _, _ = np.histogram2d(true_var, reco_var, bins=bins)
    return reco_vs_true


class Systematics:
    """Class for handling systematic uncertainty processing and computation."""
    
    def __init__(self, keys, costheta_bins, momentum_bins, differential_edges,
                 costheta_sigma_tilde, momentum_sigma_tilde, differential_sigma_tilde,
                 costheta_sel, momentum_sel, differential_sel,
                 costheta_sel_background, momentum_sel_background, differential_sel_background,
                 costheta_sel_truth, momentum_sel_truth, differential_sel_truth,
                 costheta_response, momentum_response, differential_response,
                 costheta_eff_truth, momentum_eff_truth, differential_eff_truth,
                 xsec_unit):
        """
        Initialize Systematics class.
        
        Parameters
        ----------
        keys : array-like
            Keys from truth level to identify systematics
        costheta_bins : array-like
            Bin edges for costheta
        momentum_bins : array-like
            Bin edges for momentum
        differential_edges : array-like
            Bin edges for differential (2D) binning
        costheta_sigma_tilde : array-like
            CV sigma_tilde for costheta
        momentum_sigma_tilde : array-like
            CV sigma_tilde for momentum
        differential_sigma_tilde : array-like
            CV sigma_tilde for differential
        costheta_sel : array-like
            CV selected events (reco) + background for costheta
        momentum_sel : array-like
            CV selected events (reco) + background for momentum
        differential_sel : array-like
            CV selected events (reco) + background for differential
        costheta_sel_background : array-like
            CV selected events (reco) + background for costheta
        momentum_sel_background : array-like
            CV selected events (reco) + background for momentum
        differential_sel_background : array-like
            CV selected events (reco) + background for differential
        costheta_sel_truth : array-like
            CV selected events (truth) + background for costheta
        momentum_sel_truth : array-like
            CV selected events (truth) + background for momentum
        differential_sel_truth : array-like
            CV selected events (truth) + background for differential
        costheta_response : ndarray
            CV response matrix for costheta
        momentum_response : ndarray
            CV response matrix for momentum
        differential_response : ndarray
            CV response matrix for differential
        costheta_eff_truth : array-like
            CV efficiency (truth) for costheta
        momentum_eff_truth : array-like
            CV efficiency (truth) for momentum
        differential_eff_truth : array-like
            CV efficiency (truth) for differential
        xsec_unit : float
            Cross section unit conversion factor
        """
        # Store bins
        self.costheta_bins = np.array(costheta_bins)
        self.momentum_bins = np.array(momentum_bins)
        self.differential_edges = np.array(differential_edges)
        
        # Store CV data
        self.costheta_sigma_tilde = np.array(costheta_sigma_tilde)
        self.momentum_sigma_tilde = np.array(momentum_sigma_tilde)
        self.differential_sigma_tilde = np.array(differential_sigma_tilde)
        
        self.costheta_sel = np.array(costheta_sel)
        self.momentum_sel = np.array(momentum_sel)
        self.differential_sel = np.array(differential_sel)

        self.costheta_sel_background = np.array(costheta_sel_background)
        self.momentum_sel_background = np.array(momentum_sel_background)
        self.differential_sel_background = np.array(differential_sel_background)
        
        self.costheta_sel_truth = np.array(costheta_sel_truth)
        self.momentum_sel_truth = np.array(momentum_sel_truth)
        self.differential_sel_truth = np.array(differential_sel_truth)
        
        self.costheta_response = np.array(costheta_response)
        self.momentum_response = np.array(momentum_response)
        self.differential_response = np.array(differential_response)
        
        self.costheta_eff_truth = np.array(costheta_eff_truth)
        self.momentum_eff_truth = np.array(momentum_eff_truth)
        self.differential_eff_truth = np.array(differential_eff_truth)
        
        self.xsec_unit = xsec_unit
        
        # Initialize systematic dictionary
        self.sys_dict = self.get_sys_keydict(keys)
        
        # Initialize systematic results dictionary
        self.systematics = {}
        self._initialize_systematic_dicts()
    
    def _initialize_systematic_dicts(self):
        """Initialize the systematic dictionaries with the template structure."""
        dict_template = {
            'cols': None,
            'col_names': None,
            'type': None,
            'name': None,
            'variation': None,
            # Sigma tilde
            'costheta_sigma_tilde': [],
            'momentum_sigma_tilde': [],
            'differential_sigma_tilde': [],
            # Reco event rate
            'costheta_sel': [],
            'momentum_sel': [],
            'differential_sel': [],
            # Xsec covariance
            'costheta_xsec_cov': None,
            'momentum_xsec_cov': None,
            'differential_xsec_cov': None,
            # Event covariance
            'costheta_event_cov': None,
            'momentum_event_cov': None,
            'differential_event_cov': None,
            # Xsec correlation
            'costheta_xsec_corr': None,
            'momentum_xsec_corr': None,
            'differential_xsec_corr': None,
            # Event correlation
            'costheta_event_corr': None,
            'momentum_event_corr': None,
            'differential_event_corr': None,
            # Xsec fractional covariance
            'costheta_xsec_fraccov': None,
            'momentum_xsec_fraccov': None,
            'differential_xsec_fraccov': None,
            # Event fractional covariance
            'costheta_event_fraccov': None,
            'momentum_event_fraccov': None,
            'differential_event_fraccov': None,
            # Xsec fractional uncertainty
            'costheta_xsec_fracunc': None,
            'momentum_xsec_fracunc': None,
            'differential_xsec_fracunc': None,
            # Event fractional uncertainty
            'costheta_event_fracunc': None,
            'momentum_event_fracunc': None,
            'differential_event_fracunc': None,
            # Event total uncertainty
            'costheta_event_totalunc': None,
            'momentum_event_totalunc': None,
            'differential_event_totalunc': None,
            # Xsec total uncertainty
            'costheta_xsec_totalunc': None,
            'momentum_xsec_totalunc': None,
            'differential_xsec_totalunc': None
        }
        
        for key in self.sys_dict:
            self.systematics[key] = dict_template.copy()
            self.systematics[key]['cols'] = self.sys_dict[key]['cols']
            self.systematics[key]['col_names'] = self.sys_dict[key]['col_names']
            self.systematics[key]['type'] = self.sys_dict[key]['type']
            self.systematics[key]['name'] = self.sys_dict[key]['name']
            self.systematics[key]['variation'] = self.sys_dict[key]['variation']
    def __repr__(self):
        return f'Systematics(keys={self.sys_dict.keys()})'
    def get_sys_keydict(self,keys,pattern=['GENIE','Flux','Flux','Genie','stat']):
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
            return key
        def assign_type(key):
            if 'genie' in key.lower():
                return 'xsec'
            elif 'flux' in key.lower():
                return 'flux'
            elif 'stat' in key.lower():
                return 'stat'
            else:
                print(f'Unknown systematic type: {key}')
                return 'unknown'
        def assign_variation(key):
            if 'multisigma' in key.lower():
                return 'multisigma'
            elif 'multisim' in key.lower() or 'flux' in key.lower():
                return 'multisim'
            else:
                return 'unisim'
        sys_col_dict = {}
        cnt = -1
        for k in keys:
            #assert k[0] == 'truth', f'Systematics are not at truth level: {k}'
            candidate = k[0]
            passes = False
            for p in pattern:
                if p in candidate:
                    passes = True
                    break
            if not passes:
                continue
                    #print(candidate)
            if candidate not in sys_col_dict:
                sys_col_dict[candidate] = {'cols' : [], 'col_names' : [], 'type' : assign_type(candidate), 'name' : format_name(candidate), 'variation' : assign_variation(k[0])}
            cnt+=1
            sys_col_dict[candidate]['cols'].append(tuple(['truth']+list(k)))
            sys_col_dict[candidate]['col_names'].append('_'.join(list(k)).rstrip('_'))
        return sys_col_dict
    def process_systematics(self, mc_signal_data, mc_sel_signal_data,
                           genweights_sig, genweights_sel, genweights_sel_background,
                           true_sig_costheta, true_sel_costheta, true_sel_background_costheta,
                           reco_sel_costheta, reco_sel_background_costheta,
                           true_sig_momentum, true_sel_momentum, true_sel_background_momentum,
                           reco_sel_momentum, reco_sel_background_momentum,
                           true_sig_differential, true_sel_differential, true_sel_background_differential,
                           reco_sel_differential, reco_sel_background_differential,
                           progress_bar=True):
        """
        Process all systematic universes and compute histograms, efficiencies, 
        smearing/response matrices, and sigma_tilde values.
        
        Parameters
        ----------
        mc_signal_data : data frame
            MC signal slice with data attribute containing systematic columns
        mc_sel_signal_data : data frame
            MC selected signal slice with data attribute containing systematic columns
        genweights_sig : array-like
            Generator weights for signal
        genweights_sel : array-like
            Generator weights for selected
        genweights_sel_background : array-like
            Generator weights for selected background
        true_sig_costheta : array-like
            True costheta for signal
        true_sel_costheta : array-like
            True costheta for selected
        true_sel_background_costheta : array-like
            True costheta for selected background
        reco_sel_costheta : array-like
            Reco costheta for selected
        reco_sel_background_costheta : array-like
            Reco costheta for selected background
        true_sig_momentum : array-like
            True momentum for signal
        true_sel_momentum : array-like
            True momentum for selected
        true_sel_background_momentum : array-like
            True momentum for selected background
        reco_sel_momentum : array-like
            Reco momentum for selected
        reco_sel_background_momentum : array-like
            Reco momentum for selected background
        true_sig_differential : array-like
            True differential bin for signal
        true_sel_differential : array-like
            True differential bin for selected
        true_sel_background_differential : array-like
            True differential bin for selected background
        reco_sel_differential : array-like
            Reco differential bin for selected
        reco_sel_background_differential : array-like
            Reco differential bin for selected background
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
            sys_dict['costheta_sigma_tilde'] = []
            sys_dict['momentum_sigma_tilde'] = []
            sys_dict['differential_sigma_tilde'] = []
            sys_dict['costheta_sel'] = []
            sys_dict['momentum_sel'] = []
            sys_dict['differential_sel'] = []
            
            for col, col_name in zip(cols, col_names):
                # Skip CV for multisigma variations
                if '_cv' in col_name and sys_dict['variation'] == 'multisigma':
                    continue
                
                # Get the distribution for selected and signal
                # Costheta
                _costheta_sig_truth, _ = np.histogram(
                    true_sig_costheta, bins=self.costheta_bins,
                    weights=genweights_sig * mc_signal_data[col]
                )
                _costheta_sel_truth, _ = np.histogram(
                    true_sel_costheta, bins=self.costheta_bins,
                    weights=genweights_sel * mc_sel_signal_data[col]
                )
                _costheta_sel_background_truth, _ = np.histogram(
                    true_sel_background_costheta, bins=self.costheta_bins,
                    weights=genweights_sel_background
                )
                
                _costheta_sig, _ = np.histogram(
                    true_sig_costheta, bins=self.costheta_bins,
                    weights=genweights_sig * mc_signal_data[col]
                )
                _costheta_sel, _ = np.histogram(
                    reco_sel_costheta, bins=self.costheta_bins,
                    weights=genweights_sel * mc_sel_signal_data[col]
                )
                _costheta_sel_background, _ = np.histogram(
                    reco_sel_background_costheta, bins=self.costheta_bins,
                    weights=genweights_sel_background
                )
                
                # Momentum
                _momentum_sig_truth, _ = np.histogram(
                    true_sig_momentum, bins=self.momentum_bins,
                    weights=genweights_sig * mc_signal_data[col]
                )
                _momentum_sel_truth, _ = np.histogram(
                    true_sel_momentum, bins=self.momentum_bins,
                    weights=genweights_sel * mc_sel_signal_data[col]
                )
                _momentum_sel_background_truth, _ = np.histogram(
                    true_sel_background_momentum, bins=self.momentum_bins,
                    weights=genweights_sel_background
                )
                
                _momentum_sig, _ = np.histogram(
                    true_sig_momentum, bins=self.momentum_bins,
                    weights=genweights_sig * mc_signal_data[col]
                )
                _momentum_sel, _ = np.histogram(
                    reco_sel_momentum, bins=self.momentum_bins,
                    weights=genweights_sel * mc_sel_signal_data[col]
                )
                _momentum_sel_background, _ = np.histogram(
                    reco_sel_background_momentum, bins=self.momentum_bins,
                    weights=genweights_sel_background
                )
                
                # Differential
                _differential_sig_truth, _ = np.histogram(
                    true_sig_differential, bins=self.differential_edges,
                    weights=genweights_sig * mc_signal_data[col]
                )
                _differential_sel_truth, _ = np.histogram(
                    true_sel_differential, bins=self.differential_edges,
                    weights=genweights_sel * mc_sel_signal_data[col]
                )
                _differential_sel_background_truth, _ = np.histogram(
                    true_sel_background_differential, bins=self.differential_edges,
                    weights=genweights_sel_background
                )
                
                _differential_sig, _ = np.histogram(
                    true_sig_differential, bins=self.differential_edges,
                    weights=genweights_sig * mc_signal_data[col]
                )
                _differential_sel, _ = np.histogram(
                    reco_sel_differential, bins=self.differential_edges,
                    weights=genweights_sel * mc_sel_signal_data[col]
                )
                _differential_sel_background, _ = np.histogram(
                    reco_sel_background_differential, bins=self.differential_edges,
                    weights=genweights_sel_background
                )
                
                # Get the efficiencies
                _costheta_eff_truth = _costheta_sel_truth / _costheta_sig_truth
                _momentum_eff_truth = _momentum_sel_truth / _momentum_sig_truth
                _differential_eff_truth = _differential_sel_truth / _differential_sig_truth
                
                # Get the smearing and response matrices
                _costheta_smearing = _get_smear_matrix(
                    true_sel_costheta, reco_sel_costheta, self.costheta_bins,
                    weights=genweights_sel * mc_sel_signal_data[col]
                )
                _momentum_smearing = _get_smear_matrix(
                    true_sel_momentum, reco_sel_momentum, self.momentum_bins,
                    weights=genweights_sel * mc_sel_signal_data[col]
                )
                _differential_smearing = _get_smear_matrix(
                    true_sel_differential, reco_sel_differential, self.differential_edges,
                    weights=genweights_sel * mc_sel_signal_data[col]
                )
                
                # Use universe efficiency for GENIE xsec uncertainties, CV efficiency otherwise
                if 'GENIE' in key:
                    _costheta_response = convert_smearing_to_response(
                        _costheta_smearing, _costheta_eff_truth
                    )
                    _momentum_response = convert_smearing_to_response(
                        _momentum_smearing, _momentum_eff_truth
                    )
                    _differential_response = convert_smearing_to_response(
                        _differential_smearing, _differential_eff_truth
                    )
                else:
                    _costheta_response = convert_smearing_to_response(
                        _costheta_smearing, self.costheta_eff_truth
                    )
                    _momentum_response = convert_smearing_to_response(
                        _momentum_smearing, self.momentum_eff_truth
                    )
                    _differential_response = convert_smearing_to_response(
                        _differential_smearing, self.differential_eff_truth
                    )
                
                # Compute sigma_tilde
                _costheta_sigma_tilde = self.xsec_unit * (
                    _costheta_response @ _costheta_sel_truth + _costheta_sel_background_truth
                )
                _momentum_sigma_tilde = self.xsec_unit * (
                    _momentum_response @ _momentum_sel_truth + _momentum_sel_background_truth
                )
                _differential_sigma_tilde = self.xsec_unit * (
                    _differential_response @ _differential_sel_truth + _differential_sel_background_truth
                )
                
                # Store results
                sys_dict['costheta_sigma_tilde'].append(_costheta_sigma_tilde)
                sys_dict['momentum_sigma_tilde'].append(_momentum_sigma_tilde)
                sys_dict['differential_sigma_tilde'].append(_differential_sigma_tilde)
                sys_dict['costheta_sel'].append(_costheta_sel + _costheta_sel_background)
                sys_dict['momentum_sel'].append(_momentum_sel + _momentum_sel_background)
                sys_dict['differential_sel'].append(_differential_sel + _differential_sel_background)
    
    def compute_covariances(self):
        """Compute covariance matrices, correlations, and uncertainties for each systematic."""
        for key, sys_dict in self.systematics.items():
            # Costheta
            # Xsec
            _xsec_cov, _xsec_frac, _xsec_corr, _xsec_fracunc = construct_covariance(
                self.costheta_sigma_tilde, sys_dict['costheta_sigma_tilde'], assert_cov=False
            )
            sys_dict['costheta_xsec_cov'] = _xsec_cov
            sys_dict['costheta_xsec_fraccov'] = _xsec_frac
            sys_dict['costheta_xsec_corr'] = _xsec_corr
            sys_dict['costheta_xsec_fracunc'] = _xsec_fracunc
            _xsec_total_unc = get_total_unc(
                self.costheta_sigma_tilde, sys_dict['costheta_xsec_fracunc']
            )
            sys_dict['costheta_xsec_totalunc'] = _xsec_total_unc
            
            # Event
            _cov, _frac, _corr, _fracunc = construct_covariance(
                self.costheta_sel+self.costheta_sel_background, sys_dict['costheta_sel'], assert_cov=False
            )
            sys_dict['costheta_event_cov'] = _cov
            sys_dict['costheta_event_fraccov'] = _frac
            sys_dict['costheta_event_corr'] = _corr
            sys_dict['costheta_event_fracunc'] = _fracunc
            _total_unc = get_total_unc(
                self.costheta_sel+self.costheta_sel_background, sys_dict['costheta_event_fracunc']
            )
            sys_dict['costheta_event_totalunc'] = _total_unc
            
            # Momentum
            # Xsec
            _xsec_cov, _xsec_frac, _xsec_corr, _xsec_fracunc = construct_covariance(
                self.momentum_sigma_tilde, sys_dict['momentum_sigma_tilde'], assert_cov=False
            )
            sys_dict['momentum_xsec_cov'] = _xsec_cov
            sys_dict['momentum_xsec_fraccov'] = _xsec_frac
            sys_dict['momentum_xsec_corr'] = _xsec_corr
            sys_dict['momentum_xsec_fracunc'] = _xsec_fracunc
            _xsec_total_unc = get_total_unc(
                self.momentum_sigma_tilde, sys_dict['momentum_xsec_fracunc']
            )
            sys_dict['momentum_xsec_totalunc'] = _xsec_total_unc
            
            # Event
            _cov, _frac, _corr, _fracunc = construct_covariance(
                self.momentum_sel+self.momentum_sel_background, sys_dict['momentum_sel'], assert_cov=False
            )
            sys_dict['momentum_event_cov'] = _cov
            sys_dict['momentum_event_fraccov'] = _frac
            sys_dict['momentum_event_corr'] = _corr
            sys_dict['momentum_event_fracunc'] = _fracunc
            _total_unc = get_total_unc(
                self.momentum_sel+self.momentum_sel_background, sys_dict['momentum_event_fracunc']
            )
            sys_dict['momentum_event_totalunc'] = _total_unc
            
            # Differential
            # Xsec
            _xsec_cov, _xsec_frac, _xsec_corr, _xsec_fracunc = construct_covariance(
                self.differential_sigma_tilde, sys_dict['differential_sigma_tilde'], assert_cov=False
            )
            sys_dict['differential_xsec_cov'] = _xsec_cov
            sys_dict['differential_xsec_fraccov'] = _xsec_frac
            sys_dict['differential_xsec_corr'] = _xsec_corr
            sys_dict['differential_xsec_fracunc'] = _xsec_fracunc
            _xsec_total_unc = get_total_unc(
                self.differential_sigma_tilde, sys_dict['differential_xsec_fracunc']
            )
            sys_dict['differential_xsec_totalunc'] = _xsec_total_unc
            
            # Event
            _cov, _frac, _corr, _fracunc = construct_covariance(
                self.differential_sel+self.differential_sel_background, sys_dict['differential_sel'], assert_cov=False
            )
            sys_dict['differential_event_cov'] = _cov
            sys_dict['differential_event_fraccov'] = _frac
            sys_dict['differential_event_corr'] = _corr
            sys_dict['differential_event_fracunc'] = _fracunc
            _total_unc = get_total_unc(
                self.differential_sel+self.differential_sel_background, sys_dict['differential_event_fracunc']
            )
            sys_dict['differential_event_totalunc'] = _total_unc
    
    def combine_summaries(self,summary_keys = ['xsec', 'flux', 'total']):
        """Combine systematics into summary groups (xsec, flux, total).
        
        Parameters
        ----------
        summary_keys : list
            List of summary keys to combine
        """
        
        # Create summary dictionaries
        dict_template = {
            'cols': None,
            'col_names': None,
            'type': None,
            'name': None,
            'variation': None,
            # Sigma tilde
            'costheta_sigma_tilde': None,
            'momentum_sigma_tilde': None,
            'differential_sigma_tilde': None,
            # Reco event rate
            'costheta_sel': None,
            'momentum_sel': None,
            'differential_sel': None,
            # Xsec covariance
            'costheta_xsec_cov': None,
            'momentum_xsec_cov': None,
            'differential_xsec_cov': None,
            # Event covariance
            'costheta_event_cov': None,
            'momentum_event_cov': None,
            'differential_event_cov': None,
            # Xsec correlation
            'costheta_xsec_corr': None,
            'momentum_xsec_corr': None,
            'differential_xsec_corr': None,
            # Event correlation
            'costheta_event_corr': None,
            'momentum_event_corr': None,
            'differential_event_corr': None,
            # Xsec fractional covariance
            'costheta_xsec_fraccov': None,
            'momentum_xsec_fraccov': None,
            'differential_xsec_fraccov': None,
            # Event fractional covariance
            'costheta_event_fraccov': None,
            'momentum_event_fraccov': None,
            'differential_event_fraccov': None,
            # Xsec fractional uncertainty
            'costheta_xsec_fracunc': None,
            'momentum_xsec_fracunc': None,
            'differential_xsec_fracunc': None,
            # Event fractional uncertainty
            'costheta_event_fracunc': None,
            'momentum_event_fracunc': None,
            'differential_event_fracunc': None,
            # Event total uncertainty
            'costheta_event_totalunc': None,
            'momentum_event_totalunc': None,
            'differential_event_totalunc': None,
            # Xsec total uncertainty
            'costheta_xsec_totalunc': None,
            'momentum_xsec_totalunc': None,
            'differential_xsec_totalunc': None
        }
        
        for sk in summary_keys:
            self.systematics[sk] = dict_template.copy()
            self.systematics[sk]['cols'] = None
            self.systematics[sk]['col_names'] = None
            self.systematics[sk]['type'] = sk
            self.systematics[sk]['name'] = sk
            
            # Initialize covariance matrices
            n_costheta = len(self.costheta_bins) - 1
            n_momentum = len(self.momentum_bins) - 1
            n_differential = len(self.differential_edges) - 1
            
            self.systematics[sk]['costheta_event_cov'] = np.zeros((n_costheta, n_costheta))
            self.systematics[sk]['momentum_event_cov'] = np.zeros((n_momentum, n_momentum))
            self.systematics[sk]['differential_event_cov'] = np.zeros((n_differential, n_differential))
            
            self.systematics[sk]['costheta_xsec_cov'] = np.zeros((n_costheta, n_costheta))
            self.systematics[sk]['momentum_xsec_cov'] = np.zeros((n_momentum, n_momentum))
            self.systematics[sk]['differential_xsec_cov'] = np.zeros((n_differential, n_differential))
            
            self.systematics[sk]['costheta_xsec_fraccov'] = np.zeros((n_costheta, n_costheta))
            self.systematics[sk]['momentum_xsec_fraccov'] = np.zeros((n_momentum, n_momentum))
            self.systematics[sk]['differential_xsec_fraccov'] = np.zeros((n_differential, n_differential))
            
            self.systematics[sk]['costheta_event_fraccov'] = np.zeros((n_costheta, n_costheta))
            self.systematics[sk]['momentum_event_fraccov'] = np.zeros((n_momentum, n_momentum))
            self.systematics[sk]['differential_event_fraccov'] = np.zeros((n_differential, n_differential))
        
        # Sum covariance matrices
        for key, sys_dict in self.systematics.items():
            if sys_dict['type'] == sys_dict['name']:  # Skip summary keys themselves
                continue
            
            for sk in summary_keys:
                if (sys_dict['type'] == sk) or (sk == 'total'):
                    # Event covariance
                    if sys_dict['costheta_event_cov'] is not None:
                        self.systematics[sk]['costheta_event_cov'] += sys_dict['costheta_event_cov']
                        self.systematics[sk]['momentum_event_cov'] += sys_dict['momentum_event_cov']
                        self.systematics[sk]['differential_event_cov'] += sys_dict['differential_event_cov']
                    
                    # Xsec covariance
                    if sys_dict['costheta_xsec_cov'] is not None:
                        self.systematics[sk]['costheta_xsec_cov'] += sys_dict['costheta_xsec_cov']
                        self.systematics[sk]['momentum_xsec_cov'] += sys_dict['momentum_xsec_cov']
                        self.systematics[sk]['differential_xsec_cov'] += sys_dict['differential_xsec_cov']
                    
                    # Event fractional covariance
                    if sys_dict['costheta_event_fraccov'] is not None:
                        self.systematics[sk]['costheta_event_fraccov'] += sys_dict['costheta_event_fraccov']
                        self.systematics[sk]['momentum_event_fraccov'] += sys_dict['momentum_event_fraccov']
                        self.systematics[sk]['differential_event_fraccov'] += sys_dict['differential_event_fraccov']
                    
                    # Xsec fractional covariance
                    if sys_dict['costheta_xsec_fraccov'] is not None:
                        self.systematics[sk]['costheta_xsec_fraccov'] += sys_dict['costheta_xsec_fraccov']
                        self.systematics[sk]['momentum_xsec_fraccov'] += sys_dict['momentum_xsec_fraccov']
                        self.systematics[sk]['differential_xsec_fraccov'] += sys_dict['differential_xsec_fraccov']
        
        # Compute fractional uncertainties and correlations from combined covariances
        for sk in summary_keys:
            # Event fractional uncertainty
            self.systematics[sk]['costheta_event_fracunc'] = get_fractional_uncertainty(
                self.costheta_sel+self.costheta_sel_background, self.systematics[sk]['costheta_event_cov']
            )
            self.systematics[sk]['momentum_event_fracunc'] = get_fractional_uncertainty(
                self.momentum_sel+self.momentum_sel_background, self.systematics[sk]['momentum_event_cov']
            )
            self.systematics[sk]['differential_event_fracunc'] = get_fractional_uncertainty(
                self.differential_sel+self.differential_sel_background, self.systematics[sk]['differential_event_cov']
            )
            
            # Xsec fractional uncertainty
            self.systematics[sk]['costheta_xsec_fracunc'] = get_fractional_uncertainty(
                self.costheta_sigma_tilde, self.systematics[sk]['costheta_xsec_cov']
            )
            self.systematics[sk]['momentum_xsec_fracunc'] = get_fractional_uncertainty(
                self.momentum_sigma_tilde, self.systematics[sk]['momentum_xsec_cov']
            )
            self.systematics[sk]['differential_xsec_fracunc'] = get_fractional_uncertainty(
                self.differential_sigma_tilde, self.systematics[sk]['differential_xsec_cov']
            )
            
            # Event correlation
            self.systematics[sk]['costheta_event_corr'], _ = construct_correlation_matrix(
                self.systematics[sk]['costheta_event_cov']
            )
            self.systematics[sk]['momentum_event_corr'], _ = construct_correlation_matrix(
                self.systematics[sk]['momentum_event_cov']
            )
            self.systematics[sk]['differential_event_corr'], _ = construct_correlation_matrix(
                self.systematics[sk]['differential_event_cov']
            )
            
            # Xsec correlation
            self.systematics[sk]['costheta_xsec_corr'], _ = construct_correlation_matrix(
                self.systematics[sk]['costheta_xsec_cov']
            )
            self.systematics[sk]['momentum_xsec_corr'], _ = construct_correlation_matrix(
                self.systematics[sk]['momentum_xsec_cov']
            )
            self.systematics[sk]['differential_xsec_corr'], _ = construct_correlation_matrix(
                self.systematics[sk]['differential_xsec_cov']
            )
            
            # Event total uncertainty
            self.systematics[sk]['costheta_event_totalunc'] = get_total_unc(
                self.costheta_sel+self.costheta_sel_background, self.systematics[sk]['costheta_event_fracunc']
            )
            self.systematics[sk]['momentum_event_totalunc'] = get_total_unc(
                self.momentum_sel+self.momentum_sel_background, self.systematics[sk]['momentum_event_fracunc']
            )
            self.systematics[sk]['differential_event_totalunc'] = get_total_unc(
                self.differential_sel+self.differential_sel_background, self.systematics[sk]['differential_event_fracunc']
            )
            
            # Xsec total uncertainty
            self.systematics[sk]['costheta_xsec_totalunc'] = get_total_unc(
                self.costheta_sigma_tilde, self.systematics[sk]['costheta_xsec_fracunc']
            )
            self.systematics[sk]['momentum_xsec_totalunc'] = get_total_unc(
                self.momentum_sigma_tilde, self.systematics[sk]['momentum_xsec_fracunc']
            )
            self.systematics[sk]['differential_xsec_totalunc'] = get_total_unc(
                self.differential_sigma_tilde, self.systematics[sk]['differential_xsec_fracunc']
            )
    
    def plot_event_rate_errs(self, base_key, bins, exclude_keys=[], include_keys=[],
                            xlabel='', max_uncs=None, fig=None, ax=None):
        """
        Plot the event rate uncertainties for a given base key.
        
        Parameters
        ----------
        base_key : str
            The base key to plot (e.g., 'costheta_event', 'momentum_xsec')
        bins : array-like
            The bins to plot
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
        # Sort systematics by total uncertainty (largest first)
        sorted_items = sorted(
            self.systematics.items(),
            key=lambda x: x[1].get(f'{base_key}_totalunc', 0),
            reverse=True
        )
        
        if max_uncs is None:
            max_uncs = len(sorted_items)
        
        if len(include_keys) == 0:
            include_keys = [item[0] for item in sorted_items]
            if max_uncs == len(sorted_items):
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
            
            if frac_unc is None:
                continue
            
            label = f"{sys_dict['name']} ({total_unc*100:.1f}%)"
            #color looks bad, skip for now
            makeplot.plot_hist_edges(
                bins, frac_unc, None, ax=ax, label=label#, color=c
            )
        
        ax.legend(ncol=int(np.ceil(max_uncs/24)), bbox_to_anchor=(1.05, 1.05))
        ax.set_xlabel(xlabel)
        if 'xsec' in base_key:
            ax.set_ylabel(r'Cross Section Uncertainty')
        else:
            ax.set_ylabel(r'Event Rate Uncertainty')
        if 'momentum' in base_key:
            ax.set_xlim(0, 4)
        
        return [fig,ax,None]
    
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
            
            # Determine variable and bins
            set_axlims = False
            if 'costheta' in plotkey:
                axlabel = r'Reconstructed $\cos\theta_{\mu}$'
                bins = self.costheta_bins
            elif 'momentum' in plotkey:
                axlabel = r'Reconstructed $p_{\mu}$ (GeV)'
                set_axlims = True
                bins = self.momentum_bins
            elif 'differential' in plotkey:
                axlabel = r'Reconstructed 2D Bin ID'
                bins = self.differential_edges
            else:
                continue
            
            title = f"{sys_dict['name']} {plotkey.replace('_', ' ')}"
            name = title.replace(' ', '_')
            
            fig, ax = plt.subplots()
            ax.set_title(title)
            im = ax.pcolormesh(bins, bins, sys_dict[plotkey], cmap=cmap)
            ax.set_xlabel(axlabel)
            ax.set_ylabel(axlabel)
            fig.colorbar(im, ax=ax)
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
    
    def plot_sigma_tilde_distributions(self, sys_key, plot_dir=None, save_plots=True):
        """
        Plot sigma_tilde distributions for CV and all universe variations.
        
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
            Dictionary of figures for each variable
        """
        if sys_key not in self.systematics:
            raise ValueError(f"Systematic key '{sys_key}' not found")
        
        sys_dict = self.systematics[sys_key]
        figs = {}
        
        histkeys = [k for k in sys_dict.keys() if 'sigma_tilde' in k]
        
        for plotkey in histkeys:
            if sys_dict[plotkey] is None or len(sys_dict[plotkey]) == 0:
                continue
            
            subfolder = 'dist'
            set_xlim = False
            
            if 'costheta' in plotkey:
                axlabel = r'Reconstructed $\cos\theta_{\mu}$'
                bins = self.costheta_bins
                sigma_tilde = self.costheta_sigma_tilde
            elif 'momentum' in plotkey:
                axlabel = r'Reconstructed $p_{\mu}$ (GeV)'
                set_xlim = True
                bins = self.momentum_bins
                sigma_tilde = self.momentum_sigma_tilde
            elif 'differential' in plotkey:
                axlabel = r'Reconstructed 2D Bin ID'
                bins = self.differential_edges
                sigma_tilde = self.differential_sigma_tilde
            else:
                continue
            
            title = f"{sys_dict['name']} {plotkey.replace('_', ' ')}"
            name = title.replace(' ', '_')
            
            fig, ax = plt.subplots()
            makeplot.plot_hist_edges(
                bins, sigma_tilde, None, ax=ax, color='black', label='Signal', alpha=1.
            )
            for dist in sys_dict[plotkey]:
                makeplot.plot_hist_edges(
                    bins, dist, None, ax=ax, color='blue', label=None,
                    alpha=1./len(sys_dict[plotkey])
                )
            ax.legend()
            ax.set_xlabel(axlabel)
            ax.set_ylabel(r'$\tilde{\sigma}$')
            ax.set_title(title)
            if set_xlim:
                ax.set_xlim(0, 4.)
            
            figs[plotkey] = [fig,ax,im]
            if save_plots:
                if plot_dir is not None:
                    plotters.save_plot(name, fig=fig, folder_name=f'{plot_dir}/{subfolder}')
                else:
                    raise ValueError('plot_dir is None and save_plots is True')
                plt.close(fig)
            else:
                plt.show(fig=fig)
        
        return figs

