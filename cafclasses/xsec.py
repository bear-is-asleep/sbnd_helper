from analysis_village.unfolding import wienersvd
import numpy as np
from sbnd.stats import stats

class XSec:
    def __init__(self, response, sig_truth, xsec_unit, bins, 
        bin_widths=None,name='xsec',variable=''):
        """
        Parameters
        ----------
        response: array, response matrix
        sig_truth: array, signal distribution binned by true variable
        sel_reco: array, selection distribution binned by reco variable
        xsec_unit: float, xsec unit
        bins: array, binnning of variable
        name: str, name of the xsec
        variable: str, variable of the xsec
        """
        self.response = response
        self.sig_truth = sig_truth
        self.xsec_unit = xsec_unit
        self.bins = bins
        self.bin_centers = (bins[:-1] + bins[1:]) / 2
        if bin_widths is not None:
            self.bin_widths = bin_widths
        else:
            self.bin_widths = bins[1:] - bins[:-1]
        self.name = name
        self.variable = variable
        assert len(self.bin_widths) == len(self.sig_truth), f'bin_widths and sig_truth must be the same length: {len(self.bin_widths)} != {len(self.sig_truth)}'
        #self.unfold_dict = self.unfold(C_type, Norm_type)

    def unfold(self, xsec_cov, sel, C_type=2, Norm_type=0.5, scale_factor=1.,verbose=False,eps=1e-6):
        """
        Unfold the xsec using Wiener-SVD
        Parameters
        ----------
        C_type: int, type specifier for the smoothness matrix
        Norm_type: float, normalization exponent for Signal
        sel: array, selection distribution binned by reco variable
        scale_factor: float, scale factor for the signal and measured outputs
        verbose: bool, print verbose output

        """
        sig = self.sig_truth*self.xsec_unit
        sel = sel*self.xsec_unit
        #Replace any sig or sel that are 0 with eps times minimum value
        min_value = np.min(np.abs(sig[sig != 0]))
        sig = np.where(sig == 0, eps*min_value, sig)
        min_value = np.min(np.abs(sel[sel != 0]))
        sel = np.where(sel == 0, eps*min_value, sel)
        if verbose:
            print(f'sig max: {np.max(sig)}')
            print(f'sig min: {np.min(sig)}')
            print(f'sel max: {np.max(sel)}')
            print(f'sel min: {np.min(sel)}')
            print(f'xsec_unit: {self.xsec_unit}')
            print(f'xsec_cov max: {np.max(xsec_cov)}')
            print(f'xsec_cov min: {np.min(xsec_cov)}')
            print(f'bins: {self.bins}')
        # Replace 0 values of cov with tiny
        #min_value = np.min(np.abs(xsec_cov[xsec_cov != 0]))
        #xsec_cov = np.where(xsec_cov == 0, eps*min_value, xsec_cov)
        #xsec_cov += eps*min_value*np.eye(xsec_cov.shape[0])
        #print(xsec_cov)
        self.unfold_dict = wienersvd.WienerSVD(
            self.response,
            sig,
            sel,
            xsec_cov, C_type, Norm_type, verbose=verbose)
    

        self.sig_smear = self.unfold_dict['AddSmear'] @ sig * scale_factor / self.bin_widths
        self.sig_unfold = self.unfold_dict['unfold'] * scale_factor / self.bin_widths
        self.unfold_dict['fracunc'] = stats.get_fractional_uncertainty(self.sig_unfold,self.unfold_dict['UnfoldCov'])
    
    def get_chi2(self, cov=None,scale_factor=1.):
        """
        Get the chi2,dof value for the unfolded signal and smeared signal
        """
        #Prescale everything by the scale factor
        sig_unfold = self.sig_unfold * scale_factor
        sig_smear = self.sig_smear * scale_factor
        if cov is None:
            cov = self.unfold_dict['SystUnfoldCov']
        else:
            cov = cov * scale_factor * scale_factor

        chi2, dof, pval = stats.calc_chi2(sig_unfold, sig_smear, cov)
        self.chi2 = chi2
        self.dof = dof
        self.pval = pval
        self.chi2_dof_pval_str = f'$\chi^2$/dof = {chi2:.2f}/{dof}, p = {pval:.2f}'
        return self.chi2, self.dof, self.pval, self.chi2_dof_pval_str