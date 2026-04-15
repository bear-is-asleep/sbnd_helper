from analysis_village.unfolding import wienersvd
import json
import os
import numpy as np
from sbnd.stats import stats
from sbnd.general import utils
import boost_histogram as bh
import uproot
import matplotlib.pyplot as plt
from sbnd.general import plotters
from sbnd.plotlibrary import makeplot

RETAINED_VARIANCE_TOL=0.9999
PINV_RCOND=1e-4

class XSec:
    def __init__(self, xsec_unit, bins,
                 bin_widths=None, name='xsec', variable='',yscale=1):
        """
        Parameters
        ----------
        xsec_unit: float, xsec unit
        bins: array, binnning of variable
        name: str, name of the xsec
        variable: str, variable of the xsec
        """
        self.xsec_unit = xsec_unit
        self.bins = bins
        self.bin_centers = (bins[:-1] + bins[1:]) / 2
        if bin_widths is not None:
            self.bin_widths = bin_widths
        else:
            self.bin_widths = bins[1:] - bins[:-1]
        self.name = name
        self.variable = variable
        self.yscale = yscale
        self.yscale_str = utils.get_scientific_str(1/self.yscale,round_to=1)
        self._resolve_labels(variable)
        # Dictionary of unfold results keyed by a user-provided name.
        # Each entry stores configuration, sig_smear, sig_unfold, covariances, chi2, dof, pval, etc.
        self.unfold_results = {}
    def __str__(self):
        return f'XSec(name={self.name}, variable={self.variable}), keys={list(self.unfold_results.keys())}'
    def __repr__(self):
        return self.__str__()

    def _resolve_labels(self, variable):
        if variable == 'costheta':
            variable_label = r'$\cos\theta_\mu$'
            self.unit = r'cm$^{2}$ nucleon$^{-1}$'
            self.xsec_label = r'$\frac{d\sigma}{d\cos\theta_\mu}$' + f'[{self.yscale_str}' + self.unit + ']'
        elif variable == 'momentum':
            variable_label = r'$p_\mu$ [GeV]'
            self.unit = r'cm$^{2}$ GeV$^{-1}$ nucleon$^{-1}$'
            self.xsec_label = r'$\frac{d\sigma}{dp_\mu}$' + f'[{self.yscale_str}' + self.unit + ']'
        elif variable == 'differential':
            variable_label = 'Bin ID'
            self.unit = r'cm$^{2}$ GeV$^{-1}$ nucleon$^{-1}$'
            self.xsec_label = r'$\frac{d^2\sigma}{dp_\mu d\cos\theta_\mu}$' + f'[{self.yscale_str}' + self.unit + ']'
        else:
            raise ValueError(f'Variable {variable} not supported')
        self.regularized_label = 'Regularized ' + variable_label
        self.true_label = 'True ' + variable_label
        self.reco_label = 'Reconstructed ' + variable_label

    def get_cov(self, fraccov, sigma_tilde):
        return stats.covariance_from_fraccov(fraccov, sigma_tilde)

    @staticmethod
    def _print_mode_chi2_diagnostics(diff, cov, chi2, header='Mode-space chi2 diagnostics'):
        cov = np.asarray(cov)
        if cov.ndim == 1:
            variances = cov
            mode_chi2 = diff ** 2 / variances
            total = np.sum(mode_chi2)
            order = np.argsort(mode_chi2)[::-1]
            print(header + ' (diagonal)')
            print(f'  sum(mode_chi2) = {total:.6f}  (sum in eigenbasis = {chi2:.6f})')
            print('  bin   variance      diff              chi2_contrib  frac')
            for k in order:
                frac = mode_chi2[k] / total if total > 0 else 0.0
                print(f'  {k:>4d}  {variances[k]:.3e}  {diff[k]: .3e}   {mode_chi2[k]:.3e}   {frac:.3f}')
        else:
            eigvals, eigvecs = np.linalg.eigh(cov)
            mode_coeffs = eigvecs.T @ diff
            mode_chi2 = (mode_coeffs ** 2) / eigvals
            total = np.sum(mode_chi2)
            order = np.argsort(mode_chi2)[::-1]
            print(header)
            print(f'  sum(mode_chi2) = {total:.6f}  (sum in eigenbasis = {chi2:.6f})')
            print('  mode  eigval        coeff(u^T diff)   chi2_contrib  frac')
            for k in order:
                frac = mode_chi2[k] / total if total > 0 else 0.0
                print(f'  {k:>4d}  {eigvals[k]:.3e}  {mode_coeffs[k]: .3e}   {mode_chi2[k]:.3e}   {frac:.3f}')

    def _init_unfold_record(self, name):
        self.unfold_results[name] = {
            'C_type': None,
            'Norm_type': None,
            'fractional_cov': None,
            'input_cov': None,
            'input_cov_norm': None,
            'input_cov_shape': None,
            'unfold_cov': None,
            'unfold_fraccov': None,
            'unfold_corr': None,
            'unfold_cov_norm': None,
            'unfold_cov_shape': None,
            'fracunc': None,
            'fracunc_unfold_norm': None,
            'fracunc_unfold_shape': None,
            'tikhonov_s_opt': None,
            'AddSmear': None,
            'unfold': None,
            'sig_truth': None,
            'sel_background': None,
            'sigma_tilde': None,
            'sig_smear': None,
            'sig_unfold': None,
            'chi2': None,
            'dof': None,
            'pval': None,
            'chi2_1d': None,
            'pval_1d': None,
            'chi2_1d_str': None,
            'response': None,
            'residuals': None,
            'residuals_norm': None,
        }
        return self.unfold_results[name]

    def _compute_wiener(
        self,
        cov,
        sel,
        sig_truth,
        response,
        mask_bins=None,
        stat_cov=None,
        sel_background=None,
        sigma_tilde=None,
        C_type=2,
        Norm_type=0.5,
        verbose=False,
        eps=1e-6,
        fractional_cov=False,
        tikhonov=False,
    ):
        """
        Core Wiener-SVD call shared by smear and unfold.
        Returns a dict with all raw and derived quantities.
        """
        if sel_background is not None:
            sigma_tilde = stats.compute_sigma_tilde(
                response,
                sig_truth,
                sel_background,
                self.xsec_unit,
            )
        elif sigma_tilde is None:
            raise ValueError('sigma_tilde is not set')
        if fractional_cov:
            cov = self.get_cov(cov, sigma_tilde)
            if stat_cov is not None:
                stat_cov = self.get_cov(stat_cov, sigma_tilde)

        input_cov = np.array(cov)

        sig = sig_truth * self.xsec_unit
        sel = sel * self.xsec_unit
        # Replace any sig or sel that are 0 with eps times minimum value
        min_value = np.min(np.abs(sig[sig != 0]))
        sig = np.where(sig == 0, eps * min_value, sig)
        min_value = np.min(np.abs(sel[sel != 0]))
        sel = np.where(sel == 0, eps * min_value, sel)
        if verbose:
            print(f'sig: {sig}')
            print(f'sel: {sel}')
            print(f'response shape: {response.shape}')
            print(f'sig@response: {sig @ response}')
            print(f'sig@response - sel: {sig @ response - sel}')
            print(f'sqrt(cov diag): {np.sqrt(np.diag(input_cov))}')
            print(f'sqrt(cov diag)/sel*100: {(np.sqrt(np.diag(input_cov)) / sel) * 100}')
            print(f'sigma_tilde: {sigma_tilde}')
            print(f'xsec_unit: {self.xsec_unit}')
            print(f'bins: {self.bins}')
            print(f'bin_widths: {self.bin_widths}')

        unfold_dict = wienersvd.WienerSVD(
            response,
            sig,
            sel,
            input_cov,
            C_type,
            Norm_type
        )

        sig_smear = unfold_dict['AddSmear'] @ sig / self.bin_widths
        sig_unfold = unfold_dict['unfold'] / self.bin_widths
        # The bin widths need to be included since the covariance does not include them
        unfold_cov = unfold_dict['UnfoldCov']
        s_opt = None
        if tikhonov:
            unfold_cov, s_opt = stats.tikhonov_regularize_cov(unfold_cov, RETAINED_VARIANCE_TOL, diagnose=verbose)
            if verbose:
                print(f'[tikhonov] regularized unfold_cov (s_opt={s_opt:.3e})')
        sig_unfold_bw = sig_unfold * self.bin_widths
        sig_smear_bw = sig_smear * self.bin_widths

        # Norm (plus mixed) vs shape split; pred vector matches covariance scale (MiniBooNE style)
        input_cov_norm, input_cov_shape = wienersvd.Matrix_Decomp(sel, input_cov)
        unfold_cov_norm, unfold_cov_shape = wienersvd.Matrix_Decomp(sig_unfold_bw, unfold_cov)

        unfold_fraccov = stats.fraccov_from_covariance(unfold_cov, sig_unfold_bw)
        unfold_corr, _ = stats.construct_correlation_matrix(unfold_cov)

        fracunc = stats.get_fractional_uncertainty(sig_unfold_bw, unfold_cov)
        fracunc_unfold_norm = stats.get_fractional_uncertainty(sig_unfold_bw, unfold_cov_norm)
        fracunc_unfold_shape = stats.get_fractional_uncertainty(sig_unfold_bw, unfold_cov_shape)
        if stat_cov is not None:
            #First, rotate the stat cov into regularized space
            stat_cov_rotated = stat_cov @ unfold_dict['CovRotation'] @ unfold_dict['CovRotation'].T
            fracunc_stat = stats.get_fractional_uncertainty(sig_unfold_bw, stat_cov_rotated)
        else:
            fracunc_stat = None
        if verbose:
            print(f'sig_unfold_bw: {sig_unfold_bw}')
            print(f'sig_smear_bw: {sig_smear_bw}')
            print(f'unfold cov diag: {np.diag(unfold_cov)}')
        #np.diag(unfold_cov)*np.eye(unfold_cov.shape[0]) - for diag only
        chi2, dof, pval = stats.calc_chi2(sig_smear_bw, sig_unfold_bw, unfold_cov, pinv_rcond=PINV_RCOND, diagnose=verbose, retained_variance_tol=RETAINED_VARIANCE_TOL, mask_bins=mask_bins)
        chi2_1d, dof_1d, pval_1d = stats.calc_chi2(
            sig_smear_bw, sig_unfold_bw, np.diag(unfold_cov),
            diagnose=verbose,
            mask_bins=mask_bins,
        )
        #print(f'Shape of np.diag(cov): {np.diag(unfold_cov).shape}')
        #print(f'Shape of cov: {unfold_cov.shape}')
        chi2_dof_pval_str = utils.get_chi2_dof_pval_str(chi2, dof, pval)
        chi2_1d_str = utils.get_chi2_dof_pval_str(chi2_1d, dof_1d, pval_1d, diag=True)
        residuals = sig_unfold_bw - sig_smear_bw
        residuals_norm = residuals / np.sqrt(np.diag(unfold_cov))
        if verbose:
            self._print_mode_chi2_diagnostics(
                diff=residuals,
                cov=unfold_cov,
                chi2=chi2,
                header='Mode-space chi2 diagnostics for unfold result',
            )

        # Optional per-test sigma_tilde if a background is provided
        sigma_tilde = None
        if sel_background is not None:
            sigma_tilde = stats.compute_sigma_tilde(
                response,
                sig_truth,
                sel_background,
                self.xsec_unit,
            )

        return {
            'C_type': C_type,
            'Norm_type': Norm_type,
            'fractional_cov': fractional_cov,
            'input_cov': input_cov,
            'input_cov_norm': input_cov_norm,
            'input_cov_shape': input_cov_shape,
            'unfold_cov': unfold_cov,
            'unfold_fraccov': unfold_fraccov,
            'unfold_corr': unfold_corr,
            'unfold_cov_norm': unfold_cov_norm,
            'unfold_cov_shape': unfold_cov_shape,
            'fracunc_unfold_norm': fracunc_unfold_norm,
            'fracunc_unfold_shape': fracunc_unfold_shape,
            'fracunc_unfold_stat': fracunc_stat,
            'AddSmear': unfold_dict['AddSmear'],
            'unfold': unfold_dict['unfold'],
            'response': response,
            'sig_truth': sig_truth,
            'sel_background': sel_background,
            'sigma_tilde': sigma_tilde,
            'sig_smear': sig_smear,
            'sig_unfold': sig_unfold,
            'fracunc': fracunc,
            'chi2': chi2,
            'dof': dof,
            'pval': pval,
            'chi2_1d': chi2_1d,
            'pval_1d': pval_1d,
            'chi2_1d_str': chi2_1d_str,
            'chi2_dof_pval_str': chi2_dof_pval_str,
            'sig_smear_bw': sig_smear_bw,
            'sig_unfold_bw': sig_unfold_bw,
            'residuals': residuals,
            'residuals_norm': residuals_norm,
            'tikhonov_s_opt': s_opt,
        }

    def unfold(
        self,
        cov,
        sel,
        sig_truth,
        sel_background=None,
        sigma_tilde=None,
        name=None,
        response=None,
        sel_truth=None,
        sel_sig_reco=None,
        sel_sig_truth=None,
        genweights_sel_sig=None,
        bins=None,
        mask_bins=None,
        **kwargs,
    ):
        """
        Unfold the xsec using Wiener-SVD and store unfolded results in a named entry.

        Parameters
        ----------
        cov: array, covariance matrix (or fractional covariance if fractional_cov is True)
        sel: array, selection distribution binned by reco variable
        sig_truth: array, signal truth spectrum for this test
        sel_background: array, background spectrum used to compute sigma_tilde
        sigma_tilde: array, precomputed sigma_tilde (used only if sel_background is None)
        name: str, label for this unfold configuration (defaults to self.name)
        response: 2D array or None
            If provided, use this response matrix directly.
        sel_truth, sel_sig_reco, sel_sig_truth, genweights_sel_sig:
            If response is None, these must all be provided to construct the
            response matrix via stats.get_response_matrix for this unfold.
        mask_bins: array_like, optional
            Indices of bins to mask out from the chi2 calculations.
        **kwargs: additional keyword arguments for _compute_wiener
        """
        #Run checks
        if cov.shape[0] != cov.shape[1]:
            raise ValueError(f'covariance matrix ({cov.shape[0]}x{cov.shape[1]}) is not square')
        if cov.shape[0] != sel.shape[0]:
            raise ValueError(f'covariance matrix ({cov.shape[0]}x{cov.shape[1]}) and selection distribution ({sel.shape}) have different number of columns')
        if cov.shape[0] != sig_truth.shape[0]:
            raise ValueError(f'covariance matrix ({cov.shape[0]}x{cov.shape[1]}) and signal truth distribution ({sig_truth.shape}) have different number of rows')
        if sel_background is not None:
            if cov.shape[0] != sel_background.shape[0]:
                raise ValueError(f'covariance matrix ({cov.shape[0]}x{cov.shape[1]}) and background distribution ({sel_background.shape}) have different number of rows')
        if sel_truth is not None:
            if cov.shape[0] != sel_truth.shape[0]:
                raise ValueError(f'covariance matrix ({cov.shape[0]}x{cov.shape[1]}) and truth selection distribution ({sel_truth.shape}) have different number of rows')
        if name is None:
            name = self.name
        record = self._init_unfold_record(name)
        # Determine or build the response matrix for this configuration.
        if response is None:
            builder_args = [sel_truth, sel_sig_reco, sel_sig_truth, genweights_sel_sig]
            if any(arg is None for arg in builder_args):
                raise ValueError(
                    'Either response must be provided directly '
                    'or sel_truth, sel_sig_reco, sel_sig_truth, genweights_sel_sig '
                    'must all be provided to construct it'
                )
            if bins is None:
                bins = self.bins
            #Check that all values fall within the bins
            for i,arr in enumerate([sel_sig_reco, sel_sig_truth]):
                if np.any(arr < bins[0]) or np.any(arr > bins[-1]):
                    print('Order of arrays is: sel_sig_reco, sel_sig_truth')
                    raise ValueError(f'Value {arr[np.any(arr < bins[0]) or np.any(arr > bins[-1])]} is out of bins {bins} for {i}^th array')
            response = stats.get_response_matrix(
                sig_truth=sig_truth,
                sel_truth=sel_truth,
                sel_sig_reco=sel_sig_reco,
                sel_sig_truth=sel_sig_truth,
                genweights_sel_sig=genweights_sel_sig,
                bins=bins,
            )
        core = self._compute_wiener(
            cov=cov,
            sel=sel,
            sig_truth=sig_truth,
            response=response,
            mask_bins=mask_bins,
            sel_background=sel_background,
            sigma_tilde=sigma_tilde,
            **kwargs,
        )
        # Update all fields relevant to a full unfolding
        record.update(core)

    def _prepare_plot(self, names, model_labels, verbose=False, use_diag_cov=False):
        """
        Prepare the data for plotting the unfold results. 
        In common with 2d and 1d plots.
        """
        assert len(names) == len(model_labels), f'Number of names ({len(names)}) and model labels ({len(model_labels)}) must match'
        for n in names:
            if n not in self.unfold_results:
                raise ValueError(f'Unfold result {n} not found, run unfold() first')
        str_key = 'chi2_1d_str' if use_diag_cov else 'chi2_dof_pval_str'
        chi2_strs = [self.unfold_results[names[0]][str_key]]
        for n in names[1:]:
            pred = self.unfold_results[n]['sig_smear_bw']
            true = self.unfold_results[names[0]]['sig_unfold_bw']
            cov = self.unfold_results[names[0]]['unfold_cov']
            if use_diag_cov:
                cov = np.diag(cov)
            chi2, dof, pval = stats.calc_chi2(
                pred, true, cov,
                pinv_rcond=PINV_RCOND,
                retained_variance_tol=RETAINED_VARIANCE_TOL,
                diagnose=verbose,
            )
            if verbose:
                self._print_mode_chi2_diagnostics(
                    diff=true - pred,
                    cov=cov,
                    chi2=chi2,
                    header=f'Mode-space chi2 diagnostics for _prepare_plot: {n} vs {names[0]}',
                )
            chi2_strs.append(utils.get_chi2_dof_pval_str(chi2, dof, pval, diag=use_diag_cov))
        return chi2_strs
    def plot_unfold(self, names, model_labels, title='', label='', pot_label='', data_label='Data', 
        legend_fontsize=10, verbose=False, **kwargs):
        """
        Plot the unfold results for the given names.
        They have to have been unfolded first.
        Parameters
        ----------
        names: list of str, names of the unfold results to plot
        model_labels: list of str, labels for the model
        **kwargs: additional keyword arguments for makeplot.plot_hist_edges
        """
        chi2_dof_pval_strs = self._prepare_plot(names, model_labels, verbose=verbose, use_diag_cov=False)
        #Plot the results
        fig,ax = plt.subplots()
        for i,name in enumerate(names):
            model_label = model_labels[i]
            # Plot unfolded signal
            makeplot.plot_hist_edges(self.bins,
                self.unfold_results[name]['sig_smear']*self.yscale,
                None,
                ax=ax,
                label=f"{model_label}" + "\n" + f"({chi2_dof_pval_strs[i]})", **kwargs
            )
        # Assume first is the source of uncertainty and data
        # Data - shape only syst
        ax.errorbar(self.bin_centers,
            self.unfold_results[names[0]]['sig_unfold']*self.yscale,
            yerr=self.unfold_results[names[0]]['sig_unfold']*self.unfold_results[names[0]]['fracunc_unfold_shape']*self.yscale,
            label=f'{data_label} (Shape Syst. + Stat Unc.)',c='k',fmt='o',capsize=5,markersize=5)
        # Data - stat only
        ax.errorbar(self.bin_centers,
            self.unfold_results[names[0]]['sig_unfold']*self.yscale,
            yerr=self.unfold_results[names[0]]['sig_unfold']*self.unfold_results[names[0]]['fracunc_unfold_stat']*self.yscale,
            c='k',fmt='o',capsize=3,markersize=3)
        # Data - norm unc
        ax.stairs(self.unfold_results[names[0]]['sig_unfold']*self.yscale*self.unfold_results[names[0]]['fracunc_unfold_norm'],
            self.bins, label='Norm. Syst. Unc.', fill=True, color='gray',
            alpha=0.5
        )
        ax.legend(fontsize=legend_fontsize)
        ax.set_xlabel(self.regularized_label)
        ax.set_ylabel(self.xsec_label)
        title_fontsize = 18 if len(title) > 40 else 24
        ax.set_title(title,fontsize=title_fontsize)
        plotters.add_label(ax,label,where='topleftoutside',color='gray')
        plotters.add_label(ax,pot_label,where='toprightoutside',color='gray')
        return fig,ax

    def plot_unfold_2d(self, names, model_labels, binning2D, title='', label='', pot_label='', data_label='Data', verbose=False, **kwargs):
        """
        Plot the unfold results for the given names in 2d.
        """
        chi2_dof_pval_strs = self._prepare_plot(names, model_labels, verbose=verbose, use_diag_cov=False)
        #Plot the results
        fig,axs = None,None
        for i,name in enumerate(names):
            model_label = model_labels[i]
            # Plot smeared signal
            fig,axs = binning2D.plot_differential_hist_binned(self.unfold_results[name]['sig_smear']*self.yscale,
                label=f"{model_label}" + "\n" + f"({chi2_dof_pval_strs[i]})",
                fig=fig,axs=axs)
        # Data - shape only syst
        fig,axs = binning2D.plot_differential_scatter_binned(self.unfold_results[names[0]]['sig_unfold']*self.yscale,
            yerrs=self.unfold_results[names[0]]['sig_unfold']*self.unfold_results[names[0]]['fracunc_unfold_shape']*self.yscale,
            fig=fig,axs=axs,
        label=f'{data_label} (Shape Syst. + Stat Unc.)',color='k',capsize=5,markersize=5)
        # Data - stat only
        fig,axs = binning2D.plot_differential_scatter_binned(self.unfold_results[names[0]]['sig_unfold']*self.yscale,
            yerrs=self.unfold_results[names[0]]['sig_unfold']*self.unfold_results[names[0]]['fracunc_unfold_stat']*self.yscale,
            fig=fig,axs=axs,
            color='k',capsize=3,markersize=3)
        # Data - norm unc
        fig,axs = binning2D.plot_differential_stairs_binned(self.unfold_results[names[0]]['sig_unfold']*self.yscale*self.unfold_results[names[0]]['fracunc_unfold_norm'],
            label='Norm. Syst. Unc.',fill=True,color='gray',alpha=0.5,
            fig=fig,axs=axs,add_labels=True,legend=True,xlabel=r'Regularized $p_\mu$ [GeV]',
        ylabel=self.xsec_label)

        title_fontsize = 18 if len(title) > 50 else 24
        axs[0,1].set_title(title,fontsize=title_fontsize)

        plotters.add_label(axs[0,2],pot_label,where='toprightoutsidepad',color='gray')
        plotters.add_label(axs[0,0],label,where='topleftoutsidepad',color='gray')
        for ax in axs.flatten():
            ax.set_ylim(0,None)
        return fig,axs

    def plot_residuals_2d(self, names, model_labels, binning2D, title='', label='', pot_label='',
                          legend=True, normalized=True, verbose=False, **kwargs):
        """
        Plot per-bin residuals in 2d binning panels.

        Parameters
        ----------
        names: list of str
            Names of unfold results to draw.
        model_labels: list of str
            Labels for each result.
        binning2D:
            Differential binning helper object used by plot_unfold_2d.
        normalized: bool
            If True, plot (sig_unfold_bw - sig_smear_bw)/sqrt(diag(unfold_cov)).
            If False, plot raw residuals with y-errors sqrt(diag(unfold_cov)).
        """
        chi2_1d_strs = self._prepare_plot(names, model_labels, verbose=verbose, use_diag_cov=True)
        ref = self.unfold_results[names[0]]
        ref_unfold_bw = ref['sig_unfold_bw']
        ref_sigma = np.sqrt(np.diag(ref['unfold_cov']))
        fig, axs = None, None
        for i, name in enumerate(names):
            rec = self.unfold_results[name]
            residual_bw = ref_unfold_bw - rec['sig_smear_bw']
            y = residual_bw / ref_sigma if normalized else residual_bw * self.yscale
            yerrs = None if normalized else ref_sigma * self.yscale
            fig, axs = binning2D.plot_differential_scatter_binned(
                y,
                yerrs=yerrs,
                fig=fig,
                axs=axs,
                label=f"{model_labels[i]}\n({chi2_1d_strs[i]})",
                **kwargs,
            )
        # Draw a zero-residual reference line in each panel.
        for ax in axs.flatten():
            ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
            if normalized:
                ax.axhspan(-1, 1, color='green', alpha=0.15)
                ax.axhspan(-2, 2, color='yellow', alpha=0.15)
        ylabel = r'$(d^2\sigma_\mathrm{data} - d^2\sigma_\mathrm{model})\,/\,\sqrt{\Sigma_{ii}}$' if normalized \
                 else f'Residual [{self.yscale_str}{self.unit}]'
        # Reuse stairs helper to place consistent labels/legend across panels.
        fig, axs = binning2D.plot_differential_stairs_binned(
            np.zeros_like(self.unfold_results[names[0]]['sig_unfold'] * self.yscale),
            label='',
            fill=False,
            alpha=0,
            fig=fig,
            axs=axs,
            add_labels=True,
            legend=legend,
            xlabel=r'Regularized $p_\mu$ [GeV]',
            ylabel=ylabel,
        )
        title_fontsize = 18 if len(title) > 50 else 24
        axs[0,1].set_title(title, fontsize=title_fontsize)
        plotters.add_label(axs[0,2], pot_label, where='toprightoutsidepad', color='gray')
        plotters.add_label(axs[0,0], label, where='topleftoutsidepad', color='gray')
        return fig, axs

    def plot_smear(self, name, title='', label='', pot_label='', 
        vmin=-1,vmax=1,cmap='bwr',**kwargs):
        """
        Plot the smearing matrix for the given name.
        """
        assert name in self.unfold_results, f'Smearing matrix for {name} not found, run unfold() first'
        fig,ax = plt.subplots()
        im = ax.pcolormesh(self.bins,self.bins,self.unfold_results[name]['AddSmear'],
            vmin=vmin,vmax=vmax,cmap=cmap)
        ax.set_xlabel(self.true_label)
        ax.set_ylabel(self.regularized_label)
        title_fontsize = 16 if len(title) > 40 else 24
        ax.set_title(title,fontsize=title_fontsize)
        fig.colorbar(im,ax=ax)
        plotters.add_label(ax,label,where='topleftoutside',color='gray')
        plotters.add_label(ax,pot_label,where='toprightoutside',color='gray')
        return fig,ax
    
    def plot_residuals(self, names, model_labels, title='', label='', pot_label='',
                       legend_fontsize=10, normalized=True, verbose=False, **kwargs):
        """
        Plot per-bin residuals (sig_unfold_bw - sig_smear_bw) for each named result,
        optionally normalized by sqrt(diag(unfold_cov)).

        Parameters
        ----------
        names: list of str
        model_labels: list of str
        normalized: bool
            If True (default) plot residuals / sqrt(diag(unfold_cov)) — i.e. pull per bin.
            If False plot raw residuals in xsec units.
        """
        chi2_1d_strs = self._prepare_plot(names, model_labels, verbose=verbose, use_diag_cov=True)
        ref = self.unfold_results[names[0]]
        ref_unfold_bw = ref['sig_unfold_bw']
        ref_sigma = np.sqrt(np.diag(ref['unfold_cov']))
        fig, ax = plt.subplots()
        for i, (name, model_label) in enumerate(zip(names, model_labels)):
            rec = self.unfold_results[name]
            residual_bw = ref_unfold_bw - rec['sig_smear_bw']
            y = residual_bw / ref_sigma if normalized else residual_bw * self.yscale
            yerr = None if normalized else ref_sigma * self.yscale
            ax.errorbar(self.bin_centers, y, xerr=self.bin_widths / 2, yerr=yerr,
                        fmt='o', capsize=4, markersize=5,
                        label=f"{model_label}  ({chi2_1d_strs[i]})", **kwargs)
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax.axhspan(-1, 1, color='green', alpha=0.15, label=r'$1\sigma$')
        ax.axhspan(-2, 2, color='yellow', alpha=0.15, label=r'$2\sigma$')
        ylabel = r'$(d\sigma_\mathrm{data} - d\sigma_\mathrm{model})\,/\,\sqrt{\Sigma_{ii}}$' if normalized \
                 else f'Residual [{self.yscale_str}{self.unit}]'
        ax.set_xlabel(self.regularized_label)
        ax.set_ylabel(ylabel)
        title_fontsize = 18 if len(title) > 40 else 24
        ax.set_title(title, fontsize=title_fontsize)
        ax.legend(fontsize=legend_fontsize)
        plotters.add_label(ax, label, where='topleftoutside', color='gray')
        plotters.add_label(ax, pot_label, where='toprightoutside', color='gray')
        return fig, ax

    def plot_response(self, name, title='', label='', pot_label='', vmin=None, vmax=None, cmap='RdPu', **kwargs):
        """
        Plot the response matrix for the given name.
        """
        assert name in self.unfold_results, f'Response matrix for {name} not found, run unfold() first'
        fig,ax = plt.subplots()
        im = ax.pcolormesh(self.bins,self.bins,self.unfold_results[name]['response'],
            vmin=vmin,vmax=vmax,cmap=cmap)
        ax.set_xlabel(self.true_label)
        ax.set_ylabel(self.reco_label)
        title_fontsize = 16 if len(title) > 40 else 24
        ax.set_title(title,fontsize=title_fontsize)
        fig.colorbar(im,ax=ax)
        plotters.add_label(ax,label,where='topleftoutside',color='gray')
        plotters.add_label(ax,pot_label,where='toprightoutside',color='gray')
        return fig,ax

    def save(self, save_dir, metadata_dir='metadata_xsec'):
        """
        Save the XSec metadata and any stored unfold results.

        Parameters
        ----------
        save_dir: str, directory to save into
        metadata_dir: str, subdirectory for metadata and unfold results
        """
        os.makedirs(save_dir, exist_ok=True)
        meta_dir = os.path.join(save_dir, metadata_dir)
        os.makedirs(meta_dir, exist_ok=True)
        # Core metadata arrays (truth and backgrounds are now per-unfold and live in unfold_results)
        np.savetxt(os.path.join(meta_dir, f'{self.variable}_bins.csv'), self.bins)
        np.savetxt(os.path.join(meta_dir, f'{self.variable}_bin_widths.csv'), self.bin_widths)
        if getattr(self, 'sigma_tilde', None) is not None:
            np.savetxt(os.path.join(meta_dir, f'{self.variable}_sigma_tilde.csv'), self.sigma_tilde)
        # Scalars and strings
        np.savetxt(os.path.join(meta_dir, f'{self.variable}_xsec_unit.csv'), np.array([self.xsec_unit]))
        meta = {'name': self.name, 'variable': self.variable}
        with open(os.path.join(meta_dir, f'{self.variable}_metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        # Unfold results: save each named configuration in its own subfolder, with
        # filenames prefixed by the variable name (to mirror Systematics.save layout).
        if self.unfold_results:
            for key, record in self.unfold_results.items():
                subfolder = os.path.join(save_dir, key)
                os.makedirs(subfolder, exist_ok=True)
                filename_prefix = f'{self.variable}_'
                for k, val in record.items():
                    if val is None:
                        continue
                    # Skip empty lists / arrays
                    if isinstance(val, (list, np.ndarray)) and len(val) == 0:
                        continue
                    path_base = os.path.join(subfolder, f'{filename_prefix}{k}')
                    # Arrays or lists -> CSV
                    if isinstance(val, (np.ndarray, list)):
                        arr = np.array(val)
                        # 0D arrays become scalars, handle below
                        if arr.ndim == 0:
                            with open(path_base + '.txt', 'w') as f:
                                f.write(str(arr.item()))
                        else:
                            np.savetxt(path_base + '.csv', arr)
                    # Strings
                    elif isinstance(val, str):
                        with open(path_base + '.txt', 'w') as f:
                            f.write(val)
                    # Scalars (int, float, numpy scalar)
                    elif isinstance(val, (int, float, np.number)):
                        with open(path_base + '.txt', 'w') as f:
                            f.write(str(val))
                    else:
                        # Best-effort string representation for anything odd
                        with open(path_base + '.txt', 'w') as f:
                            f.write(str(val))

    def to_nuisance(self, save_dir, names=None, filename_suffix='xsec', scale=1.):
        """
        Write covariance matrix, unfolded spectrum, and smearing matrix to a ROOT file
        in uboone data-release style: covariance_matrix (TH2D), smearing_matrix (TH2D),
        xsec_data (TH1D).

        Parameters
        ----------
        save_dir : str or path-like
            Output directory.
        names : list of str, optional
            Keys to use for the saving. If None, uses all keys.
        filename_suffix : str, optional
            Suffix for the filename.
        scale : float, optional
            Scale factor for the unfolded spectrum, covariance matrix, and smearing matrix.
        """
        for name in list(self.unfold_results.keys()):
            if names is not None and name not in names:
                continue
            save_dir = os.path.join(save_dir,name)
            filename = f'{self.variable}_{filename_suffix}.root'
            record = self.unfold_results[name]
            unfold = record.get('unfold')
            unfold_cov = record.get('unfold_cov')
            add_smear = record.get('AddSmear')
            if unfold is None or unfold_cov is None or add_smear is None:
                raise RuntimeError(
                    f'unfold config "{name}" missing unfold, unfold_cov, or AddSmear; '
                    f'run unfold("{name}") for this config first'
                )
            n = len(unfold)
            bins = np.asarray(self.bins, dtype=np.float64)
            if len(bins) != n + 1:
                raise RuntimeError(
                    f'bins length {len(bins)} != n+1 ({n}+1); check XSec bins vs unfold size'
                )
            axis = bh.axis.Variable(bins)
            h_xsec = bh.Histogram(axis, storage=bh.storage.Double())
            h_xsec.view(flow=True)[1:-1] = np.asarray(unfold/self.bin_widths, dtype=np.float64) * scale
            h_cov = bh.Histogram(axis, axis, storage=bh.storage.Double())
            h_cov.view(flow=True)[1:-1, 1:-1] = np.asarray(unfold_cov, dtype=np.float64) * scale
            h_smear = bh.Histogram(axis, axis, storage=bh.storage.Double())
            h_smear.view(flow=True)[1:-1, 1:-1] = np.asarray(add_smear, dtype=np.float64) * scale
            with uproot.recreate(os.path.join(save_dir,filename)) as f:
                # TODO: Replace this with uBooNE data-release style
                # f['covariance_matrix'] = h_cov
                # f['smearing_matrix'] = h_smear
                # f['xsec_data'] = h_xsec
                # uBooNE data-release style
                f['covariance'] = h_cov
                f['smearing_Ac'] = h_smear
                f['data_xsec'] = h_xsec

    @classmethod
    def from_saved(cls, load_dir, variable, metadata_dir='metadata_xsec'):
        """
        Reconstruct an XSec object and any stored unfold results from disk.

        Parameters
        ----------
        load_dir: str, directory to load from
        variable: str
            Variable name / type to load. This selects which variable-specific
            metadata files (by prefix) and unfold-result files to read.
        metadata_dir: str, subdirectory for metadata and unfold results
        """
        meta_dir = os.path.join(load_dir, metadata_dir)
        if not os.path.exists(meta_dir):
            raise ValueError(f'Metadata directory not found in {load_dir}')
        bins = np.loadtxt(os.path.join(meta_dir, f'{variable}_bins.csv'))
        bin_widths = np.loadtxt(os.path.join(meta_dir, f'{variable}_bin_widths.csv'))
        sigma_tilde_path = os.path.join(meta_dir, f'{variable}_sigma_tilde.csv')
        sigma_tilde = np.loadtxt(sigma_tilde_path) if os.path.exists(sigma_tilde_path) else None
        xsec_unit_arr = np.loadtxt(os.path.join(meta_dir, f'{variable}_xsec_unit.csv'))
        if np.ndim(xsec_unit_arr) == 0:
            xsec_unit = float(xsec_unit_arr)
        else:
            xsec_unit = float(xsec_unit_arr[0])
        meta_path = os.path.join(meta_dir, f'{variable}_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            name = meta.get('name', 'xsec')
        else:
            name = 'xsec'
        instance = cls(
            xsec_unit=xsec_unit,
            bins=bins,
            bin_widths=bin_widths,
            name=name,
            variable=variable,
        )
        if sigma_tilde is not None:
            instance.sigma_tilde = sigma_tilde
        # Load unfold results: each subfolder (except metadata_dir_name) corresponds to a key.
        # Only files whose names start with the chosen variable prefix are loaded,
        # so different variable types can safely coexist in the same directory.
        instance.unfold_results = {}
        filename_prefix = f'{instance.variable}_'
        for entry in os.listdir(load_dir):
            subfolder_path = os.path.join(load_dir, entry)
            if not os.path.isdir(subfolder_path):
                continue
            # Skip metadata directory for this variable
            if entry == metadata_dir:
                continue
            key = entry
            record = instance._init_unfold_record(key)
            for fname in os.listdir(subfolder_path):
                fpath = os.path.join(subfolder_path, fname)
                if not os.path.isfile(fpath):
                    continue
                # Only consider files that start with the variable prefix
                if not fname.startswith(filename_prefix):
                    continue
                # Strip prefix and extension to get the record field name
                base = fname.replace('.csv', '').replace('.txt', '')
                if not base.startswith(filename_prefix):
                    continue
                field_name = base[len(filename_prefix):]
                # CSV -> numpy array
                if fname.endswith('.csv'):
                    arr = np.loadtxt(fpath)
                    record[field_name] = arr
                # TXT -> scalar or string
                elif fname.endswith('.txt'):
                    with open(fpath, 'r') as f:
                        txt = f.read().strip()
                    # Try to parse as int / float; fall back to string
                    try:
                        num = float(txt)
                        if num.is_integer():
                            record[field_name] = int(num)
                        else:
                            record[field_name] = num
                    except ValueError:
                        record[field_name] = txt
            instance.unfold_results[key] = record
        return instance

    def load(self, load_dir, variable, metadata_dir='metadata_xsec'):
        """
        Load unfold results into an already constructed XSec object.

        Parameters
        ----------
        load_dir: str, directory to load from
        variable: str
            Variable name / type to load. This selects which variable-specific
            unfold-result files to read (by filename prefix).
        metadata_dir: str, subdirectory where shared metadata lives
        """
        meta_dir = os.path.join(load_dir, metadata_dir)
        if not os.path.exists(meta_dir):
            return

        filename_prefix = f'{variable}_'

        for entry in os.listdir(load_dir):
            subfolder_path = os.path.join(load_dir, entry)
            if not os.path.isdir(subfolder_path):
                continue
            if entry == metadata_dir:
                continue
            key = entry
            record = self._init_unfold_record(key)
            for fname in os.listdir(subfolder_path):
                fpath = os.path.join(subfolder_path, fname)
                if not os.path.isfile(fpath):
                    continue
                if not fname.startswith(filename_prefix):
                    continue
                base = fname.replace('.csv', '').replace('.txt', '')
                if not base.startswith(filename_prefix):
                    continue
                field_name = base[len(filename_prefix):]
                if fname.endswith('.csv'):
                    arr = np.loadtxt(fpath)
                    record[field_name] = arr
                elif fname.endswith('.txt'):
                    with open(fpath, 'r') as f:
                        txt = f.read().strip()
                    try:
                        num = float(txt)
                        if num.is_integer():
                            record[field_name] = int(num)
                        else:
                            record[field_name] = num
                    except ValueError:
                        record[field_name] = txt
            self.unfold_results[key] = record

    @staticmethod
    def _to_serializable(obj):
        """
        Recursively convert numpy arrays in an object to lists for JSON storage.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: XSec._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [XSec._to_serializable(v) for v in obj]
        return obj

    @staticmethod
    def _from_serializable(obj):
        """
        Recursively convert JSON loaded structures back to numpy arrays
        where appropriate (lists of numbers or nested lists).
        """
        if isinstance(obj, dict):
            return {k: XSec._from_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            # Heuristic: convert lists of numbers or nested lists to numpy arrays
            if not obj:
                return np.array([])
            if all(isinstance(v, (int, float)) for v in obj):
                return np.array(obj)
            if all(isinstance(v, list) for v in obj):
                return np.array([XSec._from_serializable(v) for v in obj])
            return [XSec._from_serializable(v) for v in obj]
        return obj