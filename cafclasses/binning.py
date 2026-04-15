import numpy as np
import matplotlib.pyplot as plt
from sbnd.plotlibrary import makeplot
from sbnd.general import plotters
from sbnd.numu.numu_constants import *


class Binning2D:
    """Class for handling 2D differential binning in costheta and momentum."""
    
    def __init__(self, diff_costheta_bins=None, diff_momentum_bins_2d=None, keep_null=True):
        """
        Initialize 2D binning configuration.
        
        Parameters
        ----------
        diff_costheta_bins : array-like, optional
            Bin edges for costheta. Default: from numu_constants.py
        diff_momentum_bins_2d : array-like, optional
            2D momentum bin edges of shape (N_costheta_bins, N_momentum_bins+1),
            allowing costheta dependent momentum binning.
        """
        # Default values
        if diff_costheta_bins is None:
            diff_costheta_bins = DIFF_COSTHETA_BINS
        self.diff_costheta_bins = np.array(diff_costheta_bins)

        n_costheta_bins = len(self.diff_costheta_bins) - 1

        if diff_momentum_bins_2d is not None:
            # Fully 2D momentum binning provided
            if diff_momentum_bins_2d.shape[0] != n_costheta_bins:
                raise ValueError(
                    f'diff_momentum_bins_2d has {diff_momentum_bins_2d.shape[0]} costheta rows, '
                    f'expected {n_costheta_bins}'
                )
            self.diff_momentum_bins_2d = diff_momentum_bins_2d
        else:
            self.diff_momentum_bins_2d = np.array(DIFF_MOMENTUM_BINS_2D)

        self.n_costheta_bins = n_costheta_bins
        self.n_momentum_bins = self.diff_momentum_bins_2d.shape[1] - 1
        
        # Costheta bins and labels
        self.diff_costheta_bin_labels = [f'{self.diff_costheta_bins[i]:.2f} - {self.diff_costheta_bins[i+1]:.2f}' 
                                         for i in range(len(self.diff_costheta_bins)-1)]
        self.diff_costheta_centers = (self.diff_costheta_bins[:-1] + self.diff_costheta_bins[1:])/2.
        
        # Momentum bins and labels
        self.diff_momentum_bin_centers_2d = []
        for i,mbins in enumerate(self.diff_momentum_bins_2d):
            self.diff_momentum_bin_centers_2d.append((mbins[:-1] + mbins[1:])/2.)
        self.diff_momentum_bin_centers_2d = np.array(self.diff_momentum_bin_centers_2d)

        # Differential bins
        self.differential_bins = (len(self.diff_costheta_bins)-2) + (self.n_momentum_bins-1)*(len(self.diff_costheta_bins)-1)
        self.differential_edges = np.arange(-1.5, np.max(self.differential_bins)+1.5, 1)
        self.differential_centers = (self.differential_edges[:-1] + self.differential_edges[1:])/2.
        
        # Template for differential dictionaries
        diff_dict_template = {
            'costheta_bin': -1,
            'momentum_bin': -1,
            'costheta_edges': [-np.inf, np.inf],
            'momentum_edges': [-np.inf, np.inf],
            'momentum_center': -np.inf,
            'costheta_center': -np.inf,
            'bin_width': -np.inf
        }
        
        # Initialize differential dictionaries
        self.differential_dicts = {c: diff_dict_template.copy() for c in sorted(self.differential_centers[1:])}
        self._initialize_differential_dicts()

        #Store bin widths - roll once to shift the null bin to the front
        self.bin_widths = np.roll(np.array([ddict['bin_width'] for ddict in self.differential_dicts.values()]),1)

        if not keep_null:
            self.differential_dicts = {k: v for k, v in self.differential_dicts.items() if k != -1}
            self.bin_widths = self.bin_widths[1:]
            self.differential_centers = self.differential_centers[1:]
            self.differential_edges = self.differential_edges[1:]

    def _initialize_differential_dicts(self):
        """Initialize the differential dictionaries with bin information."""
        for c in self.differential_dicts:
            c_int = int(c)
            cbin = np.mod(c_int, self.n_costheta_bins)
            pbin = c_int//(self.n_costheta_bins)
            self.differential_dicts[c]['costheta_bin'] = cbin
            self.differential_dicts[c]['momentum_bin'] = pbin
            self.differential_dicts[c]['costheta_edges'] = self.diff_costheta_bins[cbin:cbin+2]
            self.differential_dicts[c]['momentum_edges'] = self.diff_momentum_bins_2d[cbin, pbin:pbin+2]
            self.differential_dicts[c]['momentum_center'] = 0.5 * np.sum(self.diff_momentum_bins_2d[cbin, pbin:pbin+2])
            self.differential_dicts[c]['costheta_center'] = self.diff_costheta_centers[cbin]
            self.differential_dicts[c]['bin_width'] = np.diff(self.diff_costheta_bins[cbin:cbin+2])[0]*np.diff(self.diff_momentum_bins_2d[cbin, pbin:pbin+2])[0]
        #Add negative bin for null binning
        self.differential_dicts[-1] = {
            'costheta_bin': -1,
            'momentum_bin': -1,
            'costheta_edges': [-np.inf, np.inf],
            'momentum_edges': [-np.inf, np.inf],
            'momentum_center': -np.inf,
            'costheta_center': -np.inf,
            'bin_width': -np.inf}
    
    def init_hist_dict(self,bin_by='costheta',diff_dicts=None,include_null=False):
        """Initialize the histogram dictionary."""
        if diff_dicts is None:
            diff_dicts = self.differential_dicts
        # Initialize the histogram dictionary by the unique values of bin_by key
        keys = np.unique([v[f'{bin_by}_bin'] for v in diff_dicts.values()])
        if not include_null:
            m = keys == -1
            keys = keys[~m]
        hist_dict = {k: ([None,None]) for k in keys}
        bin_by_edges = np.unique([np.array(v[f'{bin_by}_edges']) for v in diff_dicts.values()], axis=0)
        if not include_null:
            bin_by_edges = bin_by_edges[~m]
        return hist_dict, bin_by_edges

    def bin_differential_dict(self, series2, bins1, bins2, diff_dicts=None, weights=None, bin_by='costheta',include_null=False):
        """
        Bin series2 into bins of bins1. Create histograms for each bin of series2.

        Parameters
        ----------
        series2 : array-like (N or B)
            The series to bin into the bins of series1.
        bins1 : array-like (N or B)
            The bins that series2 is being binned into. Should map 1:1 with series2.
            If length is B, then we assume series2 is arranged by the differential bins.
        bins2 : array-like (B)
            The bins to bin series2 into.
        diff_dicts : dict, optional
            The differential dictionaries to use for the histograms. Contains the edges and labels for each differential bin.
            If None, uses self.differential_dicts.
        weights : array-like (N), optional
            The weights to use for the histograms.
        bin_by : str
            The series to bin by.

        Returns
        -------
        hist_dict : dict
            Dictionary of histograms for each bin of series1. Each histogram is a tuple of (counts, bin_edges).
            The keys are the unique values of bin_by.
        bin_by_edges : array-like
            The bin edges for the bin_by series.
        """
        if diff_dicts is None:
            diff_dicts = self.differential_dicts
        if weights is None:
            weights = np.ones(len(series2))
        assert len(series2) == len(bins1) == len(weights), f'Series2 and bins1 and weights must be the same length: {len(series2)} != {len(bins1)} != {len(weights)}'
        hist_dict, bin_by_edges = self.init_hist_dict(include_null=include_null)

        # Loop over the unique values of bin_by
        for k in hist_dict.keys():
            mask = bins1 == k
            hist_dict[k] = np.histogram(series2[mask], bins=bins2, weights=weights[mask])

        assert len(bin_by_edges) == len(hist_dict), f'bin_by_edges and hist_dict must be the same length: {len(bin_by_edges)} != {len(hist_dict)}'
        return hist_dict, bin_by_edges
    
    def bin_differential_dict_binned(self, values, diff_dicts=None, bin_by='costheta'):
        """
        Bin values into the differential bins.
        Relies on the assumption that values is binned identically to the differential bins.

        Parameters
        ----------
        values : array-like (N)
            The values to bin into the differential bins. Same length as self.differential_centers.
        diff_dicts : dict, optional
            The differential dictionaries to use for the histograms. Contains the edges and labels for each differential bin.
            If None, uses self.differential_dicts.

        Returns
        -------
        hist_dict : dict
            Dictionary of histograms for each bin of values. Each histogram is a tuple of (counts, bin_edges).
            The keys point to which axis to plot on.
        """
        #TODO: Add support for bin_by='momentum'
        if diff_dicts is None:
            diff_dicts = self.differential_dicts
        assert len(values) == len(self.differential_centers), f'values and differential_centers must be the same length: {len(values)} != {len(self.differential_centers)}'
        hist_dict, bin_by_edges = self.init_hist_dict(bin_by=bin_by, diff_dicts=diff_dicts)
        # Map diff_dict key (differential bin center) to position in values (values is ordered by differential_centers)
        keys_ordered = list(diff_dicts.keys())
        inds = np.array([np.where(self.differential_centers == k)[0][0] for k in keys_ordered])
        # This means that each histogram contains all momentum bins for the same costheta bin
        if bin_by == 'costheta':
            for i,c in enumerate(hist_dict):
                #Find all indices that point to the same costheta bin   
                mask = [v['costheta_bin'] == c for v in diff_dicts.values()]
                hist_dict[c][0] = np.array(values[inds[mask]])
                # Use costheta dependent momentum edges
                hist_dict[c][1] = self.diff_momentum_bins_2d[c]
        elif bin_by == 'momentum':
            for i,c in enumerate(hist_dict):
                #Find all indices that point to the same momentum bin   
                mask = [v['momentum_bin'] == c for v in diff_dicts.values()]
                hist_dict[c][0] = np.array(values[inds[mask]])
                hist_dict[c][1] = self.diff_costheta_bins
        return hist_dict, bin_by_edges


    def plot_differential_hist(self, series2, bins1, bins2, weights=None, yerrs=None, diff_dicts=None, bin_by='costheta',
                               xlabel='', ylabel='Candidates', label='', fig=None, axs=None, legend=False, add_labels=False, frac_unc=False, **kwargs):
        """
        Plot the differential histograms for each bin of bins1.

        Parameters
        ----------
        series2 : array-like (N or B)
            The series to bin into the bins of series1. If length is N, then series2 is binned into the bins of series1. 
            If length is B, then we assume series2 is arranged by the differential bins.
        bins1 : array-like (N or B)
            The bins that series2 is being binned into. Should map 1:1 with series2.
            If length is B, then we assume series2 is arranged by the differential bins.
        bins2 : array-like (B)
            The bins to bin series2 into.
        weights : array-like (N), optional
            The weights to use for the histograms. If None, no weights are used.
        yerrs : array-like (B), optional
            The y-errors to use for the histograms. If None, no errors are plotted.
        diff_dicts : dict, optional
            The differential dictionaries to use for the histograms. Contains the edges and labels for each differential bin.
            If None, uses self.differential_dicts.
        bin_by : str
            The series to bin by.
        xlabel : str
            The label for the x-axis.
        ylabel : str
            The label for the y-axis.
        label : str
            The label to use for the histograms.
        fig : matplotlib.figure.Figure, optional
            The figure to use for the histograms.
        axs : matplotlib.axes.Axes, optional
            The axes to use for the histograms.
        legend : bool
            Whether to add the legend to the histograms.
        add_labels : bool
            Whether to add labels to the histograms.
        frac_unc : bool
            Provided yerrs are fractional uncertainties. If True, the errors are multiplied by the counts.
        **kwargs : dict
            Keyword arguments to pass to the plotting function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the differential histograms.
        axs : matplotlib.axes.Axes
            The axes containing the differential histograms.
        """
        bin_by_text = r'$\cos\theta_{\mu}$' if bin_by == 'costheta' else r'$p_{\mu}$'
        # Get hist_dict
        hist_dict, bin_by_edges = self.bin_differential_dict(series2, bins1, bins2, diff_dicts, weights=weights, bin_by=bin_by)
        # TODO: Adjust this to be dynamic based on costheta binning
        if fig is None and axs is None:
            fig, axs = plt.subplots(figsize=(12, 8), nrows=3, ncols=3)
        assert len(axs.flatten()) == len(hist_dict), f'axs and hist_dict must be the same length: {len(axs.flatten())} != {len(hist_dict)}'
        if yerrs is not None:
            yerr_hist_dict,_ = self.bin_differential_dict_binned(yerrs, diff_dicts=diff_dicts, bin_by=bin_by)
        for i, (ax, h, b) in enumerate(zip(axs.flatten(), hist_dict.values(), bin_by_edges)):
            # Always draw the histogram values (step)
            makeplot.plot_hist_edges(h[1], h[0], errors=None, label=label, ax=ax, **kwargs)

            # Optional uncertainty band overlay
            if yerrs is not None:
                errors = yerr_hist_dict[i][0]
                if frac_unc:
                    yerr_arr = errors*h[0]
                else:
                    yerr_arr = errors
                makeplot.plot_hist_with_uncertainty(bins=h[1], n_perbin=h[0], yerr_arr=yerr_arr, yerr_label='Unc', ax=ax)
            if bin_by == 'costheta':
                ax.set_xlim(0, MAX_PMOM)
            bin_text = f'{b[0]:.2f} < {bin_by_text} < {b[1]:.2f}'
            if add_labels:
                plotters.add_label(ax, bin_text, where=(1.,0.75), horizontalalignment='right', color='black', alpha=1., fontsize=8)
        # Add the labels
        if add_labels:
            axs[2, 1].set_xlabel(xlabel)
            axs[1, 0].set_ylabel(ylabel)
        if legend:
            axs[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        return fig, axs
    
    def plot_differential_scatter(self, series2, bins1, bins2, weights=None, yerrs=None, diff_dicts=None, bin_by='costheta',
                                  xlabel='', ylabel='Candidates', label='', fig=None, axs=None, add_labels=False, legend=False, bin_centers=None, **kwargs):
        """
        Plot the differential scatter plot for each bin of bins1.
        
        Parameters
        ----------
        series2 : array-like (N or B)
            The series to bin into the bins of series1. If length is N, then series2 is binned into the bins of series1. 
            If length is B, then we assume series2 is arranged by the differential bins.
        bins1 : array-like (N or B)
            The bins that series2 is being binned into. Should map 1:1 with series2.
            If length is B, then we assume series2 is arranged by the differential bins.
        bins2 : array-like (B)
            The bins to bin series2 into.
        weights : array-like (N), optional
            The weights to use for the scatter plot. If None, no weights are used.
        yerrs : array-like (B), optional
            The y-errors to use for the scatter plot. If None, errors are calculated as sqrt(counts).
        diff_dicts : dict, optional
            The differential dictionaries to use for the scatter plot. Contains the edges and labels for each differential bin.
            If None, uses self.differential_dicts.
        bin_by : str
            The series to bin by.
        xlabel : str
            The label for the x-axis.
        ylabel : str
            The label for the y-axis.
        label : str
            The label for the scatter plot.
        fig : matplotlib.figure.Figure, optional
            The figure to use for the scatter plot.
        axs : matplotlib.axes.Axes, optional
            The axes to use for the scatter plot.
        add_labels : bool
            Whether to add the labels to the scatter plot.
        legend : bool
            Whether to add the legend to the scatter plot.
        bin_centers : array-like (B-1), optional
            The centers of the bins to use for the scatter plot. If None, the centers are calculated as (bin_edges[:-1] + bin_edges[1:])/2.
        **kwargs : dict
            Keyword arguments to pass to the plotting function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the scatter plot.
        axs : matplotlib.axes.Axes
            The axes containing the scatter plot.
        """
        bin_by_text = r'$\cos\theta_{\mu}$' if bin_by == 'costheta' else r'$p_{\mu}$'
        # Get hist_dict
        hist_dict, bin_by_edges = self.bin_differential_dict(series2, bins1, bins2, diff_dicts, weights, bin_by=bin_by)
        if fig is None and axs is None:
            fig, axs = plt.subplots(figsize=(12, 8), nrows=3, ncols=3)
        assert len(axs.flatten()) == len(hist_dict), f'axs and hist_dict must be the same length: {len(axs.flatten())} != {len(hist_dict)}'
        if yerrs is not None:
            yerr_hist_dict,_ = self.bin_differential_dict_binned(yerrs, diff_dicts=diff_dicts, bin_by=bin_by)
        for i,(ax, h, b) in enumerate(zip(axs.flatten(), hist_dict.values(), bin_by_edges)):
            centers = bin_centers if bin_centers is not None else (h[1][:-1] + h[1][1:])/2
            if yerrs is not None:
                errors = yerr_hist_dict[i][0]
            else:
                errors = np.sqrt(h[0])
            ax.errorbar(centers, h[0], yerr=errors, fmt='o', label=label, **kwargs)
            if bin_by == 'costheta':
                ax.set_xlim(0, MAX_PMOM)
            bin_text = f'{b[0]:.2f} < {bin_by_text} < {b[1]:.2f}'
            if add_labels:
                plotters.add_label(ax, bin_text, where=(1.,0.75), horizontalalignment='right', color='black', alpha=1., fontsize=8)
        # Add the labels
        axs[2, 1].set_xlabel(xlabel)
        axs[1, 0].set_ylabel(ylabel)
        if legend:
            axs[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        return fig, axs
    
    def plot_differential_hist_binned(self,values,yerrs=None,bin_by='costheta',xlabel='',ylabel='Candidates',label='',fig=None,axs=None,add_labels=False,legend=False,**kwargs):
        """
        Plot the differential histograms
        """
        bin_by_text = r'$\cos\theta_{\mu}$' if bin_by == 'costheta' else r'$p_{\mu}$'
        if fig is None and axs is None:
            fig, axs = plt.subplots(figsize=(12, 8), nrows=3, ncols=3)
        assert len(values) == len(self.differential_centers), f'values and differential_centers must be the same length: {len(values)} != {len(self.differential_centers)}'
        hist_dict, bin_by_edges = self.bin_differential_dict_binned(values, diff_dicts=self.differential_dicts, bin_by=bin_by)
        if yerrs is not None:
            assert len(yerrs) == len(self.differential_centers), f'yerrs and differential_centers must be the same length: {len(yerrs)} != {len(self.differential_centers)}'
            yerr_hist_dict, _ = self.bin_differential_dict_binned(yerrs, diff_dicts=self.differential_dicts, bin_by=bin_by)
        for ax, (k, h), b in zip(axs.flatten(), hist_dict.items(), bin_by_edges):
            # Always draw the histogram values (step). No errors unless yerrs provided.
            makeplot.plot_hist_edges(h[1], h[0], errors=None, label=label, ax=ax, **kwargs)

            # Optional uncertainty band overlay
            if yerrs is not None:
                yerr_arr = yerr_hist_dict[k][0]
                makeplot.plot_hist_with_uncertainty(bins=h[1], n_perbin=h[0], yerr_arr=yerr_arr, yerr_label='Unc', ax=ax)
            if bin_by == 'costheta':
                ax.set_xlim(0, MAX_PMOM)
            bin_text = f'{b[0]:.2f} < {bin_by_text} < {b[1]:.2f}'
            if add_labels:
                plotters.add_label(ax, bin_text, where=(1.,0.75), horizontalalignment='right', color='black', alpha=1., fontsize=8)
        #Add the labels
        if add_labels:
            axs[2, 1].set_xlabel(xlabel)
            axs[1, 0].set_ylabel(ylabel)
        if legend:
            axs[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        return fig, axs

    def plot_differential_stairs_binned(
        self,
        stairs_y,
        bin_by='costheta',
        xlabel='',
        ylabel='Candidates',
        label='',
        fig=None,
        axs=None,
        add_labels=False,
        legend=False,
        diff_dicts=None,
        **kwargs,
    ):
        """
        Filled step plot of per-differential-bin heights on the 3x3 differential grid.

        ``stairs_y`` must align with ``self.differential_centers`` (same layout as
        ``plot_differential_hist_binned``). Typical use: norm syst band height
        ``sig_unfold * yscale * fracunc_unfold_norm`` computed outside, then passed here.

        Parameters
        ----------
        stairs_y : array-like
            Stair heights per differential bin, length ``len(self.differential_centers)``.
        bin_by : str
            ``'costheta'`` or ``'momentum'`` (passed to ``bin_differential_dict_binned``).
        xlabel, ylabel : str
            Axis labels when ``add_labels`` is True.
        label : str
            Legend label for ``ax.stairs``.
        fig, axs : optional
            Figure and 3x3 axes; created if both None.
        add_labels, legend : bool
            Panel annotations and legend (same placement as ``plot_differential_hist_binned``).
        diff_dicts : dict, optional
            Differential dicts for binning; default ``self.differential_dicts``.
        **kwargs
            Extra arguments for ``matplotlib.axes.Axes.stairs`` (defaults:
            ``fill=True``, ``color='gray'``, ``alpha=0.5``).

        Returns
        -------
        fig, axs
        """
        bin_by_text = r'$\cos\theta_{\mu}$' if bin_by == 'costheta' else r'$p_{\mu}$'
        if fig is None and axs is None:
            fig, axs = plt.subplots(figsize=(12, 8), nrows=3, ncols=3)
        assert len(stairs_y) == len(self.differential_centers), (
            f'stairs_y and differential_centers must be the same length: '
            f'{len(stairs_y)} != {len(self.differential_centers)}'
        )
        dd = self.differential_dicts if diff_dicts is None else diff_dicts
        stairs_hist_dict, bin_by_edges = self.bin_differential_dict_binned(
            stairs_y, diff_dicts=dd, bin_by=bin_by
        )
        stairs_kw = dict(fill=True, color='gray', alpha=0.5)
        stairs_kw.update(kwargs)
        stairs_kw.pop('label', None)
        for ax, h, b in zip(axs.flatten(), stairs_hist_dict.values(), bin_by_edges):
            ax.stairs(h[0], h[1], label=label, **stairs_kw)
            if bin_by == 'costheta':
                ax.set_xlim(0, MAX_PMOM)
            bin_text = f'{b[0]:.2f} < {bin_by_text} < {b[1]:.2f}'
            if add_labels:
                plotters.add_label(ax, bin_text, where=(1.,0.75), horizontalalignment='right', color='black', alpha=1., fontsize=8)
        if add_labels:
            axs[2, 1].set_xlabel(xlabel)
            axs[1, 0].set_ylabel(ylabel)
        if legend:
            axs[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        return fig, axs

    def plot_differential_scatter_binned(self,values,bin_centers=None,yerrs=None,bin_by='costheta',xlabel='',ylabel='Candidates',label='',fig=None,axs=None,add_labels=False,legend=False,**kwargs):
        """
        Plot the differential scatter plot for each bin of bins1.
        """
        bin_by_text = r'$\cos\theta_{\mu}$' if bin_by == 'costheta' else r'$p_{\mu}$'
        if fig is None and axs is None:
            fig, axs = plt.subplots(figsize=(12, 8), nrows=3, ncols=3)
        hist_dict, bin_by_edges = self.bin_differential_dict_binned(values, diff_dicts=self.differential_dicts, bin_by=bin_by)
        if yerrs is not None:
            yerr_hist_dict,_ = self.bin_differential_dict_binned(yerrs, diff_dicts=self.differential_dicts, bin_by=bin_by)
        for i, (ax, h, b) in enumerate(zip(axs.flatten(), hist_dict.values(), bin_by_edges)):
            centers = bin_centers[i] if bin_centers is not None else (h[1][:-1] + h[1][1:])/2
            if yerrs is not None:
                errors = yerr_hist_dict[i][0]
            else:
                errors = None
            ax.errorbar(centers, h[0], yerr=errors, fmt='o', label=label, **kwargs)
            if bin_by == 'costheta':
                ax.set_xlim(0, MAX_PMOM)
            bin_text = f'{b[0]:.2f} < {bin_by_text} < {b[1]:.2f}'
            if add_labels:
                plotters.add_label(ax, bin_text, where=(1.,0.75), horizontalalignment='right', color='black', alpha=1., fontsize=8)
        #Add the labels
        if add_labels:
            axs[2, 1].set_xlabel(xlabel)
            axs[1, 0].set_ylabel(ylabel)
        if legend:
            axs[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        return fig, axs

    def to_latex(self, caption=None, label=None, save_dir=None,
                         filename='binning2d.tex'):
        """
        Render the differential binning as a LaTeX longtblr table.

        Columns: Bin ID, p_mu low (GeV), p_mu high (GeV),
        cos(theta) low, cos(theta) high, Bin Width (GeV).

        Parameters
        ----------
        caption : str, optional
        label : str, optional
        save_dir : str, optional
            Directory to save the .tex file.
        filename : str
            Output filename (default ``binning2d.tex``).

        Returns
        -------
        str
            LaTeX source.
        """
        import os

        options = []
        if caption:
            options.append(f'  caption = {{{caption}}}')
        if label:
            options.append(f'  label = {{{label}}}')
        options_block = ',\n'.join(options)

        col_spec = 'cccc'
        lines = [r'\scriptsize']
        if options_block:
            lines.append(r'\begin{longtblr}[')
            lines.append(options_block)
            lines.append(rf']{{colspec={{{col_spec}}}}}')
        else:
            lines.append(rf'\begin{{longtblr}}{{colspec={{{col_spec}}}}}')
        lines.append(r'\hline')
        lines.append(
            r'Bin ID & $p_\mu$ (GeV) & $\cos\theta_\mu$ & Bin Width (GeV) \\'
        )
        lines.append(r'\hline')

        for bin_id in sorted(self.differential_dicts.keys()):
            if bin_id < 0:
                continue
            d = self.differential_dicts[bin_id]
            p_lo, p_hi = d['momentum_edges']
            ct_lo, ct_hi = d['costheta_edges']
            bw = d['bin_width']
            lines.append(
                f'{int(bin_id)} & [{p_lo:.2f}, {p_hi:.2f}] '
                f'& [{ct_lo:.2f}, {ct_hi:.2f}] & {bw:.4f} \\\\'
            )

        lines.append(r'\hline')
        lines.append(r'\end{longtblr}')
        result = '\n'.join(lines)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, filename)
            with open(path, 'w') as f:
                f.write(result)
            print(f'Binning table saved to {path}')

        return result

    #Works, but it's just a combination of the two above functions
    # def create_differential_histograms(self, series2, bins1, bins2, weights, data_series, data_bins, yerrs=None, diff_dicts=None, bin_by='costheta',
    #     xlabel='', ylabel='Candidates', label='', fig=None, axs=None, add_labels=False, legend=False, **kwargs):
    #     """
    #     Create the differential histograms for each bin of bins1.

    #     Parameters
    #     ----------
    #     series2 : array-like (N)
    #         The series to bin into the bins of series1.
    #     bins1 : array-like (N)
    #         The bins that series2 is being binned into. Should map 1:1 with series2.
    #         If length is B, then we assume series2 is arranged by the differential bins.
    #     bins2 : array-like (B)
    #         The bins to bin series2 into.
    #     weights : array-like (N)
    #         The weights to use for the histograms.
    #     data_series : array-like (N)
    #         The data series to use for the histograms.
    #     data_bins : array-like (N)
    #         The bins to bin data series into.
    #     yerrs : array-like (B), optional
    #         The y-errors to use for the histograms. If None, no errors are plotted.
    #     diff_dicts : dict, optional
    #         The differential dictionaries to use for the histograms. Contains the edges and labels for each differential bin.
    #         If None, uses self.differential_dicts.
    #     bin_by : str
    #         The series to bin by.
    #     xlabel : str
    #         The label for the x-axis.
    #     ylabel : str
    #         The label for the y-axis.
    #     label : str
    #         The label for the histograms.
    #     fig : matplotlib.figure.Figure, optional
    #         The figure to use for the histograms.
    #     axs : matplotlib.axes.Axes, optional
    #         The axes to use for the histograms.
    #     add_labels : bool
    #         Whether to add the labels to the histograms.
    #     legend : bool
    #         Whether to add the legend to the histograms.
    #     **kwargs : dict
    #         Keyword arguments to pass to the plotting function.

    #     Returns
    #     -------
    #     fig : matplotlib.figure.Figure
    #         The figure containing the differential histograms.
    #     axs : matplotlib.axes.Axes
    #         The axes containing the differential histograms.
    #     """
    #     bin_by_text = r'$\cos\theta_{\mu}$' if bin_by == 'costheta' else r'$p_{\mu}$'
    #     # Get hist_dicts
    #     hist_dict, bin_by_edges = self.bin_differential_dict(series2, bins1, bins2, diff_dicts, weights, bin_by=bin_by)
    #     data_hist_dict, data_bin_by_edges = self.bin_differential_dict(data_series, data_bins, bins2, diff_dicts, np.ones(len(data_series)), bin_by=bin_by)
    #     if fig is None and axs is None:
    #         fig, axs = plt.subplots(figsize=(12, 8), nrows=3, ncols=3)
    #     assert len(axs.flatten()) == len(hist_dict) == len(data_hist_dict), f'axs and hist_dict and data_hist_dict must be the same length: {len(axs.flatten())} != {len(hist_dict)} != {len(data_hist_dict)}'
    #     yerr_indexer = 0
    #     for ax, h, dh, b, db in zip(axs.flatten(), hist_dict.values(), data_hist_dict.values(), bin_by_edges, data_bin_by_edges):
    #         makeplot.plot_hist_edges(h[1], h[0], errors=yerrs[yerr_indexer:yerr_indexer+len(h[0])]*h[0], label=label, ax=ax, **kwargs)
    #         data_centers = (dh[1][:-1] + dh[1][1:])/2
    #         ax.errorbar(data_centers, dh[0], yerr=np.sqrt(dh[0]), fmt='o', color='black', label='Data')
    #         yerr_indexer += len(h[0])
    #         if bin_by == 'costheta':
    #             ax.set_xlim(0, 4.)
    #         bin_text = f'{b[0]:.2f} < {bin_by_text} < {b[1]:.2f}'
    #         if add_labels:
    #             plotters.add_label(ax, bin_text, where=(1.,0.75), horizontalalignment='right', color='black', alpha=1., fontsize=8)
    #     # Add the labels
    #     axs[2, 1].set_xlabel(xlabel)
        