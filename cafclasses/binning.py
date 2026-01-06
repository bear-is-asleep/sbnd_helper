import numpy as np
import matplotlib.pyplot as plt
from sbnd.plotlibrary import makeplot
from sbnd.general import plotters
from sbnd.numu.numu_constants import *

DIFF_COSTHETA_BINS = np.array([-1, -0.5, 0, 0.27, 0.45, 0.62, 0.76, 0.86, 0.94, 1])
DIFF_MOMENTUM_BINS = np.array([0, 0.3, 0.5, 0.7, 1.1, 1e10])


class Binning2D:
    """Class for handling 2D differential binning in costheta and momentum."""
    
    def __init__(self, diff_costheta_bins=None, diff_momentum_bins=None):
        """
        Initialize 2D binning configuration.
        
        Parameters
        ----------
        diff_costheta_bins : array-like, optional
            Bin edges for costheta. Default: [-1, -0.5, 0, 0.27, 0.45, 0.62, 0.76, 0.86, 0.94, 1]
        diff_momentum_bins : array-like, optional
            Bin edges for momentum. Default: [0, 0.3, 0.5, 0.7, 1.1, 1e10]
        """
        # Default values
        if diff_costheta_bins is None:
            diff_costheta_bins = DIFF_COSTHETA_BINS
        if diff_momentum_bins is None:
            diff_momentum_bins = DIFF_MOMENTUM_BINS
        
        # Convert to numpy arrays if needed
        self.diff_costheta_bins = np.array(diff_costheta_bins)
        self.diff_momentum_bins = np.array(diff_momentum_bins)
        
        # Costheta bins and labels
        self.diff_costheta_bin_labels = [f'{self.diff_costheta_bins[i]:.2f} - {self.diff_costheta_bins[i+1]:.2f}' 
                                         for i in range(len(self.diff_costheta_bins)-1)]
        self.diff_costheta_centers = (self.diff_costheta_bins[:-1] + self.diff_costheta_bins[1:])/2.
        
        # Theta bins (converted from costheta)
        self.diff_theta_bins = np.arccos(self.diff_costheta_bins)*180/np.pi
        self.diff_theta_bin_labels = [f'{self.diff_theta_bins[i]:.2f} - {self.diff_theta_bins[i+1]:.2f}' 
                                      for i in range(len(self.diff_theta_bins)-1)]
        self.diff_theta_centers = (self.diff_theta_bins[:-1] + self.diff_theta_bins[1:])/2.
        
        # Momentum bins and labels
        self.diff_momentum_bin_labels = [f'{self.diff_momentum_bins[i]:.2f} - {self.diff_momentum_bins[i+1]:.2f}' 
                                         if i < len(self.diff_momentum_bins)-2 
                                         else f'> {self.diff_momentum_bins[i]:.2f}' 
                                         for i in range(len(self.diff_momentum_bins)-1)]
        self.diff_momentum_centers = (self.diff_momentum_bins[:-1] + self.diff_momentum_bins[1:])/2.
        
        # Differential bins
        self.differential_bins = (len(self.diff_costheta_bins)-2) + (len(self.diff_momentum_bins)-2)*(len(self.diff_costheta_bins)-1)
        self.differential_edges = np.arange(-1.5, np.max(self.differential_bins)+1.5, 1)
        self.differential_centers = (self.differential_edges[:-1] + self.differential_edges[1:])/2.
        
        # Template for differential dictionaries
        diff_dict_template = {
            'costheta_bin': -1,
            'momentum_bin': -1,
            'costheta_edges': [-np.inf, np.inf],
            'momentum_edges': [-np.inf, np.inf],
            'momentum_center': -np.inf,
            'costheta_center': -np.inf
        }
        
        # Initialize differential dictionaries
        self.differential_dicts = {c: diff_dict_template.copy() for c in sorted(self.differential_centers[1:])}
        self._initialize_differential_dicts()
    
    def _initialize_differential_dicts(self):
        """Initialize the differential dictionaries with bin information."""
        for c in self.differential_dicts:
            c_int = int(c)
            cbin = np.mod(c_int, len(self.diff_costheta_bins)-1)
            pbin = c_int//(len(self.diff_costheta_bins)-1)
            self.differential_dicts[c]['costheta_bin'] = cbin
            self.differential_dicts[c]['momentum_bin'] = pbin
            self.differential_dicts[c]['costheta_edges'] = self.diff_costheta_bins[cbin:cbin+2]
            self.differential_dicts[c]['momentum_edges'] = self.diff_momentum_bins[pbin:pbin+2]
            self.differential_dicts[c]['momentum_center'] = self.diff_momentum_centers[pbin]
            self.differential_dicts[c]['costheta_center'] = self.diff_costheta_centers[cbin]
    
    def bin_differential_dict(self, series2, bins1, bins2, diff_dicts=None, weights=None, bin_by='costheta'):
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
        # Initialize the histogram dictionary by the unique values of bin_by key
        keys = np.unique([v[f'{bin_by}_bin'] for v in diff_dicts.values()])
        hist_dict = {k: None for k in keys}

        # Get bin edges for the bin_by series
        bin_by_edges = np.unique([np.array(v[f'{bin_by}_edges']) for v in diff_dicts.values()], axis=0)

        # Loop over the unique values of bin_by
        for k in keys:
            mask = bins1 == k
            hist_dict[k] = np.histogram(series2[mask], bins=bins2, weights=weights[mask])

        assert len(bin_by_edges) == len(hist_dict), f'bin_by_edges and hist_dict must be the same length: {len(bin_by_edges)} != {len(hist_dict)}'
        return hist_dict, bin_by_edges
    
    def plot_differential_hist(self, series2, bins1, bins2, weights=None, yerrs=None, diff_dicts=None, bin_by='costheta',
                               xlabel='', ylabel='Candidates', label='', fig=None, axs=None, legend=False, add_labels=False, **kwargs):
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
        yerr_indexer = 0
        for ax, h, b in zip(axs.flatten(), hist_dict.values(), bin_by_edges):
            errors = yerrs[yerr_indexer:yerr_indexer+len(h[0])]*h[0] if yerrs is not None else None
            makeplot.plot_hist_edges(h[1], h[0], errors=errors, label=label, ax=ax, **kwargs)
            yerr_indexer += len(h[0])
            if bin_by == 'costheta':
                ax.set_xlim(0, 4.)
            bin_text = f'{b[0]:.2f} < {bin_by_text} < {b[1]:.2f}'
            if add_labels:
                plotters.add_label(ax, bin_text, where='centerright', color='black', alpha=1., fontsize=8)
        # Add the labels
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
        yerr_indexer = 0
        for ax, h, b in zip(axs.flatten(), hist_dict.values(), bin_by_edges):
            centers = bin_centers if bin_centers is not None else (h[1][:-1] + h[1][1:])/2
            if yerrs is not None:
                errors = yerrs[yerr_indexer:yerr_indexer+len(h[0])]*h[0]
            else:
                errors = np.sqrt(h[0])
            ax.errorbar(centers, h[0], yerr=errors, fmt='o', label=label, **kwargs)
            yerr_indexer += len(h[0])
            if bin_by == 'costheta':
                ax.set_xlim(0, 4.)
            bin_text = f'{b[0]:.2f} < {bin_by_text} < {b[1]:.2f}'
            if add_labels:
                plotters.add_label(ax, bin_text, where='centerright', color='black', alpha=1., fontsize=8)
        # Add the labels
        axs[2, 1].set_xlabel(xlabel)
        axs[1, 0].set_ylabel(ylabel)
        if legend:
            axs[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        return fig, axs
    
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
    #             plotters.add_label(ax, bin_text, where='centerright', color='black', alpha=1., fontsize=8)
    #     # Add the labels
    #     axs[2, 1].set_xlabel(xlabel)
        