import matplotlib.pyplot as plt
#import matplotlib.cm as cm
from sbnd.general import utils
from sbnd.general import plotters
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

day = plotters.day

from sbnd.plotlibrary import makeplot
from sbnd.general import plotters
from sbnd.general import utils
def create_hist(series,labels,dens=False,yerr=None,yerr_label='Total',data_series=None,scale_data=False,cut_desc='',xlabel='',label='',colors=None,weights=None,bins=20,cut='',savename=''
                ,plot_dir=None,stat_label='',data_events=None,dens_norm=15,return_counts=False,pot_label='',close=True,legend=True,show_counts=True,show_pcts=True,bin_centers=None,**kwargs):
    """
    Create a histogram from a list of series each corresponding
    to a different true event type. Data series is optional.

    Parameters
    ----------
    series : list of pd.Series
        List of series, each corresponding to a different true event type.
    labels : list of str
        List of labels, each corresponding to a different true event type.
    dens : bool
        If True, the histogram is normalized to a density.
    yerr : list of float, optional
        yerr on MC event count. Will be a hatched bar centered on the MC event count.
    data_series : pd.Series, optional
        Series which contains data.
    scale_data : bool, optional
        If True, the data series is scaled to the total number of events.
    cut_desc : str, optional
        Description of the cut.
    label : str, optional
        Label of the plot.
    xlabel : str, optional
        Label of the x-axis.
    colors : list of str, optional
        List of colors, each corresponding to a different true event type.
    weights : list of float, optional
        List of weights, each corresponding to a different true event type.
    bins : int, optional
        Number of bins in the histogram.
    cut : str, optional
        Cut applied to the data.
    savename : str, optional
        Name of the file to save the histogram.
    stat_label : str, optional
        Label of the statistic.
    data_events : int, optional
        Number of data events.
    dens_norm : float, optional
        Normalization factor for the density.
    return_counts : bool, optional
        If True, return the counts of the histogram.
    pot_label : str, optional
        Label of the POT.
    legend : bool, optional
        If True, show the legend.
    show_counts : bool, optional
        If True, show the counts in the legend.
    show_pcts : bool, optional
        If True, show the percentages in the legend.
    bin_centers : list of float, optional
        List of bin centers. If None, the bin centers are calculated from the bins.
    **kwargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the histogram.
    ax : matplotlib.axes.Axes
        Axes of the histogram.
    counts : np.ndarray
        Counts of the histogram.
    """
    
    if len(series) == 0:
        print(f'No events in series'+'\n'+f'xlabel: {xlabel}'+'\n'+f'cut: {cut}'+'\n'+f'savename: {savename}')
        return None,None,None if not return_counts else None,None
    if dens:
        histtype = 'step'
        alpha = 0.9
    else:
        histtype = 'barstacked'
        alpha = 0.8
    if isinstance(bins,int):
      _mins = [np.min(s) for s in series]
      _maxs = [np.max(s) for s in series]
      _min = np.min(_mins)
      _max = np.max(_maxs)
      bins = np.linspace(_min,_max,bins+1)
    fig,ax,counts,n_perbin = plot_hist(series,labels,xlabel=xlabel,colors=colors,weights=weights,return_counts=True
                   ,histtype=histtype,lw=2,bins=bins,alpha=alpha,density=dens,show_counts=show_counts,show_pcts=show_pcts,**kwargs)
    if data_series is not None:
        #Group data by binning, get mean and std of data series
        if bin_centers is None:
          bin_centers = (bins[:-1] + bins[1:]) / 2
        data_counts = data_series.groupby(pd.cut(data_series,bins=bins)).count()
        data_stds = np.sqrt(data_counts)
        if dens:
            data_counts = data_counts * (dens_norm/len(data_series))
            data_stds = data_stds * (dens_norm/len(data_series))
        if not dens and scale_data:
            data_counts = data_counts * (np.sum(counts)/len(data_series))
            data_stds = data_stds * (np.sum(counts)/len(data_series))
        if show_counts:
          if data_events is not None:
              data_events_label = f'Data ({utils.format_number_with_suffix(data_events)})'
          else:
              data_events_label = f'Data ({utils.format_number_with_suffix(data_counts.sum())})'
        else:
          data_events_label = None
        ax.errorbar(bin_centers,data_counts,yerr=data_stds,fmt='o',color='black',label=data_events_label)
    if yerr is not None:
        yerr_arr = np.asarray(yerr) 
        if len(yerr_arr) != len(bins) - 1:
            raise ValueError('yerr must have the same length as the histogram bins')
        #ax.errorbar(bin_centers,n_perbin,yerr=yerr_arr,fmt='o',color='black')
        ax.bar(
            (bins[:-1] + bins[1:])/2, #Don't use bin_centers due to width iisue
            2 * yerr_arr,
            bottom=n_perbin - yerr_arr,
            align='center',
            width=np.diff(bins),
            facecolor='none',
            edgecolor='gray',
            alpha=1.,
            linewidth=0.,
            hatch='xxx',
            label=yerr_label
        )

    if dens: ax.set_ylabel('Density')
    else: ax.set_ylabel('Candidates')
    y_lim = ax.get_ylim()
    if y_lim[1] > 1e6: #this means scientific notation kicks on, so we need to shift the label
      label_y_shift = 0.06
    else:
      label_y_shift = 0
    #Add labels
    plotters.add_label(ax,pot_label,where='toprightoutside',color='black',alpha=1.,fontsize=12)
    plotters.add_label(ax,label,where=(0.01,1.07+label_y_shift) if '\n' not in label else (0.01,1.15+label_y_shift)
      ,color='gray',alpha=0.9,fontsize=10,horizontalalignment='left',verticalalignment='top')
    plotters.add_label(ax,stat_label,where='bottomrightoutside',fontsize=10)
    plotters.add_label(ax,cut_desc,where='bottomrightoutside',color='black',fontsize=12)
    #plotters.set_style(ax)
    if legend:
      ax.legend(fontsize=10)
    if savename != '':
        if plot_dir is None:
          raise ValueError('plot_dir is None')
        plotters.save_plot(f'{savename}',fig=fig,folder_name=plot_dir)
        if close:
            plt.close('all')
    #else:
    #  plt.show()
    if return_counts:
      return fig,ax,n_perbin
    else:
      return fig,ax
    return fig,ax

def plot_hist(series,labels,xlabel='',title=None,cmap='viridis',colors=None,weights=None,return_counts=False,
              show_counts=True,show_pcts=True,**pltkwargs):
  """
  series is a list of pd.Series
  """
  fig,ax = plt.subplots(figsize=(7,4))
  if weights is None:
    counts = [len(s) for s in series]
  else:
    counts = [round(np.sum(w)) for w in weights]
  counts_str = [utils.format_number_with_suffix(c) for c in counts]
  if show_counts and show_pcts:
    legend_labels = [f'{lab} ({counts_str[i]}, {100*counts[i]/np.sum(counts):.1f}%)' for i,lab in enumerate(labels)]
  elif show_pcts:
    legend_labels = [f'{lab} ({100*counts[i]/np.sum(counts):.1f}%)' for i,lab in enumerate(labels)]
  elif show_counts:
    legend_labels = [f'{lab} ({counts_str[i]})' for i,lab in enumerate(labels)]
  else:
    legend_labels = labels
  if colors is None:
    cmap = plt.get_cmap(cmap, len(labels))
    colors = cmap(range(len(labels)))
    colors = [tuple(color) for color in colors]
  #edgecolors = [plotters.darken_color(c,factor=0.5) for c in colors]
  #for s, c, e, w, l in zip(series, colors, edgecolors, weights, legend_labels):
  n, bins, patches = ax.hist(series, label=legend_labels, color=colors, weights=weights, **pltkwargs)
  n = [_n[-1] for _n in n.T] # Convert to list of last elements of each bin, which is the total number of events in each bin
  ax.set_xlabel(xlabel)
  if title is not None:
    ax.set_title(title)
  #ax.legend()

  if return_counts:
    return fig, ax, counts, n
  else:
    return fig, ax

def plot_hist2d(x,y,xlabel='',ylabel='',title=None,cmap='Blues',plot_line=False,label_boxes=False,
                colorbar=False,ax=None,fig=None,text_color='wb',show_frac=False,**pltkwargs):
  """
  x,y are pd.Series
  text_color = bw for black as high values, white for low values and vise-versa
  """
  if fig is None and ax is None: #Make figure if not provided
    fig,ax = plt.subplots(figsize=(6.5,6),tight_layout=True)
  elif fig is None:
    fig = plt.gcf()
  elif ax is None:
    ax = fig.gca()

  if plot_line:
    lower = np.min([0,min(x),min(y)])
    upper = np.max([max(x),max(y)])
    xy = [lower,upper]
    ax.plot(xy,xy,ls='--',color='red')
    ax.set_xlim([lower,upper])
    ax.set_ylim([lower,upper])
    hist,xbins,ybins,im = ax.hist2d(x,y,range=[xy,xy],cmap=cmap,**pltkwargs)
  else:
    hist,xbins,ybins,im = ax.hist2d(x,y,cmap=cmap,**pltkwargs)
  if label_boxes:
    max_col = np.max(hist)
    for i in range(len(ybins)-1):
      for j in range(len(xbins)-1):
        val = hist.T[i,j]
        if val > max_col/2:
          if text_color == 'bw':
            color = 'k'
          elif text_color == 'wb':
            color = 'w'
        else:
          if text_color == 'bw':
            color = 'w'
          elif text_color == 'wb':
            color = 'k'
        if show_frac:
          val = val/np.sum(hist)
        ax.text(xbins[j]+0.5,ybins[i]+0.5, f'{val:0.2f}', 
                color=color, ha="center", va="center", fontweight="bold",fontsize=16)
  if colorbar:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    # fig_cb = plt.figure(figsize=(8, 1))
    # ax_cb = fig_cb.add_axes([0.05, 0.80, 0.9, 0.15])
    # cb = plt.colorbar(im, cax=ax_cb, orientation='horizontal')
  ax.set_xlabel(f'{xlabel}')
  ax.set_ylabel(f'{ylabel}')
  if title is not None:
    ax.set_title(title)
  plotters.set_style(ax)
  return fig,ax

def plot_hist2d_frac_err(x,y,xlabel='',ylabel='',title=None,cmap='Blues',plot_line=False,label_boxes=False,
                         colorbar=False,normalize=False,**pltkwargs):
  """
  x,y are pd.Series
  assume x is the true var
  """
  fig,(ax,ax2) = plt.subplots(2,1,figsize=(6.5,7),tight_layout=True,sharex=not colorbar,gridspec_kw={'height_ratios': [6, 1]})
  if 'bins' in pltkwargs:
    bins = pltkwargs['bins']
  else:
    raise ValueError('bins must be provided')
  bin_centers = (bins[1:] + bins[:-1])/2
  bias = np.zeros(len(bins)-1)
  err = np.zeros(len(bins)-1)
  for i in range(len(bins)-1):
    in_range = (x > bins[i]) & (x < bins[i+1])
    _x = x[in_range]
    _y = y[in_range]

    if normalize:
      statistic = (_y-_x)/_x
    else:
      statistic = _y-_x
    bias[i] = np.mean(statistic)
    err[i] = np.std(statistic)
  ax2.errorbar(bin_centers,bias,yerr=err,fmt='o',color='black')
  ax2.axhline(0,ls='--',color='red')
  ax2.set_xlabel(xlabel)
  if normalize:
    ax2.set_ylabel('Fractional\nError')
  else:
    ax2.set_ylabel(r'Bias')
  #Ensure errorbar xrange matches hist2d
  ax2.set_xlim([bins[0],bins[-1]])
  fig,ax = plot_hist2d(x,y,xlabel='',ylabel=ylabel,title=title,cmap=cmap,plot_line=plot_line,label_boxes=label_boxes,colorbar=colorbar
                    ,fig=fig,ax=ax,**pltkwargs)
  plotters.set_style(ax2)
  return fig,(ax,ax2)
    

def plot_hist_edges(edges,values,errors=None,label=None,ax=None,**pltkwargs):
  """
  Make step plot from edges and values
  
  Parameters
  ----------
  edges : array-like
      Edges of histogram bins
  values : array-like
      Values of histogram bins
  errors : array-like or None
      Errors of histogram bins
  label : str
      Label for legend
  ax : matplotlib.axes.Axes or None
      Axes to plot on. If None, use current axes.
  pltkwargs : dict
      Keyword arguments to pass to matplotlib.pyplot.step()
  
  """
  centers = (edges[1:] + edges[:-1])/2
  if ax is None:
      h = plt.step(edges, list(values)+[values[-1]], where='post', label=label,**pltkwargs)
      if errors is not None: 
          e = plt.errorbar(centers, values, yerr=errors, elinewidth=3,fmt='none', color=h[0].get_color(),alpha=0.6,capsize=7)
      else:
          e = None
  else:
      h = ax.step(edges, list(values)+[values[-1]], where='post', label=label,**pltkwargs)
      if errors is not None: 
          e = ax.errorbar(centers, values, yerr=errors, elinewidth=3,fmt='none', color=h[0].get_color(),alpha=0.6,capsize=7) 
      else:
          e = None
  return h,e

def draw_confusion_matrix_binned(hist, figure_name='', class_names=[], show_counts=True, figsize=(7,5), norm_ax=-1
                                 ,xlabel='True class',ylabel='Predicted class',textsize=14,rot=45):
    
    # Initialize figure
    fig,ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    
    # Normalize the histogram counts to the total number of entries in each true class bin
    if norm_ax == -1:
      norms = np.ones(len(hist))
    else:
      norms     = np.sum(hist, axis=norm_ax, keepdims=True) #norm_ax = 0 for purity, norm_ax = 1 for efficiency
    hist_norm = hist/norms

    # Initialize plot, fill
    n_classes = len(hist)
    xedges = yedges = -0.5+np.arange(0, n_classes + 1)
    hh = ax.pcolormesh(xedges, yedges, hist_norm, cmap='Blues')
    for i in range(n_classes):
        for j in range(n_classes):
            label = '{:0.3f}\n({:,})'.format(hist_norm[i,j], int(hist[i,j])) if show_counts else '{:0.3f}'.format(hist_norm[i,j])
            ax.text(j, i, label, color="white" if hist_norm[i,j] > 0.5 else "black", ha="center", va="center",fontsize=textsize)
            
    # Set axes style and labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(class_names) == n_classes:
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_names,rotation=rot)
        ax.set_yticklabels(class_names,rotation=rot)
    else:
        print(f'class_names: {class_names} is a different length than n_classes: {n_classes}')
    fig.colorbar(hh)
    
    return fig,ax

#TODO: Move this to neutrino class?
def make_mode_plots(nu_df,mode_map,weights=None,ylabel='Events',bins=np.arange(0,5.1,0.1),density=False,title=None,
                    ax=None,fig=None,**pltkwargs):
  norm = len(nu_df)/np.sum(nu_df.genweight.values)
  modes = np.unique(nu_df.genie_mode.values)
  Es = [None]*len(modes)
  counts = Es.copy()
  labels = Es.copy()
  if weights is not None:
    weight_modes = Es.copy()
  for i,mode in enumerate(modes):
    Es[i] = list(nu_df[nu_df.genie_mode==mode].E) #Get energy from mode
    labels[i] = f'{mode_map[mode]} : {round(len(Es[i])/norm):,}' #Mode label and 
    if weights is not None:
      weight_modes[i] = nu_df[nu_df.genie_mode==mode].genweight
  if fig is None and ax is None: #Make figure if not provided
    fig,ax = plt.subplots(figsize=(10,8))
  if not density:
    if weights is None:
      ax.hist(Es,stacked=True,label=labels,bins=bins,**pltkwargs)
    else:
      ax.hist(Es,stacked=True,label=labels,weights=weight_modes,bins=bins,**pltkwargs)
  if density:
    # Calculate total counts in each bin across all modes first
    if weights is None:
      total_counts, edges = np.histogram(np.concatenate(Es), bins=bins)
    else:
      total_counts, edges = np.histogram(np.concatenate(Es), bins=bins, weights=np.concatenate(weight_modes))
    bottom = np.zeros(len(total_counts))
    actual_counts = bottom.copy()
    for i,_ in enumerate(modes):
      if weights is None:
        counts, edges = np.histogram(Es[i], bins=bins)
      else:
        counts, edges = np.histogram(Es[i], weights=weight_modes[i], bins=bins)
      actual_counts += counts
      fractions = counts / total_counts
      ax.bar(edges[:-1], height=fractions, bottom=bottom, align='edge', width=np.diff(edges), label=labels[i],fill=True, **pltkwargs)
      # Add the fraction as text in the middle of the bar
      # bar_centers = edges[:-1] + np.diff(edges) / 2  # Calculate the center of each bar
      # for i,(center, fraction) in enumerate(zip(bar_centers, fractions)):
      #   if np.isnan(fraction): continue
      #   ax.text(center, bottom[i] + fraction / 2, f'{fraction*100:.1f}%', ha='center', va='center',rotation=90)
      bottom += fractions  # Update the bottom for the next mode
      bottom = [b if not np.isnan(b) else 0 for b in bottom]
    ax.grid(True)
  if title is None:
    title = rf'{round(len(nu_df)/norm):,} $\nu_\mu CC$ events'
  ax.set_title(title)
  ax.set_xlabel(r'$E_\nu$ [GeV]')
  if ylabel is not None:
    ax.set_ylabel(f'{ylabel} / {round((bins[1]-bins[0])*1e3):,} MeV')
  return fig,ax  