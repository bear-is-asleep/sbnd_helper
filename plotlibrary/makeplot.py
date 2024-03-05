import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sbnd.general import utils
from sbnd.general import plotters
import numpy as np
from matplotlib.colors import LogNorm

day = plotters.day

def plot_hist(series,labels,xlabel='',title=None,cmap='viridis',colors=None,weights=None,
              **pltkwargs):
  """
  series is a list of pd.Series
  """
  fig,ax = plt.subplots(figsize=(7,4))
  if weights is None:
    counts = [len(s) for s in series]
  else:
    counts = [round(np.sum(w)) for w in weights]
  legend_labels = [f'{lab} ({counts[i]:,})' for i,lab in enumerate(labels)]
  if colors is None:
    colors = cm.get_cmap(cmap, len(labels))
    colors = [list(colors(i)) for i in range(len(labels))]
  #edgecolors = [plotters.darken_color(c,factor=0.5) for c in colors]
  #for s, c, e, w, l in zip(series, colors, edgecolors, weights, legend_labels):
  ax.hist(series, label=legend_labels, color=colors, weights=weights, **pltkwargs)
  ax.set_xlabel(xlabel)
  if title is not None:
    ax.set_title(title)
  ax.legend()
  return fig,ax

def plot_hist2d(x,y,xlabel='',ylabel='',title=None,cmap='Blues',plot_line=False,label_boxes=False,
                colorbar=False,ax=None,fig=None,**pltkwargs):
  """
  x,y are pd.Series
  """
  if fig is None and ax is None: #Make figure if not provided
    fig,ax = plt.subplots(figsize=(6,6),tight_layout=True)
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
    for i in range(len(ybins)-1):
      for j in range(len(xbins)-1):
        ax.text(xbins[j]+0.5,ybins[i]+0.5, f'{hist.T[i,j]:.0f}', 
                color="w", ha="center", va="center", fontweight="bold",fontsize=16)
  if colorbar:
    fig.colorbar(im, ax=ax)
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
  assume y is the true var
  """
  fig,(ax,ax2) = plt.subplots(2,1,figsize=(6,7),tight_layout=True,sharex=True,gridspec_kw={'height_ratios': [6, 1]})
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
      statistic = (_x-_y)/_y
    else:
      statistic = _x-_y
    bias[i] = np.mean(statistic)
    err[i] = np.std(statistic)
  ax2.errorbar(bin_centers,bias,yerr=err,fmt='o',color='black')
  ax2.axhline(0,ls='--',color='red')
  ax2.set_xlabel(xlabel)
  if normalize:
    ax2.set_ylabel('Fractional Error')
  else:
    ax2.set_ylabel('Bias')
  fig,ax = plot_hist2d(x,y,xlabel='',ylabel=ylabel,title=title,cmap=cmap,plot_line=plot_line,label_boxes=label_boxes,colorbar=colorbar
                    ,fig=fig,ax=ax,**pltkwargs)
  plotters.set_style(ax2)
  return fig,(ax,ax2)
    

def plot_hist_edges(edges,values,errors,label,ax=None,**pltkwargs):
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