import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sbnd.general import utils
from sbnd.general import plotters
import numpy as np

day = plotters.day

def plot_hist(series,labels,xlabel='',title=None,cmap='viridis',draw_edge=False,
              **pltkwargs):
  """
  series is a list of pd.Series
  """
  fig,ax = plt.subplots(figsize=(7,4))
  legend_labels = [f'{lab} ({len(series[i])})' for i,lab in enumerate(labels)]
  colors = cm.get_cmap(cmap, len(labels))
  colors = [list(colors(i)) for i in range(len(labels))]
  edgecolors = colors.copy()
  print(edgecolors)
  if draw_edge:
    ax.hist(series,label=legend_labels,color=colors,edgecolor=edgecolors,
            **pltkwargs)
  else:
    ax.hist(series,label=legend_labels,color=colors,**pltkwargs)
  ax.set_xlabel(xlabel)
  if title is not None:
    ax.set_title(title)
  ax.legend()
  return fig,ax

def plot_hist2d(x,y,xlabel='',ylabel='',title=None,cmap='Blues',plot_line=False,label_boxes=False,
                **pltkwargs):
  """
  x,y are pd.Series
  """
  fig,ax = plt.subplots(figsize=(6,6),tight_layout=True)

  if plot_line:
    lower = np.min([0,min(x),min(y)])
    upper = np.max([max(x),max(y)])
    xy = [lower,upper]
    ax.plot(xy,xy,ls='--',color='red')
    ax.set_xlim([lower,upper])
    ax.set_ylim([lower,upper])
    hist,xbins,ybins,im = ax.hist2d(x,y,range=[xy,xy],**pltkwargs)
  else:
    hist,xbins,ybins,im = ax.hist2d(x,y,**pltkwargs)
  if label_boxes:
    for i in range(len(ybins)-1):
      for j in range(len(xbins)-1):
        ax.text(xbins[j]+0.5,ybins[i]+0.5, f'{hist.T[i,j]:.0f}', 
                color="w", ha="center", va="center", fontweight="bold",fontsize=16)
  ax.set_xlabel(f'{xlabel}')
  ax.set_ylabel(f'{ylabel}')
  if title is not None:
    ax.set_title(title)
  plotters.set_style(ax)
  return fig,ax

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
          e = plt.errorbar(centers, values, yerr=errors, elinewidth=3,fmt='none', color=h[0].get_color(),alpha=0.6)
  else:
      h = ax.step(edges, list(values)+[values[-1]], where='post', label=label,**pltkwargs)
      if errors is not None: 
          e = ax.errorbar(centers, values, yerr=errors, elinewidth=3,fmt='none', color=h[0].get_color(),alpha=0.6) 
  
  
