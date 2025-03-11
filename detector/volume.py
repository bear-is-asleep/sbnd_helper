import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .definitions import *

def involume(coords,volume=AV):
  xb,yb,zb = volume
  if isinstance(coords, pd.DataFrame):
    return ((xb[0] < coords.x) & (coords.x < xb[1]) &
        (yb[0] < coords.y) & (coords.y < yb[1]) &
        (zb[0] < coords.z) & (coords.z < zb[1]))
  else:
    print('Make a dataframe or GET OUT OF MY FACE')
    raise Exception('deal with it')

def plot_volume_boundary(ax=None,volume=FV,view='xy',**kwargs):
  """
  Plot lines of the volume
  """
  if ax is None:
    _,ax = plt.subplots(figsize=(8,8))
  xb,yb,zb = volume
  if view == 'xy':
    #bottom
    ax.plot([xb[0],xb[1]],[yb[0],yb[0]],**kwargs)
    ax.plot([xb[0],xb[1]],[yb[1],yb[1]],**kwargs)
    ax.plot([xb[0],xb[0]],[yb[0],yb[1]],**kwargs)
    ax.plot([xb[1],xb[1]],[yb[0],yb[1]],**kwargs)
    #top
    ax.plot([xb[0],xb[1]],[yb[0],yb[0]],**kwargs)
    ax.plot([xb[0],xb[1]],[yb[1],yb[1]],**kwargs)
    ax.plot([xb[0],xb[0]],[yb[0],yb[1]],**kwargs)
    ax.plot([xb[1],xb[1]],[yb[0],yb[1]],**kwargs)
    #sides
    ax.plot([xb[0],xb[0]],[yb[0],yb[0]],**kwargs)
    ax.plot([xb[1],xb[1]],[yb[1],yb[1]],**kwargs)
    ax.plot([xb[0],xb[0]],[yb[1],yb[1]],**kwargs)
    ax.plot([xb[1],xb[1]],[yb[0],yb[0]],**kwargs)
  elif view == 'xz':
    #bottom
    ax.plot([xb[0],xb[1]],[zb[0],zb[0]],**kwargs)
    ax.plot([xb[0],xb[1]],[zb[1],zb[1]],**kwargs)
    ax.plot([xb[0],xb[0]],[zb[0],zb[1]],**kwargs)
    ax.plot([xb[1],xb[1]],[zb[0],zb[1]],**kwargs)
    #top
    ax.plot([xb[0],xb[1]],[zb[0],zb[0]],**kwargs)
    ax.plot([xb[0],xb[1]],[zb[1],zb[1]],**kwargs)
    ax.plot([xb[0],xb[0]],[zb[0],zb[1]],**kwargs)
    ax.plot([xb[1],xb[1]],[zb[0],zb[1]],**kwargs)
    #sides
    ax.plot([xb[0],xb[0]],[zb[0],zb[0]],**kwargs)
    ax.plot([xb[1],xb[1]],[zb[1],zb[1]],**kwargs)
    ax.plot([xb[0],xb[0]],[zb[1],zb[1]],**kwargs)
    ax.plot([xb[1],xb[1]],[zb[0],zb[0]],**kwargs)
  elif view == 'yz':
    #bottom
    ax.plot([yb[0],yb[1]],[zb[0],zb[0]],**kwargs)
    ax.plot([yb[0],yb[1]],[zb[1],zb[1]],**kwargs)
    ax.plot([yb[0],yb[0]],[zb[0],zb[1]],**kwargs)
    ax.plot([yb[1],yb[1]],[zb[0],zb[1]],**kwargs)
    #top
    ax.plot([yb[0],yb[1]],[zb[0],zb[0]],**kwargs)
    ax.plot([yb[0],yb[1]],[zb[1],zb[1]],**kwargs)
    ax.plot([yb[0],yb[0]],[zb[0],zb[1]],**kwargs)
    ax.plot([yb[1],yb[1]],[zb[0],zb[1]],**kwargs)
    #sides
    ax.plot([yb[0],yb[0]],[zb[0],zb[0]],**kwargs)
    ax.plot([yb[1],yb[1]],[zb[1],zb[1]],**kwargs)
    ax.plot([yb[0],yb[0]],[zb[1],zb[1]],**kwargs)
    ax.plot([yb[1],yb[1]],[zb[0],zb[0]],**kwargs)
  return ax

