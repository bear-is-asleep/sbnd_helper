import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .definitions import *

def coord_columns(coords, label='coords'):
  if not isinstance(coords, pd.DataFrame):
    raise TypeError(f'{label} must be a pandas DataFrame')
  cols = coords.columns
  if set(('x', 'y', 'z')).issubset(cols):
    return coords['x'], coords['y'], coords['z']
  if coords.shape[1] != 3:
    raise ValueError(
      f'{label} must have x/y/z columns or exactly 3 coordinate columns; got {coords.shape[1]}'
    )
  return coords.iloc[:, 0], coords.iloc[:, 1], coords.iloc[:, 2]

def involume(coords, volume=AV):
  xb, yb, zb = volume
  x, y, z = coord_columns(coords)
  return (
      (xb[0] < x) & (x < xb[1]) &
      (yb[0] < y) & (y < yb[1]) &
      (zb[0] < z) & (z < zb[1])
  )

def involume_FV(coords):
  """
  Check if coordinates are within the FV volume, excluding the rejection regions
  """
  return involume(coords,volume=FV) & ~involume(coords,volume=NOT_FV_HIGH_Z)

def enters(start, end, volume=FV):
  """
  True if either endpoint is inside the volume (involume) or the segment
  between start and end passes through the open axis-aligned box.
  """
  xb, yb, zb = volume
  xs, ys, zs = coord_columns(start, 'start')
  xe, ye, ze = coord_columns(end, 'end')
  if not xs.index.equals(xe.index):
    raise ValueError('start and end must have the same index')

  inside = involume(start, volume) | involume(end, volume)

  # Slab method: overlap of [t0, t1] with [0, 1] means segment hits the open box.
  t0 = np.zeros(len(xs), dtype=float)
  t1 = np.ones(len(xs), dtype=float)
  for s, e, b in ((xs, xe, xb), (ys, ye, yb), (zs, ze, zb)):
    lo, hi = b[0], b[1]
    d = e - s
    parallel = np.isclose(d, 0)
    in_open = (lo < s) & (s < hi)
    with np.errstate(divide='ignore', invalid='ignore'):
      t_lo = (lo - s) / d
      t_hi = (hi - s) / d
    t_near = np.minimum(t_lo, t_hi)
    t_far = np.maximum(t_lo, t_hi)
    t_near = np.where(parallel, np.where(in_open, 0.0, 1.0), t_near)
    t_far = np.where(parallel, np.where(in_open, 1.0, 0.0), t_far)
    t0 = np.maximum(t0, t_near)
    t1 = np.minimum(t1, t_far)

  crosses = (t0 < t1) & (t0 < 1) & (t1 > 0)
  return inside | crosses

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

