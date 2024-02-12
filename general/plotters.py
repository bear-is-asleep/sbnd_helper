"""
General plotting functions for SBND
"""
import matplotlib.colors as mcolors
import os
from datetime import date
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

day = date.today().strftime("%Y_%m_%d")

def map_value_to_color(value, min_val, max_val, cmap='viridis',return_hex=True):
    # Create a colormap
    cmap = plt.get_cmap(cmap)  # replace 'viridis' with your colormap
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    normalized_value = norm(value)
    rgb_color = cmap(normalized_value)
    if return_hex:
        return mcolors.rgb2hex(rgb_color[:3])
    return rgb_color
def move_file(fname,folder_name,overwrite=True):
  # Move the file to the folder
  src_file = f'{fname}'
  dst_folder = Path(folder_name)
  
  # Make sure the folder exists
  os.makedirs(folder_name, exist_ok=True)

  # Check if the file exists in the source path
  if os.path.isfile(src_file):

    # Generate the destination path by appending the filename to the destination directory
    dst_file = os.path.join(folder_name, os.path.basename(src_file))

    if overwrite:
      shutil.move(src_file, dst_file)
    else:
      # If the file already exists in the destination directory, make a copy with a numerical suffix
      if os.path.exists(dst_file):
        suffix = 1
        while True:
          new_dst_file = os.path.join(folder_name, os.path.splitext(os.path.basename(src_file))[0] + "_" + str(suffix) + os.path.splitext(os.path.basename(src_file))[1])
          if not os.path.exists(new_dst_file):
            shutil.copy2(src_file, new_dst_file)
            dst_file = new_dst_file
            break
          suffix += 1
      shutil.move(src_file, dst_file)

def save_plot(fname, fig=None, ftype='.png', dpi=300, folder_name=None, overwrite=True):
  if folder_name is None:
    folder_name = f'Plots/Plots_{day}'
  # Save the plot
  if fig == None:
    plt.savefig(f'{fname}{ftype}', bbox_inches="tight", dpi=dpi)
  else:
    fig.savefig(f'{fname}{ftype}', bbox_inches="tight", dpi=dpi)
  move_file(f'{fname}{ftype}',folder_name,overwrite=overwrite)
    
def set_style(ax,legend_size=12,legend_loc='best',axis_size=16,title_size=20,tick_size=16,
              bbox_to_anchor=None):
  #based on (8,6 figure)
  #plt.style.use('science')
  ax.tick_params(labelsize=tick_size)
  ax.xaxis.label.set_size(axis_size)
  ax.yaxis.label.set_size(axis_size)
  ax.title.set_size(title_size)
  if ax.get_legend() is not None:
    if bbox_to_anchor is not None:
      ax.legend(bbox_to_anchor=bbox_to_anchor,fontsize=legend_size)
    else:
      ax.legend(loc=legend_loc,fontsize=legend_size)

def add_label(ax,text,where='topright',**kwargs):
  if where == 'topright':
    ax.text(0.99,0.99,text,
          transform=ax.transAxes,
          horizontalalignment='right',
          verticalalignment='top',**kwargs)
  elif where == 'centerright':
    ax.text(0.99,0.5,text,
          transform=ax.transAxes,
          horizontalalignment='right',
          verticalalignment='center',**kwargs)
  elif where == 'centerleft':
    ax.text(0.01,0.5,text,
          transform=ax.transAxes,
          horizontalalignment='left',
          verticalalignment='center',**kwargs)
  elif where == 'topcenter':
    ax.text(0.5,0.99,text,
          transform=ax.transAxes,
          horizontalalignment='center',
          verticalalignment='top',**kwargs)
  elif where == 'center':
    ax.text(0.5,0.5,text,
          transform=ax.transAxes,
          horizontalalignment='center',
          verticalalignment='center',**kwargs)
  elif where == 'bottomright':
    ax.text(0.99,0,text,
          transform=ax.transAxes,
          horizontalalignment='right',
          verticalalignment='bottom',**kwargs)
  elif where == 'bottomishright':
    ax.text(0.99,0.2,text,
          transform=ax.transAxes,
          horizontalalignment='right',
          verticalalignment='bottom',**kwargs)
  elif where == 'bottomleft':
    ax.text(0.01,0,text,
          transform=ax.transAxes,
          horizontalalignment='left',
          verticalalignment='bottom',**kwargs)
  elif where == 'bottomishleft':
    ax.text(0.01,0.2,text,
          transform=ax.transAxes,
          horizontalalignment='left',
          verticalalignment='bottom',**kwargs)
  elif where == 'topleft':
    ax.text(0.01,0.99,text,
          transform=ax.transAxes,
          horizontalalignment='left',
          verticalalignment='top',**kwargs)
  elif where == 'bottomcenter':
    ax.text(0.5,0,text,
          transform=ax.transAxes,
          horizontalalignment='center',
          verticalalignment='bottom',**kwargs)
  elif where == 'toprightoutside':
    ax.text(0.99,1.15,text,
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top',**kwargs)
  elif where == 'bottomrightoutside':
    ax.text(1.05,0.05,text,
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='bottom',**kwargs)

def get_colors(cmap, n):
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i/n) for i in range(n)]
    return colors
