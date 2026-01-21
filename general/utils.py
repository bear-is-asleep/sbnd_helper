"""
General helper functions
"""

import numpy as np
import shutil
from pathlib import Path
import os
import numba as nb
from itertools import chain
import pandas as pd
from contextlib import contextmanager
import tempfile
import subprocess
import h5py
from .h5filemanager import H5FileManager

def get_nperbin(series,bins,weights=None):
  """
  Count the number of events in each bin given a list of series and weights.
  """
  assert len(series) == len(weights), f'Length of series and weights must be the same, {len(series)} != {len(weights)}'
  nperbin = []
  for i,s in enumerate(series):
    nperbin.append(np.histogram(s,bins=bins,weights=weights[i])[0])
  return nperbin

def h5py_file_xrootd(path, mode='r', **kwargs):
    """Wrapper for h5py.File that handles XRootD URLs - NO COPYING!"""
    return H5FileManager(path, mode=mode, **kwargs)

def read_hdf_xrootd(path, key=None, **kwargs):
    """Wrapper for pd.read_hdf that handles XRootD URLs."""
    with xrootd_file(path) as local_path:
        if key is not None:
            return pd.read_hdf(local_path, key=key, **kwargs)
        else:
            return pd.read_hdf(local_path, **kwargs)

@contextmanager
def xrootd_file(xrootd_path, cleanup=True, verbose=False):
    """
    Context manager to handle XRootD URLs for pandas read_hdf.
    """
    from .h5filemanager import xrootd_file as _xrootd_file
    with _xrootd_file(xrootd_path, cleanup=cleanup, verbose=verbose) as path:
        yield path

def get_sys_keys(pattern,keys):
  sys_keys = []
  for k in keys:
    for t in k: #tuple
      if pattern in t:
        sys_keys.append(k)
  return sys_keys

def convert_pnfs_to_xroot(path):
    if path.startswith("/pnfs"):
        return path.replace("/pnfs", "root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr")
    return path

def get_weights_from_sys_keys(sys_keys,data):
  """
  Get the weights from the systematic keys
  """
  weights = []
  for key in sys_keys:
    weights.append(data.truth[key].values)
  return np.array(weights).T


def format_number_with_suffix(num,assert_greater_than_1=False):
    """
    Format a number with appropriate suffix (k, M, B, etc.) and 3 significant figures.
    
    Parameters
    ----------
    num : float or int
        Number to format
        
    Returns
    -------
    str
        Formatted string with suffix
    """
    if assert_greater_than_1 and num < 1:
        raise ValueError(f"Number {num} is not greater than 1")
    if num == 0:
        return "0.00"
    
    # Handle negative numbers
    sign = "-" if num < 0 else ""
    num = abs(num)
    
    # Define suffixes and their corresponding powers of 1000
    suffixes = [
        (1e12, 'T'),  # Trillion
        (1e9, 'B'),   # Billion
        (1e6, 'M'),   # Million
        (1e3, 'k'),   # Thousand
        (1, '')       # No suffix
    ]
    
    # Find the appropriate suffix
    for threshold, suffix in suffixes:
        if num >= threshold:
            # Scale the number
            scaled_num = num / threshold
            
            # Format to 3 significant figures
            if scaled_num >= 100:
                # 3 digits before decimal: 123.4 -> 123
                formatted = f"{scaled_num:.0f}"
            elif scaled_num >= 10:
                # 2 digits before decimal: 12.34 -> 12.3
                formatted = f"{scaled_num:.1f}"
            else:
                # 1 digit before decimal: 1.234 -> 1.23
                formatted = f"{scaled_num:.2f}"
            
            return f"{sign}{formatted}{suffix}"
    
    # Fallback for very small numbers
    return f"{sign}{num:.2e}"

def calc_ke_from_momentum(momentum,mass):
  """
  Calculate kinetic energy from momentum and mass
  """
  return np.sqrt(momentum**2+mass**2) - mass

def calc_momentum_from_ke(ke,mass):
  """
  Calculate momentum from kinetic energy and mass
  """
  return np.sqrt((ke+mass)**2-mass**2)

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

def flatten_list(l):
  """
  Flatten a list of lists into a list.
  """
  ret_l = []
  for i,sublist in enumerate(l):
    ret_l.extend(sublist)
    
  return ret_l
  #return [item for sublist in l for item in sublist]

def sqrt_sum_of_squares(numbers):
  """
  Takes a list of numbers and returns the sum of their squares.
  """
  return np.sqrt(np.sum([x**2 for x in numbers]))

def find_indices_in_common(list1,list2):
  """
  Return indices in common between list1 and list2
  """
  set1 = set(list1)
  set2 = set(list2)
  return list(set1.intersection(set2))

def find_indices_not_in_common(list1,list2):
  """
  Find list of indices not in common between two lists.
  Return the values not in list 1
  """
  set1 = set(list1)
  set2 = set(list2)
  return list(set2.difference(set1))

def common_indices(*lists):
  """
  Return a list of common values from a list of lists.
  """
  common = []
  for elem in lists[0]:
    if all(elem in lst for lst in lists[1:]):
      common.append(elem)
  return common

@nb.jit(nopython=True)
def numba_dot(a, b):
  result = np.zeros(len(a))
  for i in range(len(a)):
    result[i] = np.dot(a[i],b[i])
  return result

def get_inds_from_sub_inds(inds,sub_inds,length):
  """
  Get the indices of inds that are in sub_inds
  
  inds: set of indices (bigger)
  sub_inds: set of indices to check (smaller)
  """
  inds = set(inds)
  sub_inds = set(sub_inds)
  matched_inds = set()
  
  for _,ind in enumerate(inds):
      if ind[:length] in sub_inds:
          matched_inds.add(ind)
  return list(matched_inds)

def get_sub_inds_from_inds(inds,sub_inds,length):
  """
  Get the indices of sub_inds that are in inds
  
  inds: set of indices (smaller)
  sub_inds: set of indices to check (bigger)
  """
  sub_inds = set(sub_inds)
  inds = set(inds)
  matched_inds = set()
  
  for _,ind in enumerate(inds):
      if ind[:length] in sub_inds:
          matched_inds.add(ind[:length])
  return list(matched_inds)

def flatten_list(l):
  """
  Flatten a list of lists into a list.
  
  l: list of lists
  """
  return list(chain.from_iterable(l))

def join_dataframes(list1, list2, wrapper=None):
  """
  Join two dataframes with common indices
  """
  matrix = []
  for i in range(len(list1)):
      row = []
      for j in range(len(list2)):
          # join dataframes with common indices
          list1_inds = list1[i].index.values
          list2_inds = list2[j].index.values
          common_inds = find_indices_in_common(list1_inds,list2_inds)
          df = list1[i].loc[common_inds]
          if wrapper is not None:
              df = wrapper(df)
          row.append(df)
      matrix.append(row)
  return matrix

def join_three_dataframes(list1, list2, list3, wrapper=None):
  """
  Join three dataframes with common indices
  """
  matrix = []
  for i in range(len(list1)):
      row = []
      for j in range(len(list2)):
          col = []
          for k in range(len(list3)):
              # join dataframes with common indices
              list1_inds = list1[i].index.values
              list2_inds = list2[j].index.values
              common_inds = find_indices_in_common(list1_inds,list2_inds)
              list3_inds = list3[k].index.values
              common_inds = find_indices_in_common(common_inds,list3_inds)
              df = list1[i].loc[common_inds]
              if wrapper is not None:
                df = wrapper(df)
              col.append(df)
          row.append(row)
      matrix.append(row)
  return matrix

def sort_by_other_list(main_list, other_list):
    sorted_tuples = sorted(zip(other_list, main_list))
    sorted_list = [element for _, element in sorted_tuples]
    return sorted_list
  
def calculate_fwhm_from_histogram(x, **kwargs):
    # Create the histogram
    counts, bin_edges = np.histogram(x, **kwargs)
    
    # Find the peak of the histogram
    peak_index = np.argmax(counts)
    peak_value = counts[peak_index]
    
    # Calculate the half maximum value
    half_max = peak_value / 2.0
    
    # Interpolate to find the points where the histogram crosses the half maximum
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    above_half_max_indices = np.where(counts >= half_max)[0]
    
    # Ensure there are at least two points
    if len(above_half_max_indices) < 2:
        raise ValueError("Not enough points above half maximum to calculate FWHM")
    
    # Get the first and last bin center where the counts are above the half maximum
    first_index = above_half_max_indices[0]
    last_index = above_half_max_indices[-1]
    
    # Calculate the FWHM
    fwhm = bin_centers[last_index] - bin_centers[first_index]
    
    return fwhm

import numpy as np

def extract_parameter_name(key,variables):
    """
    Extract the essential parameter name from a systematic key.
    
    Parameters
    ----------
    key : str
        The full systematic key
    variables : list
        The variable names we are evaluating systematic uncertainties for
        
    Returns
    -------
    key : str
        The essential parameter name
    var_name : str
        The variable name we are evaluating systematic uncertainties for
    """
    #If it's a csv name, remove the .csv. We also expect the name to be in the format of {name}_{variable}_cv.csv
    if '.csv' in key:
      key = key.replace('.csv', '')
      keep = key.split('_')[:-1]
      key = '_'.join(keep)

    for var in variables:
      if var in key:
        var_name = var
        break
    
    #First check if it's a stat error
    if 'stat' in key:
      return 'stat',var_name
    
    # Remove the trailing ";1" first
    key = key.replace(';1', '')
    key = key.replace('reco_leading_muon_', '')
    key = key.replace('true_leading_muon_', '')
    key = key.replace('momentum_gev', '')
    key = key.replace('costheta', '')
    key = key.replace('multisigma_', '')
    key = key.replace('multisim_', '')
    key = key.replace('nsigma_', '')
    key = key.replace('GENIEReWeight_SBN_v1_', '')
    #Remove trailing _
    while key[-1] == '_':
      key = key[:-1]
    return key, var_name