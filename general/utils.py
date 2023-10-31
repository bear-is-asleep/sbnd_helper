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
  dummy_ind = (0,0)
  
  for _,ind in enumerate(inds):
      if ind[:length] in sub_inds:
          matched_inds.add(ind)
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
