import numpy as np
import pandas as pd
from sbnd.general import utils
from sbnd.constants import *
from sbnd.prism import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
from scipy.stats import pearsonr
import copy
from pyanalib import panda_helpers
#from pyanalib import panda_helpers

def get_neutrino_dir(start):
  """
  start is start location is either shower, pfp, or other 
  """
  
  #Get the direction
  neutrino_dir = np.array([start.x - prism_centroid[0],
                          start.y - prism_centroid[1],
                          start.z + distance_from_bnb])
  neutrino_dir /= np.linalg.norm(neutrino_dir,axis=0)
  
  return neutrino_dir.T
def get_theta(pfpdir,nudir):
  """
  Return angle wrt beam modified by reconstucted vertex
  """
  if (pfpdir.shape != nudir.shape):
    print('neutrino and pfp direction are not the same shape')
  
  #Normalize
  pfpdir /= np.linalg.norm(pfpdir,axis=1).reshape((len(pfpdir),1))
  nudir /= np.linalg.norm(nudir,axis=1).reshape((len(nudir),1))
  
  #Calc theta
  theta = np.zeros(len(nudir))
  
  #Mask nan
  mask_nan = np.isnan(pfpdir).any(axis=1) | np.isnan(nudir).any(axis=1)
  pfpdir_no_nan = pfpdir[~mask_nan]
  nudir_no_nan = nudir[~mask_nan]
  
  #Fill theta values
  theta[~mask_nan] = np.arccos(utils.numba_dot(pfpdir_no_nan, nudir_no_nan))
  theta[mask_nan] = np.full(len(mask_nan[mask_nan]),np.nan)
  return theta
def get_err(reco,true,normalize=True):
  """
  Get error between two series
  """
  if len(reco.shape) != len(true.shape):
    print('reco and true are different shapes')
  if len(reco.shape) == 2:
    err = np.linalg.norm(reco-true,axis=1)
    if normalize: err /= np.linalg.norm(true,axis=1)
  elif len(reco.shape) == 1:
    err = reco-true
    if normalize: err /= true
  return err

def get_row_vals(series,indices=None,mode='sum'):
  """
  returns a series of length indices where the series is the series to act on
  """
  if indices is None:
    indices = series.index.drop_duplicates()
  series_subset = series.loc[indices]
  if mode == 'sum':
    return series_subset.groupby(series_subset.index).sum()
  elif mode == 'min':
    return series_subset.groupby(series_subset.index).min()
  elif mode == 'mean':
    return series_subset.groupby(series_subset.index).mean()
  else: #assume mode is a key now - doesn't work
    max_idx = series_subset.groupby(series_subset.index)[mode].idxmax()
    return series_subset.loc[max_idx,key]
    

def get_r2(reco,true):
  """
  Get r2 between true and reco
  """
  if len(reco.shape) != len(true.shape):
    print('reco and true are different shapes')
  #2d not supported yet
  if len(reco.shape) == 2:
    return np.full(reco.shape,np.nan)
  elif len(reco.shape) == 1:
    corr_matrix = np.corrcoef(true,reco)
    corr = corr_matrix[0,1]
    R_sq = corr**2
  pass

def count_true_tracks_showers(pdgs):
  """
  Return true tracks and showers as two integers
  """
  nshw = 0
  ntrk = 0
  #if isinstance(pdgs,np.int32) or isinstance(pdgs,np.int64):
  #  return ntrk,nshw
  counts_dict = pdgs.value_counts().to_dict()
  for pdg,count in counts_dict.items():
    if abs(pdg) in [11,22]: nshw += count 
    if pdg in [111]: nshw += 2*count 
    if abs(pdg) in [13,2212,211]: ntrk += count
  return ntrk,nshw

def get_nonna(x,y):
  """
  REturn non na in common for x and y series
  """
  nainds = x.isna() | y.isna()
  return x[~nainds],y[~nainds]
  
def merge_objs(objs,keys,clas):
  """
  Merge two dataframe like objects and convert to clas
  """  
  objs_copy = [copy.deepcopy(obj) for obj in objs]
  #modify keys
  for i,obj in enumerate(objs_copy):
    cols = pd.MultiIndex.from_tuples([(keys[i],) + key for key in obj.columns])
    obj.columns = cols
    objs_copy[i] = obj #update
  return clas(pd.concat(objs_copy,axis=0))

def check_reference(df,df_comp):
  """
  Checks that the dataframe can refer to the other one
  
  df: dataframe with indices we wish to reference
  df_comp: dataframe we are referencing
  """
  #Index depth
  df_ind_depth = len(df.index.values[0])
  df_comp_ind_depth = len(df_comp.index.values[0])
  
  #Assert that the depth of the comparison dataframe matches the number of matched keys
  assert np.sum([k in df_comp.index.names for k in df.index.names]) == df_comp_ind_depth, "Depth doesn't match the number of matched keys"
  assert df_ind_depth >= df_comp_ind_depth, "Depth of compared dataframe is less than that of dataframe, this is not supported"
  
  return True
  

def split_df_into_bins(df,df_comp,bins,key,low_to_high=True):
  """
  Split an dataframe into bins, need df_comp to parse the bins
  Make sure df_comp.loc[:,key] matches units of bins
  
  df: dataframe to split
  df_comp: dataframe to compare to, must have key with binning of df_comp
  bins: bins to split into
  key: key to split on
  low_to_high: if true, bins are aranged from low to high, otherwise high to low
  
  """
  if not check_reference(df,df_comp): return None #check that the dataframe can refer to the other one
  #Index depth
  df_ind_depth = len(df.index.values[0])
  df_comp_ind_depth = len(df_comp.index.values[0])
  
  #Get key into tuple format
  key = panda_helpers.getcolumns([key],depth=len(df_comp.keys()[0]))[0]
  df_inds_set = set(df.index.values)
  dfs_list = [None]*(len(bins)-1)
  for i,b in enumerate(bins):
    if b == bins[-1]: break #skip last bin to avoid range errors
    #Get indices that are within prism bins
    if low_to_high:
      df_comp_inds_inrange_set = set(df_comp[(df_comp.loc[:,key] <= bins[i+1]) & (df_comp.loc[:,key] > bins[i])].index.values)
    else:
      df_comp_inds_inrange_set = set(df_comp[(df_comp.loc[:,key] >= bins[i+1]) & (df_comp.loc[:,key] < bins[i])].index.values)
    #Get the dataframe's indices
    if df_ind_depth > df_comp_ind_depth:
      df_inds_inrange = utils.get_inds_from_sub_inds(df_inds_set,df_comp_inds_inrange_set,df_comp_ind_depth)
    elif df_ind_depth == df_comp_ind_depth:
      df_inds_inrange = df_comp_inds_inrange_set #One to one correspondence
    dfs_list[i] = df.loc[df_inds_inrange]
  return dfs_list

def get_df_from_bins(df,df_comp,bins,key,assign_key='binning',low_to_high=True):
  """
  Find an dataframe's bins and assign new key to store these bins
  Need df_comp to parse the bins
  Make sure df_comp.loc[:,key] matches units of bins
  
  df: dataframe to split
  df_comp: dataframe to compare to, must have key with binning of df_comp
  bins: bins to split into
  key: key to split on
  assign_key: key to assign bin id to
  low_to_high: if true, bins are aranged from low to high, otherwise high to low
  
  """
  if not check_reference(df,df_comp): return None #check that the dataframe can refer to the other one
  #Index depth
  df_ind_depth = len(df.index.values[0])
  df_comp_ind_depth = len(df_comp.index.values[0])
  
  #Get key into tuple format
  key = panda_helpers.getcolumns([key],depth=len(df_comp.keys()[0]))[0]
  assign_key = panda_helpers.getcolumns([assign_key],depth=len(df.keys()[0]))[0]
  
  #Reference indices
  df_inds_set = set(df.index.values)
  df_comp_inds = df_comp.index.values
  
  #Set dummy value to -1
  df.loc[:,assign_key] = np.full(len(df),-1)
  
  for i,b in enumerate(bins):
    if b == bins[-1]: break #skip last bin to avoid range errors
    #Get indices that are within prism bins
    if low_to_high:
      df_comp_inds_inrange_set = set(df_comp[(df_comp.loc[df_comp_inds,key] <= bins[i+1])\
        & (df_comp.loc[df_comp_inds,key] > bins[i])].index.values)
    else:
      df_comp_inds_inrange_set = set(df_comp[(df_comp.loc[df_comp_inds,key] >= bins[i+1])\
        & (df_comp.loc[df_comp_inds,key] < bins[i])].index.values)
    #Get the dataframe's indices
    if df_ind_depth > df_comp_ind_depth:
      df_inds_inrange = utils.get_inds_from_sub_inds(df_inds_set,df_comp_inds_inrange_set,df_comp_ind_depth)
    elif df_ind_depth == df_comp_ind_depth:
      df_inds_inrange = df_comp_inds_inrange_set #One to one correspondence
    array_size = len(df.loc[df_inds_inrange])
    df.loc[df_inds_inrange,assign_key] = np.full(array_size,i)
    print(i,array_size,len(df_comp_inds))
    
    #Clean comp inds, we don't need the ones that have been assigned already
    df_comp_inds = list(subtract_sets(set(df_comp_inds),df_comp_inds_inrange_set))
    df_comp = df_comp.loc[df_comp_inds]
  return df

def subtract_sets(set1, set2):
  return set1.difference(set2)
  
  
  

  
  
