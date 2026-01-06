from pandas import DataFrame
import numpy as np
import pandas as pd
import gc
from pyanalib import pandas_helpers
from sbnd.detector.volume import *
from sbnd.constants import *
from sbnd.cafclasses import object_calc
from sbnd.general import utils

def filter_univ_columns(df):
    """
    Filter out columns that contain 'univ' in any level of the MultiIndex column name.
    This is a memory optimization since universe weights take up ~95% of the file size.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potentially MultiIndex columns
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with 'univ' columns removed
    """
    if df.empty:
        return df
    
    # Check if columns are MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # Filter columns where no level contains 'univ'
        # Check each column tuple - if any non-empty level contains 'univ', exclude it
        mask = []
        for col in df.columns:
            has_univ = any('univ' in str(level).lower() for level in col if level)\
              or any('ps1' in str(level).lower() for level in col if level)\
              or any('ps2' in str(level).lower() for level in col if level)\
              or any('ps3' in str(level).lower() for level in col if level)\
              or any('ms1' in str(level).lower() for level in col if level)\
              or any('ms2' in str(level).lower() for level in col if level)\
              or any('ms3' in str(level).lower() for level in col if level)\
              or any('cv' in str(level).lower() for level in col if level)\
              or any('morph' in str(level).lower() for level in col if level)
            mask.append(not has_univ)
        filtered_df = df.loc[:, mask]
        return filtered_df
    else:
        # Regular Index - check if column name contains 'univ'
        mask = [not ('univ' in str(col).lower()) for col in df.columns]
        return df.loc[:, mask]

def load_univ_columns(fname=None, key=None, df_index=None, store=None):
    """
    Load only universe weight columns from an HDF5 file.
    This allows you to add universe weights back after filtering them out for memory efficiency.
    
    Parameters
    ----------
    fname : str
        Path to HDF5 file
    key : str
        HDF5 key to read from
    df_index : pd.Index, optional
        Index of the dataframe to align the universe weights with.
        If None, returns universe weights with original index from the file.
        
    Returns
    -------
    univ_df : pd.DataFrame
        DataFrame containing only universe weight columns, optionally aligned with df_index
    """
    assert key is not None, "HDF5 key must be provided"
    if store is None:
        assert fname is not None, "Either fname or store must be provided"
        full_df = pd.read_hdf(fname, key=key)
    else:
        full_df = store.get(key)
    
    # Filter to only universe columns
    if isinstance(full_df.columns, pd.MultiIndex):
        mask = []
        for col in full_df.columns:
            has_univ = any('univ' in str(level).lower() for level in col if level)
            mask.append(has_univ)
        univ_df = full_df.loc[:, mask]
    else:
        mask = ['univ' in str(col).lower() for col in full_df.columns]
        univ_df = full_df.loc[:, mask]
    
    # Align with the target index if provided (only keep rows that exist in both)
    if df_index is not None:
        univ_df = univ_df.reindex(df_index)
    
    return univ_df

class CAF:
    #-------------------- constructor/rep --------------------#
    def __init__(self,data,prism_bins=None,pot=None,livetime=None,**kwargs):
      if isinstance(data, pd.Series):
          self.data = pd.Series(data, **kwargs)
      elif isinstance(data, pd.DataFrame):
          self.data = pd.DataFrame(data, **kwargs)
      else:
          raise ValueError("Invalid data type for CAF: " + str(type(data)))
      if not self.check_key_structure():
        raise ValueError("key structure not correct")
      self.set_prism_bins(prism_bins)
      self.pot = pot #POT to make sample
      self.livetime = livetime #livetime to make sample
      self.key_depth = self.key_length() #depth of keys
      if len(self.data.index.values) == 0:
        print(f'WARNING: No data for CAF')
        return None
      self.index_depth = len(self.data.index.values[0]) #depth of indices
      self.index_names = self.data.index.names #names of indices
      self.check_for_duplicates() #assert there are no indexing duplicates 
      self.clean() #set dummy values to nan
      self.data.sort_index(inplace=True)
    def copy(self,deep=True):
      return CAF(self.data.copy(deep),pot=self.pot)
    def combine(self,other,duplicate_ok=False,offset=int(1e5)):
      """
      Combine two CAFs

      Parameters
      ----------
      other : CAF
        The other CAF to combine with.
      duplicate_ok : bool
        If True, handle duplicate indices the same way as combine() does (add offset).
      offset : int
        The offset to add to the indices of the other CAF.

      Returns
      -------
      self : CAF
        Returns self for method chaining
      """
      if self.pot is None:
        pot = other.pot
      elif other.pot is None:
        pot = self.pot
      else:
        pot = self.pot + other.pot
      if self.livetime is None:
        livetime = other.livetime
      elif other.livetime is None:
        livetime = self.livetime
      else:
        livetime = self.livetime + other.livetime
      #Check if there are any indices in common
      if len(set(self.data.index.values) & set(other.data.index.values)) > 0:
        if not duplicate_ok:
          raise ValueError('Duplicate indices found in combined CAFs')
        else:
          # Add big number to the other indices - handle MultiIndex
          if isinstance(other.data.index, pd.MultiIndex):
              # For MultiIndex, we need to modify the first level
              new_levels = list(other.data.index.levels)
              new_codes = list(other.data.index.codes)
              # Add offset to the first level
              new_levels[-1] = new_levels[-1] + offset
              other.data.index = pd.MultiIndex(levels=new_levels, codes=new_codes, names=other.data.index.names)
          else:
              other.data.index = other.data.index + offset
      self.data = pd.concat([self.data,other.data],axis=0)
      self.pot = pot
      self.livetime = livetime
      return self
    def __getitem__(self, item):
        data = super().__getitem__(item) #Series or dataframe get item
        return CAF(data,pot=self.pot)
    def print_funcs(self):
      """
      Print all functions
      """
      print('--- adders ---')
      for key in self.__dir__():
        if key.startswith('add_'):
          print(key)
      print('--- cutters ---')
      for key in self.__dir__():
        if key.startswith('cut_'):
          print(key)
      print('--- assigners ---')
      for key in self.__dir__():
        if key.startswith('assign_'):
          print(key)
      print('--- getters ---')
      for key in self.__dir__():
        if key.startswith('get_'):
          print(key)
      print('--- setters ---')
      for key in self.__dir__():
        if key.startswith('set_'):
          print(key)
      print('--- cleaners ---')
      for key in self.__dir__():
        if key.startswith('clean_'):
          print(key)
      print('--- checkers ---')
      for key in self.__dir__():
        if key.startswith('check_'):
          print(key)
      print('--- fixers ---')
      for key in self.__dir__():
        if key.startswith('fix_'):
          print(key)
    #Get rid of this?
    def keys(self):
      return self.data.keys()
    def load(fname,key='slice',**kwargs):
      df = pd.read_hdf(fname,key=key,**kwargs)
      return CAF(df,**kwargs)
    #-------------------- cutters --------------------#
    def apply_cut(self, cut_name, condition=None, cut=True):
      """
      Apply a cut based on a specified condition.

      Parameters
      ----------
      cut_name : str
          The name of the cut to apply.
      condition : array_like
          Boolean array where True indicates the row passes the cut.
      cut : bool, optional
          Whether to actually apply the cut to the data. Default is True.
      """
      if 'cut.' not in cut_name: #Allow us to be lazy
          cut_name = 'cut.'+cut_name
      if cut and self.check_key(cut_name):
          orig_size = len(self.data)
          cut_col = self.get_key(cut_name)
          self.data = self.data[self.data[cut_col].values]
          new_size = len(self.data)
          print(f'Applied cut on key: {cut_name} ({orig_size:,} --> {new_size:,})')
          return
      elif condition is None:
          raise Exception(f'Attempting to cut on key not in data: {cut_name}')

      self.add_key([cut_name],fill=False)
      #print(f'added key: {cut_name}')
      col = pandas_helpers.getcolumns([cut_name], depth=self.key_length())[0]
      self.data.loc[:, col] = condition

      if cut:
          orig_size = len(self.data)
          self.data = self.data[self.data[col]]
          new_size = len(self.data)
          print(f'Applied cut on key: {cut_name} ({orig_size:,} --> {new_size:,})')
          return
    #-------------------- setters --------------------#
    def add_universe_weights(self, fname, keys, duplicate_ok=False, keep_patterns=None):
        """
        Load universe weight columns from HDF5 file(s) and add them back to the dataframe.
        This is useful when universe weights were filtered out during loading for memory efficiency.
        Handles multiple keys by combining them similar to how data was originally combined.
        
        Parameters
        ----------
        fname : str
            Path to HDF5 file containing universe weights
        keys : str or list
            HDF5 key(s) to read from. If list, will combine universe weights from all keys.
        duplicate_ok : bool
            If True, handle duplicate indices the same way as combine() does (add offset).
            Should match the duplicate_ok parameter used when combining the original data.
            
        Returns
        -------
        self : CAF
            Returns self for method chaining
        """
        if isinstance(keys, str):
            keys = [keys]

        if keep_patterns is not None and not isinstance(keep_patterns, (list, tuple, set)):
            raise TypeError('keep_patterns must be a list, tuple, or set of substrings to keep')

        combined_univ = None
        target_index = self.data.index

        with pd.HDFStore(fname, mode='r') as store:
            for i, key in enumerate(keys):
                if key not in store:
                    print(f'  WARNING: Key {key} not found in {fname}, skipping')
                    continue

                univ_df = load_univ_columns(fname=fname, key=key, store=store)
                if univ_df.empty:
                    continue

                if keep_patterns:
                    if isinstance(univ_df.columns, pd.MultiIndex):
                        mask = []
                        for col in univ_df.columns:
                            col_str = '.'.join(str(level) for level in col if level)
                            mask.append(any(pattern in col_str for pattern in keep_patterns))
                        univ_df = univ_df.loc[:, mask]
                    else:
                        mask = [any(pattern in str(col) for pattern in keep_patterns) for col in univ_df.columns]
                        univ_df = univ_df.loc[:, mask]
                    if univ_df.empty:
                        del univ_df
                        gc.collect()
                        continue

                # Keep only rows that survived cuts in the working dataframe
                if len(target_index) != len(univ_df.index) or not univ_df.index.equals(target_index):
                    relevant_index = univ_df.index.intersection(target_index)
                    print(f'  Keeping {len(relevant_index)}/{len(univ_df.index)} rows from {key}')
                    if len(relevant_index) == 0:
                        del univ_df
                        gc.collect()
                        continue
                    univ_df = univ_df.loc[relevant_index]

                # Handle duplicated indices within the chunk
                if univ_df.index.duplicated().any():
                    if duplicate_ok:
                        univ_df = univ_df.loc[~univ_df.index.duplicated(keep='first')]
                    else:
                        raise ValueError(f'Duplicate indices found within universe weights for key {key}')

                if combined_univ is None:
                    combined_univ = univ_df
                else:
                    overlap = combined_univ.index.intersection(univ_df.index)
                    if len(overlap) > 0 and not duplicate_ok:
                        raise ValueError('Duplicate indices found when combining universe weights from keys. Use duplicate_ok=True if this is expected.')
                    combined_univ = pd.concat([combined_univ, univ_df], axis=0)

                del univ_df
                gc.collect()

        if combined_univ is None:
            return self

        matching_indices = combined_univ.index.intersection(self.data.index)
        total_current = len(self.data.index)
        total_univ = len(combined_univ.index)
        matched = len(matching_indices)

        print(f'  Universe weights: {total_univ:,} total indices, {matched:,} match current dataframe ({total_current:,} rows)')
        if matched < total_current * 0.9:
            print(f'  WARNING: Only {matched/total_current*100:.1f}% of indices match! This may indicate index misalignment.')

        combined_univ = combined_univ.reindex(self.data.index)

        nan_count = combined_univ.isna().any(axis=1).sum()
        if nan_count > 0:
            print(f'  WARNING: {nan_count:,} rows have NaN universe weights after alignment. This may indicate missing indices.')

        self.data = pd.concat([self.data, combined_univ], axis=1)

        del combined_univ
        gc.collect()

        return self
    #Change to setter?
    def key_length(self):
      return len(self.data.keys()[0])
    def set_prism_bins(self,prism_bins):
      """
      Set prism bins
      """
      self.prism_binning = prism_bins
    def set_momentum_bins(self,momentum_bins):
      """
      Set momentum bins
      """
      self.momentum_binning = momentum_bins
    def set_costheta_bins(self,costheta_bins):
      """
      Set costtheta bins
      """
      self.costheta_binning = costheta_bins
    #-------------------- adders --------------------#
    def add_key(self,keys,fill=np.nan):
      """
      Add key to dataframe
      """
      updated_df = pandas_helpers.multicol_addkey(self.data, keys,fill=fill,inplace=False)
      # Update the current df with the new DataFrame
      new_cols = updated_df.columns.difference(self.data.columns)
      cols_to_add = {col: updated_df[col] for col in new_cols}
      self.data = pd.concat([self.data, pd.DataFrame(cols_to_add)], axis=1)
    def add_cols(self,keys,values,conditions=None,fill=np.nan,pad_cols=True):
      """
      Generalized method to add a column based on conditions and corresponding values.

      Parameters
      ----------
      keys: list
        Names of the new columns to add.
      values: list
        A list of values corresponding to each condition.
      conditions: list
        A list of boolean conditions for each value.
      fill: 
        The value to fill the new columns with.
      pad_cols: bool
        Whether to pad the new columns with the fill value.
      """

      # Allow us to be lazy, the values should be in a list
      if not isinstance(values, list):
        print('WARNING: Implicitly converting values to list')
        values = [values]
      if not isinstance(keys, list):
        print('WARNING: Implicitly converting keys to list')
        keys = [keys]

      if len(keys) != len(values):
        raise ValueError(f'keys ({len(keys)}) and values ({len(values)}) must be the same length')

      # Determine the fill type
      fill_type = type(fill)
      if hasattr(fill, 'dtype'):
        fill_type = fill.dtype.type

      #Check datatype compatibility
      for value in values:
        value_type = type(value)
        if hasattr(value, 'dtype'):
          value_type = value.dtype.type

          # Allow for compatible types rather than exact matches
          if not (issubclass(value_type, fill_type) or issubclass(fill_type, value_type)):
            #Handle specific cases I've come across, since it's the easiest way for now
            if (isinstance(fill,float) and value_type == np.float32)\
              or (isinstance(fill,float) and value_type == np.float64)\
              or (isinstance(fill,int) and value_type == np.int32)\
              or (isinstance(fill,str) and value_type == np.str_)\
              or (isinstance(fill,bool) and value_type == np.bool_):
              continue 
            raise TypeError(f'fill type ({fill_type}) is not compatible with value type ({value_type})')
      #Verify conditions are in a list and of the right form
      if conditions is None: 
        conditions = np.full(np.shape(values),True)
      if len(np.shape(conditions)) == 1:
        conditions = [conditions] #Make sure it's a list of lists
      #Convert
      self.add_key(keys, fill=fill)
      cols = pandas_helpers.getcolumns(keys, depth=self.key_length())
      #Pad cols
      if len(cols) == 1 and len(cols) < len(values):
        if pad_cols:
          cols = [cols] * len(values)
        else: 
          raise ValueError(f'cols ({cols}) and values ({values}) must be the same length')
      elif len(cols) != len(values):
        raise NotImplementedError(f'cols ({len(cols)}) and values ({len(values)}) must be the same length')
      # print('add_cols keys: ',keys)
      # print('add_cols values: ',values)
      # print('add_cols conditions: ',conditions)
      for col, condition, value in zip(cols, conditions, values):
          try:
            self.data.loc[condition, col] = value
          except:
            print(f'col: {col}')
            print(f'condition: {condition}')
            print(f'value: {value}')
            print(f'self.data.columns: {self.data.columns}')
            raise
            
    def assign_bins(self,bins,key,df_comp=None,assign_key=None,low_to_high=True,mask=None,replace_nan=-1):
      """
      Assign bins in dataframe, either based on self or df_comp
      When mask is provided, only assigns bins to masked rows, preserving existing values for other rows
      """
      if df_comp is None: 
        df_comp = self.data
      
      if mask is not None: 
        # Filter df_comp to masked rows only
        df_comp_masked = df_comp[mask]
        
        # Get bins only for the masked subset
        data_masked = self.data[mask].copy()
        result_masked = object_calc.get_df_from_bins(data_masked,df_comp_masked,bins,key,assign_key=assign_key,low_to_high=low_to_high,replace_nan=replace_nan)
        
        # Get the converted column name (get_df_from_bins converts assign_key internally)
        assign_key_col = self.get_key(assign_key)[0] if assign_key is not None else 'binning'
        
        # Only assign to the masked rows, preserving existing values for other rows
        self.data.loc[mask, assign_key_col] = result_masked[assign_key_col].values
      else:
        # No mask - assign to entire dataframe
        self.data = object_calc.get_df_from_bins(self.data,df_comp,bins,key,assign_key=assign_key,low_to_high=low_to_high,replace_nan=replace_nan)
    def postprocess(self):
      """
      Run all post processing
      """
      pass
    def scale_to_pot(self,nom_pot,sample_pot=None):
      """
      Scale to nominal protons on target (POT). Need sample POT as input
      """
      #if sample_pot is None: sample_pot = self.pot
      assert sample_pot is not None, 'sample POT is None'
      if sample_pot == nom_pot: print('WARNING: sample POT is equal to nominal POT')
      if not self.check_key('genweight'): #key not in dataframe
        keys = ['genweight']
        self.add_key(keys)
        cols = pandas_helpers.getcolumns(keys,depth=self.key_length())
        self.data.loc[:,cols[0]] = np.ones(len(self.data)) #initialize to ones
      print(f'--scaling to POT ({nom_pot/sample_pot:.2e}): {sample_pot:.2e} -> {nom_pot:.2e}')
      self.data.genweight = self.data.genweight*nom_pot/sample_pot
    def scale_to_livetime(self,nom_livetime,sample_livetime=None):
      """
      Scale to nominal livetime. Need sample livetime as input
      """
      assert sample_livetime is not None, 'sample livetime is None'
      if sample_livetime == nom_livetime: print('WARNING: sample livetime is equal to nominal livetime')
      if not self.check_key('genweight'): #key not in dataframe
        keys = ['genweight']
        self.add_key(keys)
        cols = pandas_helpers.getcolumns(keys,depth=self.key_length())
        self.data.loc[:,cols[0]] = np.ones(len(self.data)) #initialize to ones
      print(f'--scaling to livetime ({nom_livetime/sample_livetime:.2e}): {sample_livetime:.2e} --> {nom_livetime:.2e}')
      self.data.genweight = self.data.genweight*nom_livetime/sample_livetime
      self.livetime = nom_livetime
    def scale_to_prism_coeff(self,prism_coeff):
      """
      First set the prism binnings of events, then scale to the prism coefficient.
      """
      pass
    #-------------------- getters --------------------#
    def get_reference_df(self,ref):
      """
      Use index of self to reference another object
      """
      #I don't think we need this line anymore...
      #if not object_calc.check_reference(self.data,ref.data): return None #check that the object can refer to the other one
      if self.index_depth < ref.index_depth:
        ref_inds = utils.get_inds_from_sub_inds(set(ref.data.index.values),set(self.data.index.values),self.index_depth)
      elif self.index_depth > ref.index_depth:
        ref_inds = utils.get_sub_inds_from_inds(set(self.data.index.values),set(ref.data.index.values),ref.index_depth)
      return ref.data.loc[ref_inds]
    def get_split(self,bins,key,df_comp=None,low_to_high=True):
      """
      Split self into list of self's split by bins determined by df_comp
      """
      if df_comp is None: df_comp = self.copy()
      return object_calc.split_df_into_bins(self.data,df_comp.data,bins,key,low_to_high=low_to_high)
    def get_key(self,key):
      """
      Converts key to tuple format
      """
      if isinstance(key, list):
        return pandas_helpers.getcolumns(key,depth=self.key_depth) #Only provide one key
      else:
        return pandas_helpers.getcolumns([key],depth=self.key_depth) #Only provide one key
    def get_binned_numevents(self,bin_key,binning=None):
      """
      Get number of events per bin of some binning scheme
      """
      if not self.check_key(bin_key):
          raise Exception(f'bin_key: {bin_key} not in keys')
      bin_col = self.get_key(bin_key)
      bins = self.data[bin_col].values.flatten()
      #Set binning
      if binning is None:
          nbins = np.nanmax(bins)
          assert nbins > 0, f'nbins is less than 1 {nbins}'
          binning = np.arange(0,nbins+2,1.) - 0.5 #offset to account for digitize method
      weights = self.data.genweight
      #Get number of events in each bin
      hist = np.histogram(bins,bins=binning,weights=weights)
      return hist[0]
    #-------------------- cleaners --------------------#
    def clean(self,dummy_vals=[-9999,-999,999,9999],fill=np.nan):
      """
      set all dummy vals to nan
      """
      for i,val in enumerate(dummy_vals):
        self.data.replace(val,fill,inplace=True)
    def purge(self,cols):
        """
        Remove all rows with nan in specified columns
        """
        self.data = self.data.dropna(subset=cols,how='all')
    #-------------------- checkers --------------------#
    def check_key(self,key):
      """
      check if a key is in the dataframe
      """
      col_key = self.get_key(key)
      if col_key[0] in self.data.columns:
        return True
      return False
    def check_key_structure(self):
      #Keys should each be a tuple of the same size
      length = self.key_length()
      for i,key in enumerate(self.data.keys()[1:]):
        if len(key) != length: return False
      return True
    def check_for_duplicates(self):
      assert not self.data.index.duplicated().any(), "Duplicate indices found"
        
      
      
      
    
      
    
  
  
      
      
      
      
      
        
      
      

