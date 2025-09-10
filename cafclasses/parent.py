from pandas import DataFrame
import numpy as np
from pyanalib import panda_helpers
from sbnd.detector.volume import *
from sbnd.constants import *
from sbnd.cafclasses import object_calc
from sbnd.general import utils

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
      self.index_depth = len(self.data.index.values[0]) #depth of indices
      self.index_names = self.data.index.names #names of indices
      self.check_for_duplicates() #assert there are no indexing duplicates 
      self.clean() #set dummy values to nan
      self.data.sort_index(inplace=True)
    def copy(self,deep=True):
      return CAF(self.data.copy(deep),pot=self.pot)
    def combine(self,other,duplicate_ok=False):
      """
      Combine two CAFs
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
              new_levels[0] = new_levels[0] + int(1e10)
              other.data.index = pd.MultiIndex(levels=new_levels, codes=new_codes, names=other.data.index.names)
          else:
              other.data.index = other.data.index + int(1e10)
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
    #-------------------- setters --------------------#
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
      updated_df = panda_helpers.multicol_addkey(self.data, keys,fill=fill,inplace=False)
      # Update the current df with the new DataFrame
      new_cols = updated_df.columns.difference(self.data.columns)
      cols_to_add = {col: updated_df[col] for col in new_cols}
      self.data = pd.concat([self.data, pd.DataFrame(cols_to_add)], axis=1)
    def add_cols(self,keys,values,conditions=None,fill=np.nan,pad_cols=True):
      """
      Generalized method to add a column based on conditions and corresponding values.
      :param keys: Names of the new columns to add.
      :param conditions: A list of boolean conditions for each value.
      :param values: A list of values corresponding to each condition.
      """

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
      elif len(np.shape(conditions)) == 1:
        conditions = [conditions] #Make sure it's a list of lists
      #Convert
      self.add_key(keys, fill=fill)
      cols = panda_helpers.getcolumns(keys, depth=self.key_length())
      #Pad cols
      if len(cols) == 1 and len(cols) < len(values):
        if pad_cols:
          cols = [cols] * len(values)
        else: 
          raise ValueError(f'cols ({cols}) and values ({values}) must be the same length')
      elif len(cols) != len(values):
        raise NotImplementedError(f'cols ({len(cols)}) and values ({len(values)}) must be the same length')
      #print('add_cols keys: ',keys)
      #print('add_cols values: ',values)
      #print('add_cols conditions: ',conditions)
      for col, condition, value in zip(cols, conditions, values):
          try:
            self.data.loc[condition, col] = value
          except:
            print(f'col: {col}')
            print(f'condition: {condition}')
            print(f'value: {value}')
            print(f'self.data.columns: {self.data.columns}')
            raise
            
    def assign_bins(self,bins,key,df_comp=None,assign_key=None,low_to_high=True,mask=None):
      """
      Assign bins in dataframe, either based on self or df_comp
      """
      if df_comp is None: df_comp = self.data
      if mask is not None: df_comp = df_comp[mask]
      self.data = object_calc.get_df_from_bins(self.data,df_comp,bins,key,assign_key=assign_key,low_to_high=low_to_high)
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
        cols = panda_helpers.getcolumns(keys,depth=self.key_length())
        self.data.loc[:,cols[0]] = np.ones(len(self.data)) #initialize to ones
      print(f'--scaling to POT: {sample_pot:.2e} -> {nom_pot:.2e}')
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
        cols = panda_helpers.getcolumns(keys,depth=self.key_length())
        self.data.loc[:,cols[0]] = np.ones(len(self.data)) #initialize to ones
      print(f'--scaling to livetime: {sample_livetime:.2e} -> {nom_livetime:.2e}')
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
      return panda_helpers.getcolumns([key],depth=self.key_depth) #Only provide one key
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
        
      
      
      
    
      
    
  
  
      
      
      
      
      
        
      
      

