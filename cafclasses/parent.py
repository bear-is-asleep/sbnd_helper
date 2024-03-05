from pandas import DataFrame
import numpy as np
from pyanalib import panda_helpers
from sbnd.volume import *
from sbnd.constants import *
from sbnd.cafclasses import object_calc
from sbnd.general import utils

class CAF:
    #-------------------- constructor/rep --------------------#
    def __init__(self,data,prism_bins=None,pot=None,**kwargs):
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
      self.key_depth = self.key_length() #depth of keys
      self.index_depth = len(self.data.index.values[0]) #depth of indices
      self.index_names = self.data.index.names #names of indices
      self.check_for_duplicates() #assert there are no indexing duplicates 
      self.clean() #set dummy values to nan
      self.data.sort_index(inplace=True)
    def copy(self,deep=True):
      return CAF(self.data.copy(deep),pot=self.pot)
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
    #-------------------- setters --------------------#
    #Change to setter?
    def key_length(self):
      return len(self.data.keys()[0])
    def set_prism_bins(self,prism_bins):
      """
      Set prism bins
      """
      self.prism_binning = prism_bins
    #-------------------- adders --------------------#
    def add_key(self,keys,fill=np.nan):
      """
      Add key to dataframe
      """
      updated_df = panda_helpers.multicol_addkey(self.data, keys,fill=fill,inplace=False)
      # Update the current df with the new DataFrame
      for col in updated_df.columns.difference(self.data.columns):
          self.data[col] = updated_df[col]
      return self
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
        
      
      
      
    
      
    
  
  
      
      
      
      
      
        
      
      

