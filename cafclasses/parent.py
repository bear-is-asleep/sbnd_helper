from pandas import DataFrame
import numpy as np
from pyanalib import panda_helpers
from sbnd.volume import *
from sbnd.constants import *
from sbnd.cafclasses import object_calc
from sbnd.general import utils

class CAF:
    def __init__(self,data,pot=None,**kwargs):
      if isinstance(data, pd.Series):
          self.data = pd.Series(data, **kwargs)
      elif isinstance(data, pd.DataFrame):
          self.data = pd.DataFrame(data, **kwargs)
      else:
          raise ValueError("Invalid data type for CAF: " + str(type(data)))
      if not self.check_key_structure():
        raise ValueError("key structure not correct")
      self.pot = pot #POT to make sample
      self.key_depth = self.key_length() #depth of keys
      self.check_for_duplicates() #assert there are no indexing duplicates 
      self.clean() #set dummy values to nan
      self.data.sort_index(inplace=True)
    def key_length(self):
      return len(self.data.keys()[0])
    def check_key_structure(self):
      #Keys should each be a tuple of the same size
      length = self.key_length()
      for i,key in enumerate(self.data.keys()[1:]):
        if len(key) != length: return False
      return True
    def check_for_duplicates(self):
      assert not self.data.index.duplicated().any(), "Duplicate indices found"
    def copy(self,deep=True):
      data = super().copy(deep)
      return CAF(data)
    def __getitem__(self, item):
        data = super().__getitem__(item) #Series or dataframe get item
        return CAF(data)
    def add_key(self,keys,fill=np.nan):
      """
      Add key to dataframe
      """
      updated_df = panda_helpers.multicol_addkey(self.data, keys,fill=fill,inplace=False)
      # Update the current dfect with the new DataFrame
      for col in updated_df.columns.difference(self.data.columns):
          self.data[col] = updated_df[col]
      return self
    def postprocess(self):
      """
      Run all post processing
      """
      pass
    def clean(self,dummy_vals=[-9999,-999,999,9999],fill=np.nan):
      """
      set all dummy vals to nan
      """
      for i,val in enumerate(dummy_vals):
        self.data.replace(val,fill,inplace=True)
    def check_key(self,key):
      """
      check if a key is in the dataframe
      """
      col_key = panda_helpers.getcolumns([key],depth=self.key_depth) #Only provide one key
      if col_key[0] in self.columns:
        return True
      return False
    def split(self,bins,key,df_comp=None,low_to_high=True):
      """
      Split self into list of self's split by bins determined by df_comp
      """
      if df_comp is None: df_comp = self.copy()
      return object_calc.split_df_into_bins(self.data,df_comp.data,bins,key,low_to_high=low_to_high)
    def get_reference_df(self,ref):
      """
      Use index of self to reference another object
      """
      if not object_calc.check_reference(self.data,ref.data): return None #check that the dfect can refer to the other one
      ref_inds = utils.get_inds_from_sub_inds(set(self.data.index.values),set(ref.data.index.values),len(ref.data.index.values[0]))
      return ref.data.loc[ref_inds]
    def assign_bins(self,bins,key,df_comp=None,assign_key=None,low_to_high=True):
      """
      Assign bins in dataframe, either based on self or df_comp
      """
      if df_comp is None: df_comp = self.data
      self.data = object_calc.get_df_from_bins(self.data,df_comp,bins,key,assign_key=assign_key,low_to_high=low_to_high)
    def add_costheta(self,convert_to_rad=True):
      """
      add costheta
      """
      keys = [
        'costheta'
      ]
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      if convert_to_rad:
        self.data.loc[:,cols[0]] = np.cos(self.data.theta/180*np.pi)*np.sign(self.data.theta)
      else:
        self.data.loc[:,cols[0]] = np.cos(self.data.theta)*np.sign(self.data.theta)
      
      
      
      
      
        
      
      

