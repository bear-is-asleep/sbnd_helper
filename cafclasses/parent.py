from pandas import DataFrame
import numpy as np
from pyanalib import panda_helpers
from sbnd.volume import *
from sbnd.constants import *
from sbnd.cafclasses import object_calc
from sbnd.general import utils

class CAF(DataFrame):
    def __init__(self,df):
      super().__init__(df)
      if not self.check_key_structure():
        raise ValueError("key structure not correct")
      self.key_depth = self.key_length()
    def key_length(self):
      return len(self.keys()[0])
    def check_key_structure(self):
      #Keys should each be a tuple of the same size
      length = self.key_length()
      for i,key in enumerate(self.keys()[1:]):
        if len(key) != length: return False
      return True
    def add_key(self,keys,fill=np.nan):
      """
      Add key to dataframe
      """
      updated_df = panda_helpers.multicol_addkey(self, keys,fill=fill,inplace=False)
      # Update the current PFP object with the new DataFrame
      for col in updated_df.columns.difference(self.columns):
          self[col] = updated_df[col]
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
        self.replace(val,fill,inplace=True)
      return self
    def check_key(self,key):
      """
      check if a key is in the dataframe
      """
      col_key = panda_helpers.getcolumns([key],depth=self.key_depth) #Only provide one key
      if col_key[0] in self.columns:
        return True
      return False
    def split(self,bins,key,obj_comp=None,low_to_high=True):
      """
      Split self into list of self's split by bins determined by obj_comp
      """
      if obj_comp is None: obj_comp = self
      return object_calc.split_obj_into_bins(self,obj_comp,bins,key,low_to_high=low_to_high)
    def get_reference_obj(self,ref):
      """
      Use index of self to reference another object
      """
      if not object_calc.check_reference(self,ref): return None #check that the object can refer to the other one
      ref_inds = utils.get_inds_from_sub_inds(set(self.index.values),set(ref.index.values),len(ref.index.values[0]))
      return ref.loc[ref_inds]
      
      
      
      
        
      
      

