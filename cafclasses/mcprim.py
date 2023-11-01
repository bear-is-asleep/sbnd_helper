from pandas import DataFrame
import numpy as np
from time import time

from pyanalib import panda_helpers
from sbnd.volume import *
from sbnd.constants import *
from sbnd.cafclasses.parent import CAF
from sbnd.cafclasses.object_calc import *

class MCPRIM(CAF):
    def __init__(self,*args,prism_bins=None,momentum_bins=None,costheta_bins=None,**kwargs):
      super().__init__(*args,**kwargs)
      self.nu_inrange_df = None
      self.set_prism_bins(prism_bins)
      self.set_momentum_bins(momentum_bins)
      self.set_costheta_bins(costheta_bins)
    @property
    def _constructor(self):
        return MCPRIM
    def __getitem__(self, item):
        data = super().__getitem__(item) #Series or dataframe get item
        return MCPRIM(data
                      ,prism_bins=self.prism_binning
                      ,momentum_bins=self.momentum_binning
                      ,costheta_bins=self.costheta_binning
                      ,pot=self.pot)
    def copy(self, deep=True):
        return MCPRIM(self.data.copy(deep)
                      ,prism_bins=self.prism_binning
                      ,momentum_bins=self.momentum_binning
                      ,costheta_bins=self.costheta_binning
                      ,pot=self.pot)
    def postprocess(self,nu=None):
      """
      Run all post processing
      """
      s0 = time()
      self.apply_nu_cuts(nu=nu) 
      self.drop_noninteracting()
      s1 = time()
      print(f'--apply cuts: {s1-s0:.2f} s')
      #self.drop_neutrinos() - this is taken care of by drop noninteracting
      self.add_fv()
      self.add_nu_dir()
      self.add_theta()
      self.add_costheta()
      self.add_momentum_mag()
      self.add_genweight(nu=nu) #generator weights
      self.add_genmode(nu=nu) #generator modes
      s2 = time()
      print(f'--add variables: {s2-s1:.2f} s')
      #Assign binning
      self.assign_prism_bins(nu=nu)
      self.assign_costheta_bins()
      self.assign_momentum_bins()
      s3 = time()
      print(f'--assign bins: {s3-s2:.2f} s')
    def apply_nu_cuts(self,nu=None):
      """
      Apply cuts from nu object - MAKE SURE THE CUTS ARE ALREADY APPLIED TO THE NU OBJECT
      """
      nu_ind_depth = len(nu.data.index.values[0])
      inds = utils.get_inds_from_sub_inds(self.data.index.values,nu.data.index.values,nu_ind_depth)
      self.data = self.data.loc[inds]
      self.data.sort_index(inplace=True)
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
      
    def add_fv(self):
      """
      Add containment for start and end of particle
      """
      keys = [
        'in_tpc',
      ]
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[:,cols[0]] = involume(self.data.start) & involume(self.data.end)
      return None
    def get_true_parts(self,remove_nan=True,**dropna_args):
      """
      return true particles from track and shower matching
      """
      mcprim = self.copy()
      if remove_nan:
        mcprim.data = self.data.dropna(**dropna_args)
      return mcprim
    
    def get_true_parts_from_pdg(self,pdg,remove_nan=True,**dropna_args):
      """
      Return particles from pdg
      """
      mcprim = self.get_true_parts(remove_nan=remove_nan,**dropna_args)
      mcprim.data = mcprim.data[abs(mcprim.data.pdg) == pdg]
      return mcprim
    def drop_neutrinos(self):
      """
      Drop rows with neutrinos
      """
      has_nu = (abs(self.data.pdg) == 14) | (abs(self.data.pdg) == 12)
      self.data = self.data[~has_nu]
    def drop_noninteracting(self):
      """
      Drop rows where the visible energy is zero
      """
      visible = (((self.data.plane.I0.I0.nhit + self.data.plane.I0.I1.nhit + self.data.plane.I0.I2.nhit) > 0) &
                 [val is not np.nan for val in self.data.plane.I0.I0.nhit])
      self.data = self.data[visible] 
    def add_nu_dir(self):
      """
      add nu dir - we are using the start point of the primary which is not technically correct 
      but is close enough
      """
      keys = [
        'nu.dir.x','nu.dir.y','nu.dir.z'
      ]
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[:,cols[0:3]] = get_neutrino_dir(self.data.start)
    def add_theta(self,convert_to_deg=True):
      """
      add dir
      """
      keys = [
        'theta'
      ]
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[:,cols[0]] = get_theta(np.array(self.data.genp.values,dtype=np.float64),
                                      np.array(self.data.nu.dir.values,dtype=np.float64))
      if convert_to_deg:
        self.data.loc[:,cols[0]] = self.data.loc[:,cols[0]]*180/np.pi #usefule for prism
    def add_momentum_mag(self):
      """
      Add magnitude of momentum
      """
      mag = np.linalg.norm(self.data.genp,axis=1) #calc magnitude before adding key
      keys = [
        'genp.tot'
      ]
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[:,cols[0]] = mag
    def set_nu_inrange(self,nu):
      """
      Get neutrino in current indices
      """
      self.nu_inrange_df = self.get_reference_df(nu)
    def clear_nu_inrange(self):
      self.nu_inrange_df = None #overwrite data
    def check_nu_inrange(self,nu=None):
      """
      Check that neutrino is in current indices
      """
      if self.nu_inrange_df is None:
        if nu is None:
          raise Exception("Need to provide nu object since it hasn't been set yet")
        else:
          self.set_nu_inrange(nu)
          if self.nu_inrange_df.index.duplicated().any():
            raise Exception("There are duplicated indices in the nu object")
      return self.nu_inrange_df is not None
    def get_genweight(self,nu=None):
      """
      Get genie weights from nu object
      """
      if not self.check_nu_inrange(nu=nu): return None
      genweight = self.nu_inrange_df.genweight 
      self.clear_nu_inrange()
      return genweight
    def get_genmode(self,nu=None):
      """
      Get interaction mode from nu object
      """
      if not self.check_nu_inrange(nu=nu): return None
      genmode = self.nu_inrange_df.genie_mode
      self.clear_nu_inrange()
      return genmode
    def get_numevents(self,nu=None):
      """
      Get number of events from nu object
      """
      weights = self.get_genweight(nu=nu)
      return np.sum(weights)
    def assign_prism_bins(self,prism_bins=None,nu=None):
      """
      Assign prism bins to dataframe
      
      prism_bins: prism bins set 
      nu: nu object
      """
      if prism_bins is not None: self.set_prism_bins(prism_bins=prism_bins)
      keys = [
        'prism_bins'
      ]
      self.add_key(keys)
      if not self.check_nu_inrange(nu=nu): return None
      self.assign_bins(self.prism_binning,'theta',df_comp=self.nu_inrange_df,assign_key='prism_bins',low_to_high=True)
      self.clear_nu_inrange()
    def assign_costheta_bins(self,costheta_bins=None):
      """
      Assign costheta bins to dataframe
      
      costheta_bins: costheta bins set 
      """
      if costheta_bins is not None: self.set_costheta_bins(costheta_bins=costheta_bins)
      keys = [
        'costheta_bins'
      ]
      self.add_key(keys)
      self.assign_bins(self.costheta_binning,'costheta',df_comp=None,assign_key='costheta_bins',low_to_high=True)
    def assign_momentum_bins(self,momentum_bins=None):
      """
      Assign momentum bins to dataframe
      
      momentum_bins: momentum bins set 
      """
      if momentum_bins is not None: self.set_momentum_bins(momentum_bins=momentum_bins)
      keys = [
        'momentum_bins'
      ]
      self.add_key(keys)
      self.assign_bins(self.momentum_binning,'genp.tot',df_comp=None,assign_key='momentum_bins',low_to_high=True)
    def add_genweight(self,nu=None):
      """
      Add genie weights from nu object
      """
      if not self.check_nu_inrange(nu=nu): return None
      keys = [
        'genweight'
      ]
      self.add_key(keys)
      #Get neutrino indices
      nu_inds = utils.get_sub_inds_from_inds(set(self.data.index.values)
                                       ,set(self.nu_inrange_df.index.values)
                                       ,len(self.nu_inrange_df.index.values[0]))
      assert len(nu_inds) == len(self.nu_inrange_df.loc[nu_inds]), "Number of indices don't match"
      self.data.loc[:,keys[0]] = self.nu_inrange_df.loc[nu_inds].genweight
      self.clear_nu_inrange()
    def add_genmode(self,nu=None):
      """
      Add interaction mode from nu object
      """
      if not self.check_nu_inrange(nu=nu): return None
      keys = [
        'genie_mode'
      ]
      self.add_key(keys)
      #Get neutrino indices
      nu_inds = utils.get_sub_inds_from_inds(set(self.data.index.values)
                                       ,set(self.nu_inrange_df.index.values)
                                       ,len(self.nu_inrange_df.index.values[0]))
      assert len(nu_inds) == len(self.nu_inrange_df.loc[nu_inds]), "Number of indices don't match"
      #Sort indices
      self.data.loc[:,keys[0]] = self.nu_inrange_df.loc[nu_inds].genie_mode
      self.clear_nu_inrange()
    def get_part_count(self,pdg=None):
      """
      Return the number of particles with pdg. 
      If no pdg specified return total number of particles
      """
      if pdg is None:
        return np.sum(self.data.genweight)
      else:
        return np.sum(self.data.genweight[self.data.pdg == pdg])
      
    
    
      
      