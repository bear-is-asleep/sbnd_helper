import numpy as np
from time import time

from pyanalib import panda_helpers
from sbnd.detector.volume import *
from sbnd.constants import *
from sbnd.numu import selection as numusel
from sbnd.cafclasses.object_calc import *
from sbnd.cafclasses.parent import CAF

class NU(CAF):
  def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs)
  def __getitem__(self, item):
        data = super().__getitem__(item) #Series or dataframe get item
        return NU(data)
  def load(fname,key='mcnu',**kwargs):
      df = pd.read_hdf(fname,key=key,**kwargs)
      return NU(df,**kwargs)
  def all_cuts(self,volume=FV):
    """
    Apply all cuts
    """
    if volume == FV: self.cut_fv()
    else: self.cut_av()
    self.cut_isnumucc()
    #return self #Return self since cuts cannot be made in place
  def postprocess_and_cut(self,volume=FV):
    """
    Run all post processing and cuts in order to optimize timing
    """
    s0 = time()
    self.add_av()
    self.add_fv()
    self.add_isnumucc()
    s1 = time()
    print(f'--add variables: {s1-s0:.2f} s')
    #Now cut
    self.all_cuts(volume=volume)
    s2 = time()
    print(f'--cut variables: {s2-s1:.2f} s')
    #Now add rest of attributes
    self.add_nudir()
    self.add_theta()
    self.add_costheta()
    s3 = time()
    print(f'--add rest: {s3-s2:.2f} s')
    #Bin by prism bin
    self.assign_prism_bins()
    s4 = time()
    print(f'--assign binning: {s4-s3:.2f} s')
    #Scale to POT
    self.scale_to_pot(self.pot)
    s5 = time()
    print(f'--scale to POT: {s5-s4:.2f} s')
  def add_av(self):
    """
    Add containment 1 or 0 for each pfp
    """
    keys = [
      'in_tpc',
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[:,cols[0]] = involume(self.data.position)
    return None
  def add_fv(self):
    """
    Is it in the fiducial volume?
    """
    keys = [
      'in_fv',
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[:,cols[0]] = involume(self.data.position,volume=FV)
  def cut_fv(self):
    """
    Cut to only in fv
    """
    self.data = self.data[self.data.in_fv]
    
  def cut_av(self):
    """
    Cut to only in av
    """
    self.data = self.data[self.data.in_tpc]
  def add_isnumucc(self):
    """
    Add is numu cc
    """
    keys = [
      'is_numucc',
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[:,cols[0]] = numusel.get_numucc_mask(self.data)
  def cut_isnumucc(self):
    """
    Cut to only numu cc
    """
    self.data = self.data[self.data.is_numucc]
  def add_nudir(self):
    """
    add nu direction
    """
    keys = [
      'nu_dir.x','nu_dir.y','nu_dir.z'
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[:,cols[0:3]] = get_neutrino_dir(self.data.position)
  def add_theta(self,convert_to_deg=True):
    """
    add theta
    """
    keys = [
      'theta'
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[:,cols[0]] = np.arccos(self.data.nu_dir.z)
    if convert_to_deg:
      self.data.loc[:,cols[0]] = self.data.loc[:,cols[0]]*180/np.pi #usefule for prism
  def assign_prism_bins(self,prism_bins=None):
    """
    Assign prism bins to dataframe
    
    prism_bins is a list of prism bin edges
    """
    if prism_bins is not None: self.set_prism_bins(prism_bins)
    keys = [
      'prism_bins'
    ]
    self.add_key(keys)
    self.assign_bins(self.prism_binning,'theta',assign_key='prism_bins',low_to_high=True)