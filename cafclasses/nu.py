from pandas import DataFrame
import numpy as np
from pyanalib import panda_helpers
from sbnd.volume import *
from sbnd.constants import *
from sbnd.numu import selection as numusel
from sbnd.cafclasses.object_calc import *
from sbnd.cafclasses.parent import CAF

class NU(CAF):
  def __init__(self,df):
    super().__init__(df)
  def postprocess(self):
    """
    Run all post processing
    """
    self.clean()
    self.add_av()
    self.add_isnumucc()
    self.add_nudir()
    self.add_theta()
    self.add_costheta()
  def all_cuts(self):
    """
    Apply all cuts
    """
    self = self.cut_av()
    self = self.cut_isnumucc()
    return self #Return self since cuts cannot be made in place
  def add_av(self):
    """
    Add containment 1 or 0 for each pfp
    """
    keys = [
      'in_tpc',
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.loc[:,cols[0]] = involume(self.position)
    return None
  def cut_av(self):
    """
    Cut to only in av
    """
    return NU(self[self.in_tpc])
  def add_isnumucc(self):
    """
    Add is numu cc
    """
    keys = [
      'is_numucc',
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.loc[:,cols[0]] = numusel.get_numucc_mask(self)
  def cut_isnumucc(self):
    """
    Cut to only numu cc
    """
    return NU(self[self.is_numucc])
  def add_nudir(self):
    """
    add nu direction
    """
    keys = [
      'nu_dir.x','nu_dir.y','nu_dir.z'
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.loc[:,cols[0:3]] = get_neutrino_dir(self.position)
    return None
  def add_theta(self,convert_to_deg=True):
    """
    add theta
    """
    keys = [
      'theta'
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.loc[:,cols[0]] = np.arccos(self.nu_dir.z)
    if convert_to_deg:
      self.loc[:,cols[0]] = self.loc[:,cols[0]]*180/np.pi #usefule for prism
    return None
  def scale_to_pot(self,sample_pot,nom_pot):
    """
    Scale to nominal protons on target (POT). Need sample POT as input
    """
    if not self.check_key('genweight'): #key not in dataframe
      keys = ['genweight']
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.loc[:,cols[0]] = np.ones(len(self)) #initialize to ones
    self.genweight = self.genweight/nom_pot*sample_pot
    return None #inplace
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
        self.loc[:,cols[0]] = np.cos(self.theta/180*np.pi)*np.sign(self.theta)
      else:
        self.loc[:,cols[0]] = np.cos(self.theta)*np.sign(self.theta)