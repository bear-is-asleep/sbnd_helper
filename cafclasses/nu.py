import numpy as np
from time import time

from pyanalib import pandas_helpers
from sbnd.detector.volume import *
from sbnd.constants import *
from sbnd.cafclasses.object_calc import *
from sbnd.cafclasses.parent import CAF

class NU(CAF):
  def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs)
  def __getitem__(self, item):
        data = super().__getitem__(item) #Series or dataframe get item
        return NU(data)
  def load(fname,key='mcnu',**kwargs):
    if isinstance(key,list):
      for i,k in enumerate(key):
        if i == 0:
          thisnu = NU(pd.read_hdf(fname,key=k,**kwargs),**kwargs)
        else:
          thisnu.combine(NU(pd.read_hdf(fname,key=k,**kwargs),**kwargs))
      return thisnu
    elif isinstance(key,str):
      thisnu = NU(pd.read_hdf(fname,key=key,**kwargs),**kwargs)
      return thisnu
    else:
      raise ValueError(f'Invalid key: {key}')
  def add_av(self):
    """
    Add containment 1 or 0 for interaction
    """
    keys = [
      'av',
    ]
    self.add_key(keys, fill=False)
    cols = pandas_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[:,cols[0]] = involume(self.data.position,volume=AV).astype(bool)
  def add_fv(self):
    """
    Add containment 1 or 0 for interaction in the fiducial volume
    """
    keys = [
      'fv',
    ]
    self.add_key(keys, fill=False)
    cols = pandas_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[:,cols[0]] = involume(self.data.position,volume=FV).astype(bool)
  def add_nudir(self):
    """
    add nu direction
    """
    keys = [
      'nu_dir.x','nu_dir.y','nu_dir.z'
    ]
    self.add_key(keys)
    cols = pandas_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[:,cols[0:3]] = get_neutrino_dir(self.data.position)
  def add_theta(self,convert_to_deg=True):
    """
    add theta
    """
    keys = [
      'theta'
    ]
    self.add_key(keys)
    cols = pandas_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[:,cols[0]] = np.arccos(self.data.nu_dir.z)
    if convert_to_deg:
      self.data.loc[:,cols[0]] = self.data.loc[:,cols[0]]*180/np.pi #usefule for prism
  def add_event_type(self,algo,min_ke=0.1,suffix=""):
    """
    Add event type

    Parameters
    ----------
    algo : str
      Algorithm to use for event type. Either 'pandora' or 'spine' 
    min_ke : float
      Minimum kinetic energy to be considered a muon [GeV]
    """
    iscc = self.data.iscc == 1
    isnumu = abs(self.data.pdg) == 14
    isnue = abs(self.data.pdg) == 12
    iscosmic = (self.data.pdg == -1) | (self.data.pdg.isna())
    istrueav = self.data.av #filled to bool
    istruefv = self.data.fv #filled to bool
    if min_ke == 0.1:
      ismuon = self.data.nmu_100MeV > 0
    elif min_ke == 0.027:
      ismuon = self.data.nmu_27MeV > 0
    else:
      raise ValueError(f'Invalid min_ke: {min_ke}')
    if algo == 'pandora':
      col = self.get_key(f'mu{suffix}.is_pandora_contained')[0]
      iscont = (self.data.loc[:,col] == 1) | (self.data.loc[:,col] == True) #contained, break down signal and background
    elif algo == 'spine':
      col = self.get_key(f'mu{suffix}.is_spine_contained')[0]
      iscont = (self.data.loc[:,col] == 1) | (self.data.loc[:,col] == True) #contained, break down signal and background
    else:
      raise ValueError(f'Invalid algo: {algo}')
    
    #aggregate true types
    isnumuccav = iscc & isnumu & istrueav & ~iscosmic #numu cc av
    isnumuccfv = iscc & isnumu & istruefv & ~iscosmic #numu cc fv
    isnumuccfv_cont = isnumuccfv & iscont & ismuon
    isnumuccfv_uncont = isnumuccfv & ~iscont & ismuon
    isnumuccoops = isnumuccav & ~isnumuccfv_cont & ~isnumuccfv_uncont #numu cc out of phase space (oops)
    isnueccav = iscc & isnue & istrueav & ~iscosmic #nue cc av
    isncav = ~iscc & istrueav & ~iscosmic #nc av
    iscosmicav = istrueav & iscosmic #cosmic av
    isdirt = ~istrueav & ~iscosmic #dirt

    keys = [f'event_type_{algo}']
    self.add_key(keys,fill=-1)
    cols = pandas_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[isnumuccfv_cont,cols[0]] = 0 #numu cc (contained)
    self.data.loc[isnumuccfv_uncont,cols[0]] = 1 #numu cc (uncontained)
    self.data.loc[isnumuccoops,cols[0]] = 2 #numu cc oops
    self.data.loc[isdirt,cols[0]] = 3 #dirt
    self.data.loc[isnueccav,cols[0]] = 4 #nue cc
    self.data.loc[isncav,cols[0]] = 5 #nc
    self.data.loc[iscosmic,cols[0]] = 6 #cosmic
    #self.data.loc[isnumuccfv_cont | isnumuccfv_uncont,cols[0]] = 7 #numu cc (either)
    #the rest are unknown
  def cut_muon(self,cut=True,min_ke=0.1):
    """
    Cut to only muon
    """
    if min_ke == 0.1:
      self.apply_cut('cut.muon', self.data.nmu_100MeV > 0, cut=cut)
    elif min_ke == 0.027:
      self.apply_cut('cut.muon', self.data.nmu_27MeV > 0, cut=cut)
    else:
      raise ValueError(f'Invalid min_ke: {min_ke}')
    
  def cut_fv(self,cut=True):
    """
    Cut to only in fv
    """
    self.apply_cut('cut.fv', self.data.fv == True, cut=cut)
  def cut_cosmic(self,cut=True):
    """
    Cut to only cosmic
    """
    self.apply_cut('cut.cosmic', (self.data.pdg != -1) & ~(self.data.pdg.isna()), cut=cut)
  def cut_cont(self,cut=True):
    """
    Cut to only contained
    """
    self.apply_cut('cut.cont', self.data.mu.is_contained == 1, cut=cut)
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
  