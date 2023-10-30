from pandas import DataFrame
import numpy as np
from pyanalib import panda_helpers
from sbnd.volume import *
from sbnd.constants import *
from .parent import CAF
from sbnd.cafclasses.object_calc import *

class MCPRIM(CAF):
    def __init__(self,df,prism_bins=None,momentum_bins=None,theta_bins=None,nu=None):
      super().__init__(df)
      self.nu_inrange = None
      self.assign_prism_bins(prism_bins)
      self.assign_momentum_bins(momentum_bins)
      self.assign_theta_bins(theta_bins)
    def assign_prism_bins(self,prism_bins):
      """
      Assign prism bins
      """
      self.prism_bins = prism_bins
    def assign_momentum_bins(self,momentum_bins):
      """
      Assign momentum bins
      """
      self.momentum_bins = momentum_bins
    def assign_theta_bins(self,theta_bins):
      """
      Assign theta bins
      """
      self.theta_bins = theta_bins
    def postprocess(self,nu=None):
      """
      Run all post processing
      """
      self.clean()
      self = self.drop_noninteracting()
      #self = self.drop_neutrinos() - this is taken care of by drop noninteracting
      self.add_fv()
      self.add_nu_dir()
      self.add_theta()
      self.add_costheta()
      self.add_momentum_mag()
      #self.add_genweight(nu=nu)
      return MCPRIM(self) #Have to return since we're dropping rows
    def add_fv(self):
      """
      Add containment for start and end of particle
      """
      keys = [
        'in_tpc',
      ]
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.loc[:,cols[0]] = involume(self.start) & involume(self.end)
      return None
    def get_true_parts(self,remove_nan=True,**dropna_args):
      """
      return true particles from track and shower matching
      """
      particles = self.copy()
      if remove_nan:
        particles = particles.dropna(**dropna_args)
      return MCPRIM(particles)
    
    def get_true_parts_from_pdg(self,pdg,remove_nan=True,**dropna_args):
      """
      Return particles from pdg
      """
      particles = self.get_true_parts(remove_nan=remove_nan,**dropna_args)
      particles = particles[abs(particles.pdg) == pdg]
      
      return MCPRIM(particles)
    def drop_neutrinos(self):
      """
      Drop rows with neutrinos
      """
      has_nu = (abs(self.pdg) == 14) | (abs(self.pdg) == 12)
      self = MCPRIM(self[~has_nu])
      return self
    def drop_noninteracting(self):
      """
      Drop rows where the visible energy is zero
      """
      visible = (((self.plane.I0.I0.nhit + self.plane.I0.I1.nhit + self.plane.I0.I2.nhit) > 0) &
                 [val is not np.nan for val in self.plane.I0.I0.nhit])
      self = MCPRIM(self[visible])
      return self
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
      self.loc[:,cols[0:3]] = get_neutrino_dir(self.start)
    def add_theta(self,convert_to_deg=True):
      """
      add dir
      """
      keys = [
        'theta'
      ]
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.loc[:,cols[0]] = get_theta(np.array(self.genp.values,dtype=np.float64),
                                      np.array(self.nu.dir.values,dtype=np.float64))
      if convert_to_deg:
        self.loc[:,cols[0]] = self.loc[:,cols[0]]*180/np.pi #usefule for prism
    def add_momentum_mag(self):
      """
      Add magnitude of momentum
      """
      mag = np.linalg.norm(self.genp,axis=1) #calc magnitude before adding key
      keys = [
        'genp.tot'
      ]
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.loc[:,cols[0]] = mag
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
    # def add_genweight(self,nu=None):
    #   """
    #   Add gen weight from mcnu object
    #   """
    #   keys = [
    #     'genweight'
    #   ]
    #   self.add_key(keys)
    #   cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    #   if nu is not None:
    #     inds = utils.get_inds_from_sub_inds(set(self.index.values),set(nu.index.values),3)
    #     weights = nu.loc[inds,'genweight'].values
    #   else:
    #     weights = np.ones(len(self))
    #   self.loc[:,cols[0]] = weights
    def set_nu_inrange(self,nu):
      """
      Get neutrino in current indices
      """
      self.nu_inrange = self.get_reference_obj(nu)
    def check_nu_inrange(self,nu=None):
      """
      Check that neutrino is in current indices
      """
      if self.nu_inrange is None:
        if nu is None:
          raise Exception("Need to provide nu object since it hasn't been set yet")
          return None
        else:
          self.set_nu_inrange(nu)
      return self.nu_inrange is not None
    def get_genweight(self,nu=None):
      """
      Get genie weights from nu object
      """
      check_nu_inrange(nu=nu)
      return self.nu_inrange.genweight
    def get_genmode(self,nu=None):
      """
      Get interaction mode from nu object
      """
      check_nu_inrange(nu=nu)
      return self.nu_inrange.genie_mode
    def get_numevents(self,nu=None):
      """
      Get number of events from nu object
      """
      weights = self.get_genweight(nu=nu)
      return np.sum(weights)
      
    
      
      