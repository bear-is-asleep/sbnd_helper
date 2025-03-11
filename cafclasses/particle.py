import pandas as pd
from .parent import CAF

class Particle(CAF):
    """ 
    Particle class - 
    Represents particles from ML Reco, Truth, and Pandora
    """
    #-------------------- constructor/rep --------------------#
    def __init__(self,*args,momentum_bins=None,costheta_bins=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.set_momentum_bins(momentum_bins)
        self.set_costheta_bins(costheta_bins)
    @property
    def _constructor(self):
        return Particle
    def __getitem__(self, item):
        data = super().__getitem__(item)
        return Particle(data
                ,prism_bins=self.prism_binning
                ,momentum_bins=self.momentum_binning
                ,costheta_bins=self.costheta_binning
                ,pot=self.pot)
    def copy(self, deep=True):
        return Particle(self.data.copy(deep)
                    ,prism_bins=self.prism_binning
                    ,momentum_bins=self.momentum_binning
                    ,costheta_bins=self.costheta_binning
                    ,pot=self.pot)
    def load(fname,key,**kwargs):
        """
        Load data from hdf5 file
        """
        df = pd.read_hdf(fname,key=key,**kwargs)
        return Particle(df,**kwargs)
    #-------------------- setters --------------------#
    def set_nu_inrange(self,nu):
        """
        Get neutrino in current indices
        """
        self.nu_inrange_df = self.get_reference_df(nu)
    #-------------------- helpers --------------------#
    #-------------------- adders --------------------#
    #-------------------- cutters --------------------#
    def apply_cut(self,partgrp,key):
      """
      Cut particle from the particle group based on the key
      """
      _partgrp = partgrp.copy()
      _partgrp.apply_cut(key)
      self.data = _partgrp.get_reference_df(self)
    #-------------------- getters --------------------#
    def get_particles(self,pdgs,remove_nan=True,use_reco=False,**dropna_args):
        """
        Return list of particles from list of pdgs
        """
        parts = [None]*len(pdgs)
        for i,pdg in enumerate(pdgs):
            parts[i] = self.get_parts_from_pdg(pdg,remove_nan=remove_nan,use_reco=use_reco,**dropna_args)
        return parts
    def get_parts_from_pdg(self,pdg,remove_nan=True,use_reco=False,**dropna_args):
        """
        Return particles from pdg
        """
        pass
    #-------------------- checkers --------------------#
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
    