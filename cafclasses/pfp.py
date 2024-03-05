import numpy as np
from pandas import DataFrame
import pandas as pd
from pyanalib import panda_helpers
from sbnd.cafclasses.object_calc import *
from sbnd.volume import *
from sbnd.constants import *
from .parent import CAF


class PFP(CAF):
  #-------------------- constructor/rep --------------------#
  def __init__(self,*args,momentum_bins=None,costheta_bins=None,**kwargs):
    super().__init__(*args,**kwargs)
    self.set_momentum_bins(momentum_bins)
    self.set_costheta_bins(costheta_bins)
    #self.clean([-5]) #set dummy values to nan, for some reason pandora uses -5??
  @property
  def _constructor(self):
      return PFP
  def __getitem__(self, item):
    data = super().__getitem__(item)
    return PFP(data
               ,prism_bins=self.prism_binning
               ,momentum_bins=self.momentum_binning
               ,costheta_bins=self.costheta_binning
               ,pot=self.pot)
  def copy(self, deep=True):
    return PFP(self.data.copy(deep)
                  ,prism_bins=self.prism_binning
                  ,momentum_bins=self.momentum_binning
                  ,costheta_bins=self.costheta_binning
                  ,pot=self.pot)
  def load(fname,key,**kwargs):
    """
    Load data from hdf5 file
    """
    df = pd.read_hdf(fname,key=key,**kwargs)
    pfp = PFP(df,**kwargs)
    return pfp
  #-------------------- setters --------------------#
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
  def set_nu_inrange(self,nu):
    """
    Get neutrino in current indices
    """
    self.nu_inrange_df = self.get_reference_df(nu)
  
  #-------------------- adders --------------------#
  def add_pfp_semantics(self,method='naive',threshold=0.5):
    """
    classify each as shower or track
    """
    #naive method for now
    if method == 'naive':
      is_shw = (self.data.trackScore < threshold) & (self.data.trackScore >= 0)
      is_trk = (self.data.trackScore >= threshold) & (self.data.trackScore <= 1)
    else:
      print(f'Method "{method}" not an option, choose from - "naive"')
      raise ValueError('Method not implemented')
    other = ~(is_trk | is_shw)
    keys = [
      'semantic_type'
    ]
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.add_key(keys,fill=False)
    self.data.loc[is_trk,cols[0]] = 0
    self.data.loc[is_shw,cols[0]] = 1
    self.data.loc[other,cols[0]] = -1
  
  def add_containment(self):
    """
    Add containment 1 or 0 for each pfp
    """
    keys = [
      'shw.cont_tpc',
      'trk.cont_tpc',
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[:,cols[0]] = involume(self.data.shw.start) & involume(self.data.shw.end)
    self.data.loc[:,cols[1]] = involume(self.data.trk.start) & involume(self.data.trk.end)
  
  def add_stats(self):
    self.add_stat(get_err,'mae',normalize=True)
    self.add_stat(get_err,'dif',normalize=False)
  
  def add_bestpdg(self,method='dazzle_best',include_other=False
                  ,length=25
                  ,bdt_score=0.5
                  ,chi2_muon=30
                  ,chi2_proton=60 #cut on muon
                  ,chi2_proton2=90 #cut on proton
                  ,dedx=2.5):
    """
    https://sbn-docdb.fnal.gov/cgi-bin/sso/RetrieveFile?docid=24747&filename=20220215_Mun%20Jung%20Jung.pdf&version=1
    Get best pdg for track and showers
    """
    is_trk = self.data.semantic_type == 0
    if include_other: #assume other is track if not matched to semantic type
      is_trk = is_trk | (self.data.semantic_type == -1)
    is_shw = self.data.semantic_type == 1
    is_other = ~(is_trk | is_shw)
    
    if method == 'dazzle_best':
      is_muon = (self.data.trk.dazzle.pdg == 13) & is_trk
      is_proton = (self.data.trk.dazzle.pdg == 2212) & is_trk
      is_pion = (self.data.trk.dazzle.pdg == 211) & is_trk
      is_electron = (self.data.shw.razzle.pdg == 11) & is_shw
      is_photon = (self.data.shw.razzle.pdg == 22) & is_shw
    elif method == 'dazzle': #can tune the values
      is_muon = (self.data.trk.dazzle.muonScore > bdt_score) & is_trk
      is_proton = (self.data.trk.dazzle.protonScore > bdt_score) & is_trk
      is_pion = (self.data.trk.dazzle.pionScore > bdt_score) & is_trk
      is_electron = (self.data.shw.razzle.electronScore > bdt_score) & is_shw
      is_photon = (self.data.shw.razzle.photonScore > bdt_score) & is_shw
    elif method == 'x2': #can tune values
      print('Assigning pdg using x2 method : ')
      print(f'length: {length:.2f}')
      print(f'chi2_muon: {chi2_muon:.2f}')
      print(f'chi2_proton: {chi2_proton:.2f} (cut on muon)')
      print(f'chi2_proton2: {chi2_proton2:.2f} (cut on proton)')
      print(f'dedx: {dedx:.2f}')
      
      is_muon = (self.data.trk.len > length) \
            & (self.data.trk.chi2pid.I2.chi2_muon < chi2_muon) \
            & (self.data.trk.chi2pid.I2.chi2_proton > chi2_proton) \
            & is_trk
      is_proton = (self.data.trk.chi2pid.I2.chi2_proton < chi2_proton2) & is_trk
      is_pion = ~is_muon & ~is_proton & is_trk \
        & (~self.data.trk.chi2pid.I2.isna().all(axis=1)) #not all NAN
      #naive estimate using dedx https://arxiv.org/pdf/1610.04102.pdf fig 9
      is_electron = (self.data.shw.bestplane_dEdx < dedx) & is_shw
      is_photon = ~is_electron & is_shw \
        & (~self.data.shw.bestplane_dEdx.isna()) #not all NAN
    
    else:
      print(f'Method "{method}" not an option, choose from - "x2", "dazzle_best", "dazzle"')
      raise ValueError('Method not implemented')
    
    #add key
    keys = [
      'bestpdg'
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[is_muon,cols[0]] = 13
    self.data.loc[is_proton,cols[0]] = 2212
    self.data.loc[is_pion,cols[0]] = 211
    self.data.loc[is_electron,cols[0]] = 11
    self.data.loc[is_photon,cols[0]] = 22
  
  def add_trk_bestenergy(self):
    """
    Get best energy for track
    """
    cont = self.data.trk.cont_tpc #containment boolean
    #PDG booleans
    muon = self.data.bestpdg == 13
    proton = self.data.bestpdg == 2212
    pion = self.data.bestpdg == 211
    
    keys = [
      'trk.bestenergy',
      'trk.bestmom'
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    #muon
    self.data.loc[cont & muon,cols[0]] = np.sqrt(self.data[cont & muon].trk.rangeP.p_muon**2+kMuonMass**2)
    self.data.loc[cont & muon,cols[1]] = self.data[cont & muon].trk.mcsP.fwdP_muon
    
    self.data.loc[~cont & muon,cols[0]] = np.sqrt(self.data[~cont & muon].trk.mcsP.fwdP_muon**2+kMuonMass**2)
    self.data.loc[~cont & muon,cols[1]] = self.data[~cont & muon].trk.mcsP.fwdP_muon
    #pion
    self.data.loc[cont & pion,cols[0]] = np.sqrt(self.data[cont & pion].trk.rangeP.p_pion**2+kPionMass**2)
    self.data.loc[cont & pion,cols[1]] = self.data[cont & pion].trk.mcsP.fwdP_pion
    
    self.data.loc[~cont & pion,cols[0]] = np.sqrt(self.data[~cont & pion].trk.mcsP.fwdP_pion**2+kPionMass**2)
    self.data.loc[~cont & pion,cols[1]] = self.data[~cont & pion].trk.mcsP.fwdP_pion
    #proton
    self.data.loc[cont & proton,cols[0]] = np.sqrt(self.data[cont & proton].trk.rangeP.p_proton**2+kProtonMass**2)
    self.data.loc[cont & proton,cols[1]] = self.data[cont & proton].trk.mcsP.fwdP_proton
    
    self.data.loc[~cont & proton,cols[0]] = np.sqrt(self.data[~cont & proton].trk.mcsP.fwdP_proton**2+kProtonMass**2)
    self.data.loc[~cont & proton,cols[1]] = self.data[~cont & proton].trk.mcsP.fwdP_proton
    
  def add_neutrino_dir(self):
    """
    add neutrino direction to df
    """
    keys = [
      'shw.nudir.x','shw.nudir.y','shw.nudir.z',
      'shw.truth.p.nudir.x','shw.truth.p.nudir.y','shw.truth.p.nudir.z',
      'trk.nudir.x','trk.nudir.y','trk.nudir.z',
      'trk.truth.p.nudir.x','trk.truth.p.nudir.y','trk.truth.p.nudir.z',
      'shw.prism_theta','shw.truth.p.prism_theta','trk.prism_theta','trk.truth.p.prism_theta',
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[:,cols[0:3]] = get_neutrino_dir(self.data.shw.start)
    self.data.loc[:,cols[3:6]]= get_neutrino_dir(self.data.shw.truth.p.start)
    self.data.loc[:,cols[6:9]] = get_neutrino_dir(self.data.trk.start)
    self.data.loc[:,cols[9:12]] = get_neutrino_dir(self.data.trk.truth.p.start)
    self.data.loc[:,cols[12]] = np.arccos(self.data.shw.nudir.z)*180/np.pi
    self.data.loc[:,cols[13]] = np.arccos(self.data.shw.truth.p.nudir.z)*180/np.pi
    self.data.loc[:,cols[14]] = np.arccos(self.data.trk.nudir.z)*180/np.pi
    self.data.loc[:,cols[15]] = np.arccos(self.data.trk.truth.p.nudir.z)*180/np.pi
  
  def add_theta(self):
    """
    add theta wrt nu direction
    """
    keys = [
      'shw.theta',
      'shw.truth.p.theta',
      'trk.theta',
      'trk.truth.p.theta',
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[:,cols[0]] = get_theta(self.data.shw.dir.values,self.data.shw.nudir.values)
    self.data.loc[:,cols[1]]= get_theta(self.data.shw.truth.p.genp.values,self.data.shw.truth.p.nudir.values)
    self.data.loc[:,cols[2]] = get_theta(self.data.trk.dir.values,self.data.trk.nudir.values)
    self.data.loc[:,cols[3]] = get_theta(self.data.trk.truth.p.genp.values,self.data.trk.truth.p.nudir.values)
  def add_Etheta(self):
    """
    add theta wrt nu direction
    """
    keys = [
      'shw.Etheta2',
      'shw.truth.p.Etheta2',
      'trk.Etheta2',
      'trk.truth.p.Etheta2',
    ]
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    self.data.loc[:,cols[0]] = self.data.shw.theta**2*self.data.shw.bestplane_energy
    self.data.loc[:,cols[1]]= self.data.shw.truth.p.theta**2*self.data.shw.truth.p.genE
    self.data.loc[:,cols[2]] = self.data.trk.theta**2*self.data.trk.bestenergy
    self.data.loc[:,cols[3]]= self.data.trk.truth.p.theta**2*self.data.trk.truth.p.genE
  # def add_visenergy(self):
  #   """
  #   Best plane true visible energy
  #   """
  #   keys = [
  #     'shw.truth.p.bestplane_energy',
  #     'trk.truth.p.bestplane_energy',
  #   ]
  #   self.add_key(keys)
  #   cols = panda_helpers.getcolumns(keys,depth=self.key_length())
  #   self.data.loc[:,cols[0]] = self.data.shw.truth
    
  def add_stat(self,func,key,**funkwargs):
    """
    add relevant errors to df
    """
    keys = [
      # Key names for comparing true and reco
      f'shw.stat.{key}.Etheta2',
      f'shw.stat.{key}.energy',
      f'shw.stat.{key}.theta',
      f'shw.stat.{key}.start.x',
      f'shw.stat.{key}.start.y',
      f'shw.stat.{key}.start.z',
      f'shw.stat.{key}.vtx',
      f'trk.stat.{key}.Etheta2',
      f'trk.stat.{key}.energy',
      f'trk.stat.{key}.theta',
      f'trk.stat.{key}.start.x',
      f'trk.stat.{key}.start.y',
      f'trk.stat.{key}.start.z',
      f'trk.stat.{key}.vtx',
    ]
    
    self.add_key(keys)
    cols = panda_helpers.getcolumns(keys,depth=self.key_length())
    
    #showers
    self.data.loc[:,cols[0]] = func(self.data.shw.Etheta2,self.data.shw.truth.p.Etheta2,**funkwargs)
    self.data.loc[:,cols[1]] = func(self.data.shw.bestplane_energy,self.data.shw.truth.p.genE,**funkwargs)
    self.data.loc[:,cols[2]] = func(self.data.shw.theta,self.data.shw.truth.p.theta,**funkwargs)
    self.data.loc[:,cols[3]] = func(self.data.shw.start.x,self.data.shw.truth.p.start.x,**funkwargs)
    self.data.loc[:,cols[4]] = func(self.data.shw.start.y,self.data.shw.truth.p.start.y,**funkwargs)
    self.data.loc[:,cols[5]] = func(self.data.shw.start.z,self.data.shw.truth.p.start.z,**funkwargs)
    self.data.loc[:,cols[6]] = func(self.data.shw.start.values,self.data.shw.truth.p.start.values,**funkwargs)
    #tracks
    self.data.loc[:,cols[7]] = func(self.data.trk.Etheta2,self.data.trk.truth.p.Etheta2,**funkwargs)
    self.data.loc[:,cols[8]] = func(self.data.trk.bestenergy,self.data.trk.truth.p.genE,**funkwargs)
    self.data.loc[:,cols[9]] = func(self.data.trk.theta,self.data.trk.truth.p.theta,**funkwargs)
    self.data.loc[:,cols[10]] = func(self.data.trk.start.x,self.data.trk.truth.p.start.x,**funkwargs)
    self.data.loc[:,cols[11]] = func(self.data.trk.start.y,self.data.trk.truth.p.start.y,**funkwargs)
    self.data.loc[:,cols[12]] = func(self.data.trk.start.z,self.data.trk.truth.p.start.z,**funkwargs)
    self.data.loc[:,cols[13]] = func(self.data.trk.start.values,self.data.trk.truth.p.start.values,**funkwargs)

    #If any stat is na, the others are also na
    self.data[self.data.trk.stat.isna().any(axis=1)].trk.stat = np.nan
    self.data[self.data.shw.stat.isna().any(axis=1)].shw.stat = np.nan
  
  def assign_costheta_bins(self,key='costheta',assign_key='costheta_bin',costheta_bins=None):
    """
    Assign costheta bins to dataframe
    
    costheta_bins: costheta bins set 
    """
    if costheta_bins is not None: self.set_costheta_bins(costheta_bins=costheta_bins)
    keys = [
      key
    ]
    self.check_key(key)
    #self.add_key(keys)
    self.assign_bins(self.costheta_binning,key,df_comp=None,assign_key=assign_key,low_to_high=True)
  def assign_momentum_bins(self,key='p',assign_key='momentum_bin',momentum_bins=None):
    """
    Assign momentum bins to dataframe
    
    momentum_bins: momentum bins set 
    """
    if momentum_bins is not None: self.set_momentum_bins(momentum_bins=momentum_bins)
    keys = [
      key
    ]
    self.check_key(key)
    #self.add_key(keys)
    self.assign_bins(self.momentum_binning,key,df_comp=None,assign_key=assign_key,low_to_high=True)
  def assign_prism_bins(self,key='prism_theta',assign_key='prism_bin',prism_bins=None):
    """
    Assign prism bins to dataframe
    
    prism_bins: prism bins set 
    """
    if prism_bins is not None: self.set_prism_bins(prism_bins=prism_bins)
    keys = [
      key
    ]
    self.check_key(key)
    #self.add_key(keys)
    self.assign_bins(self.prism_binning,key,df_comp=None,assign_key=assign_key,low_to_high=True)
  
  def postprocess(self,fill=np.nan,dummy=np.nan,method_sem='naive',method_pdg='x2'):
    """
    Do all the post processing in the correct order
    """
    #fixers
    self.fix_shw_energy(fill=fill,dummy=dummy)
    #adders
    self.add_pfp_semantics(method=method_sem)
    self.add_bestpdg(method=method_pdg)
    self.add_reco_containment()
    self.add_neutrino_dir()
    self.add_theta()
    self.add_trk_bestenergy()
    self.add_Etheta()
    self.add_stats()
  

  #-------------------- getters --------------------#
  def get_best_muon(self,method='energy'):
    """
    Return best muon
    """
    pfp_muon = self.get_parts_from_pdg(13,remove_nan=True,use_reco=True)
    if method == 'energy':
      best_muon = pfp_muon.data.groupby(level=[0,1,2]).apply(lambda x: x.loc[x.trk.bestenergy.idxmax()])
    elif method == 'length':
      best_muon = pfp_muon.data.groupby(level=[0,1,2]).apply(lambda x: x.loc[x.trk.len.idxmax()])
    else:
      print(f'Method "{method}" not an option, choose from - "energy","length"')
      raise ValueError('Method not implemented')
    return PFP(best_muon
               ,prism_bins=self.prism_binning
               ,momentum_bins=self.momentum_binning
               ,costheta_bins=self.costheta_binning
               ,pot=self.pot)
    
  
  def get_particles(self,pdgs,remove_nan=True,use_reco=False,**dropna_args):
    """
    Return list of particles from list of pdgs
    """
    parts = [None]*len(pdgs)
    for i,pdg in enumerate(pdgs):
      parts[i] = self.get_parts_from_pdg(pdg,remove_nan=remove_nan,use_reco=use_reco,**dropna_args)
    return parts
    
  def get_total_reco_energy(self):
    """
    Return total reco energy from pfp
    """
    return get_row_vals(self.data.shw.bestplane_energy) #Add tracks later
  def get_min_theta(self):
    """
    Return minimum theta from pfp
    """
    return get_row_vals(self.data.shw.theta,mode='min')
  def get_true_parts(self,remove_nan=True,**dropna_args):
    """
    return true particles from track and shower matching
    """
    particles = self.copy()
    p = pd.concat([particles.data.trk.truth.p,particles.data.shw.truth.p],axis=0).drop_duplicates()
    if remove_nan:
      p = p.dropna(**dropna_args)
    particles.data = particles.data.loc[p.index] 
    return particles
  
  def get_parts_from_pdg(self,pdg,remove_nan=True,use_reco=False,**dropna_args):
    """
    Return particles from pdg
    """
    if not use_reco:
      particles = self.get_true_parts(remove_nan=remove_nan,**dropna_args)
      p = particles.data[particles.data.trk.truth.p.pdg == pdg]
    else:
      particles = self.copy()
      p = particles.data[particles.data.bestpdg == pdg]
    return PFP(p
               ,prism_bins=self.prism_binning
               ,momentum_bins=self.momentum_binning
               ,costheta_bins=self.costheta_binning
               ,pot=self.pot)
    
  #-------------------- cleaners --------------------#
  def clean_dummy_track_score(self):
    """
    Make a column that determines if the pfp is valid or not. 
    For now this just means it has a track score E [0,1]
    """
    mask_lower = self.data.trackScore >= 0
    mask_upper = self.data.trackScore <= 1
    mask = np.logical_and(mask_lower,mask_upper)
    self.data = self.data[mask]
  def clean_nu_inrange(self):
    self.nu_inrange_df = None #overwrite data
  
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

  #-------------------- fixers --------------------#
  def fix_shw_energy(self,fill=-5,dummy=-5):
      ## correct the reconstructed shower energy column
      nhits2 = ((self.data.shw.plane.I2.nHits >= self.data.shw.plane.I1.nHits) & (self.data.shw.plane.I2.nHits>= self.data.shw.plane.I0.nHits))
      nhits1 = ((self.data.shw.plane.I1.nHits >= self.data.shw.plane.I2.nHits) & (self.data.shw.plane.I1.nHits>= self.data.shw.plane.I0.nHits))
      nhits0 = ((self.data.shw.plane.I0.nHits >= self.data.shw.plane.I2.nHits) & (self.data.shw.plane.I0.nHits>= self.data.shw.plane.I1.nHits))
      # if energy[plane] is positive
      energy2 = (self.data.shw.plane.I2.energy > 0 )
      energy1 = (self.data.shw.plane.I1.energy > 0 )
      energy0 = (self.data.shw.plane.I0.energy > 0 )
      conditions = [(nhits2 & energy2),
                  (nhits1 & energy1),
                  (nhits0 & energy0),
                  (((nhits2 & energy2)== False) & (energy1) & (self.data.shw.plane.I1.nHits>= self.data.shw.plane.I0.nHits)), # if 2 is invalid, and 1 is positive and 1>0, go with 1 
                  (((nhits2 & energy2)== False) & (energy0) & (self.data.shw.plane.I0.nHits>= self.data.shw.plane.I1.nHits)), # if 2 is invalid, and 0 is positive and 0>1, go with 0
                  (((nhits1 & energy1)== False) & (energy2) & (self.data.shw.plane.I2.nHits>= self.data.shw.plane.I0.nHits)), # if 1 is invalid, and 2 is positive and 2>0, go with 2 
                  (((nhits1 & energy1)== False) & (energy0) & (self.data.shw.plane.I0.nHits>= self.data.shw.plane.I2.nHits)), # if 1 is invalid, and 0 is positive and 0>2, go with 0
                  (((nhits0 & energy0)== False) & (energy2) & (self.data.shw.plane.I2.nHits>= self.data.shw.plane.I1.nHits)), # if 0 is invalid, and 2 is positive and 2>1, go with 2              
                  (((nhits0 & energy0)== False) & (energy1) & (self.data.shw.plane.I1.nHits>= self.data.shw.plane.I2.nHits)), # if 0 is invalid, and 1 is positive and 1>2, go with 1 
                  ((self.data.shw.plane.I2.nHits==dummy) & (self.data.shw.plane.I1.nHits==dummy) & (self.data.shw.plane.I0.nHits==dummy))]
      shw_choices = [ self.data.shw.plane.I2.energy,
                      self.data.shw.plane.I1.energy,
                      self.data.shw.plane.I0.energy,
                      self.data.shw.plane.I1.energy,
                      self.data.shw.plane.I0.energy,
                      self.data.shw.plane.I2.energy,
                      self.data.shw.plane.I0.energy,
                      self.data.shw.plane.I2.energy,
                      self.data.shw.plane.I1.energy,
                      -1]
      self.data.loc[:,panda_helpers.getcolumns(['shw.bestplane_energy'],depth=self.key_length())] = np.select(conditions, shw_choices, default = fill)
  #-------------------- cutters --------------------#
  def apply_cut(self,slc,key):
      """
      Cut pfp from the slc based on the key
      """
      _slc = slc.copy()
      _slc.apply_cut(key)
      self.data = _slc.get_reference_df(self)