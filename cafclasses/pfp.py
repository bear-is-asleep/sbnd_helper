import numpy as np
from pandas import DataFrame
import pandas as pd
from pyanalib import pandas_helpers
from sbnd.cafclasses.object_calc import *
from sbnd.detector.volume import *
from sbnd.constants import *
from .particle import Particle


class PFP(Particle):
  #-------------------- constructor/rep --------------------#
  def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs)
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
    if isinstance(key,list):
      for i,k in enumerate(key):
        if i == 0:
          thispfp = PFP(pd.read_hdf(fname,key=k,**kwargs))
        else:
          thispfp.combine(PFP(pd.read_hdf(fname,key=k,**kwargs)))
      return thispfp
    elif isinstance(key,str):
      thispfp = PFP(pd.read_hdf(fname,key=key,**kwargs))
    else:
      raise ValueError(f'Invalid key: {key}')
  #-------------------- helpers --------------------#
  def classify_semantics(self,method='naive',threshold=0.5):
    """
    How do we classify tracks, showers, etc.?
    """
    if method == 'naive':
      is_shw = (self.data.trackScore < threshold) & (self.data.trackScore >= 0)
      is_trk = (self.data.trackScore >= threshold) & (self.data.trackScore <= 1)
    else:
      print(f'Method "{method}" not an option, choose from - "naive"')
      raise ValueError('Method not implemented')
    other = ~(is_trk | is_shw)
    return is_shw,is_trk,other
  def classify_pid(self,method='chi2',include_other=False
                  ,length=25
                  ,bdt_score=0.5
                  ,chi2_muon=30
                  ,chi2_proton=60 #cut on muon
                  ,chi2_proton2=90 #cut on proton
                  ,dedx=2.5):
    """
    How do we classify the particle type?
    """
    is_trk = self.data.semantic_type == 0
    if include_other: #assume other is track if not matched to semantic type
      is_trk = is_trk | (self.data.semantic_type == -1)
    is_shw = self.data.semantic_type == 1
    
    if method == 'dazzle_best':
      print('Assigning pdg using dazzle_best method : ')
      is_muon = (self.data.trk.dazzle.pdg == 13) & is_trk
      is_proton = (self.data.trk.dazzle.pdg == 2212) & is_trk
      is_pion = (self.data.trk.dazzle.pdg == 211) & is_trk
      is_electron = (self.data.shw.razzle.pdg == 11) & is_shw
      is_photon = (self.data.shw.razzle.pdg == 22) & is_shw
    elif method == 'dazzle': #can tune the values
      print('Assigning pdg using dazzle method : ')
      #print(f'bdt_score: {bdt_score:.2f}')
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
    return is_photon,is_electron,is_muon,is_pion,is_proton
  def calculate_track_energy_momentum(self,part_mask,key,mass):
    """
    Calculate energy and momentum for a particle
    """
    cont = self.data.trk.cont_tpc & part_mask
    uncont = ~self.data.trk.cont_tpc & part_mask
    
    eng_col,mom_col = panda_helpers.getcolumns([f'trk.bestenergy',f'trk.bestmom'],depth=self.key_length())
    
    #Range method
    self.data.loc[cont,eng_col] = np.sqrt(self.data[cont].trk.rangeP[f'p_{key}']**2+mass**2)
    self.data.loc[cont,mom_col] = self.data[cont].trk.rangeP[f'p_{key}']
    
    #MCS method
    self.data.loc[uncont,eng_col] = np.sqrt(self.data[uncont].trk.mcsP[f'fwdP_{key}']**2+mass**2)
    self.data.loc[uncont,mom_col] = self.data[uncont].trk.mcsP[f'fwdP_{key}']
    
  #-------------------- adders --------------------#
  def add_pfp_semantics(self,method='naive',threshold=0.5):
    """
    classify each as shower or track
    """
    #Get semantic conditions
    is_shw,is_trk,is_other = self.classify_semantics(method=method,threshold=threshold)
      
    #Set keys, conditions, and values for add_col method
    keys = [
      'semantic_type'
    ]
    conditions = [
        is_trk,is_shw,is_other
    ]
    values = [0, 1, -1]
    self.add_cols(keys,values,conditions=conditions,fill=-1)
  
  def add_containment(self,volume=AV_BUFFER):
    """
    Add containment 1 or 0 for each pfp
    """
    #Set keys, conditions, and values for add_col method
    keys = [
      'trk.cont_tpc',
    ]
    conditions = [
      involume(self.data.trk.start,volume=volume) & involume(self.data.trk.end,volume=volume),
    ]
    values = [True]
    self.add_cols(keys,values,conditions=conditions,fill=False)
  
  def add_stats(self):
    self.add_stat(get_err,'mae',normalize=True)
    self.add_stat(get_err,'dif',normalize=False)
  
  def add_bestpdg(self,**kwargs):
    """
    https://sbn-docdb.fnal.gov/cgi-bin/sso/RetrieveFile?docid=24747&filename=20220215_Mun%20Jung%20Jung.pdf&version=1
    Get best pdg for track and showers
    """
    #Get pids
    is_photon,is_electron,is_muon,is_pion,is_proton = self.classify_pid(**kwargs)
    
    #Set keys, conditions, and values for add_col method
    keys = [
      'bestpdg'
    ]
    conditions = [
      is_photon,is_electron,is_muon,is_pion,is_proton
    ]
    values = [22,11,13,211,2212]
    self.add_cols(keys,values,conditions=conditions,fill=DUMMY_INT)
  
  def add_trk_bestenergy(self):
    """
    Get best energy for track
    """
    #PDG booleans
    muon = self.data.bestpdg == 13
    proton = self.data.bestpdg == 2212
    pion = self.data.bestpdg == 211
    
    keys = [
      'trk.bestenergy',
      'trk.bestmom'
    ]
    self.add_key(keys)
    
    #Calculate the values which will add it to the dataframe
    self.calculate_track_energy_momentum(muon,'muon',kMuonMass)
    self.calculate_track_energy_momentum(proton,'proton',kProtonMass)
    self.calculate_track_energy_momentum(pion,'pion',kPionMass)
    
  def add_neutrino_dir(self):
    """
    add neutrino direction to df
    """
    # Set keys, conditions, and values for add_col method
    keys = [
      'shw.nudir.x','shw.nudir.y','shw.nudir.z',
      'shw.truth.p.nudir.x','shw.truth.p.nudir.y','shw.truth.p.nudir.z',
      'trk.nudir.x','trk.nudir.y','trk.nudir.z',
      'trk.truth.p.nudir.x','trk.truth.p.nudir.y','trk.truth.p.nudir.z',
    ]
    neutrino_dirs = [
        get_neutrino_dir(self.data.shw.start),
        get_neutrino_dir(self.data.shw.truth.p.start),
        get_neutrino_dir(self.data.trk.start),
        get_neutrino_dir(self.data.trk.truth.p.start)
    ]
    #Safely flatten the neutrino directions
    stacked = np.array(neutrino_dirs) #4, N, 3
    #Transpose to get to 4,3,N then reshape to 12,N
    neutrino_dirs = stacked.transpose(0,2,1).reshape(12, -1)
    #Add neutrino directions first
    self.add_cols(keys,neutrino_dirs,fill=np.float64(np.nan))
    
    keys = [
      'shw.prism_theta','shw.truth.p.prism_theta','trk.prism_theta','trk.truth.p.prism_theta',
    ]
    prism_thetas = [
        np.arccos(self.data.shw.nudir.z) * 180 / np.pi,
        np.arccos(self.data.shw.truth.p.nudir.z) * 180 / np.pi,
        np.arccos(self.data.trk.nudir.z) * 180 / np.pi,
        np.arccos(self.data.trk.truth.p.nudir.z) * 180 / np.pi
    ]
    #Now add prism info
    self.add_cols(keys,prism_thetas,fill=np.float64(np.nan))
  
  def add_theta(self):
    """
    add theta wrt nu direction
    """
    # Set keys, conditions, and values for add_col method
    keys = [
      'shw.theta',
      'shw.truth.p.theta',
      'trk.theta',
      'trk.truth.p.theta',
    ]
    thetas = [
        get_theta(self.data.shw.dir.values,self.data.shw.nudir.values),
        get_theta(self.data.shw.truth.p.genp.values,self.data.shw.truth.p.nudir.values),
        get_theta(self.data.trk.dir.values,self.data.trk.nudir.values),
        get_theta(self.data.trk.truth.p.genp.values,self.data.trk.truth.p.nudir.values)
    ]
    self.add_cols(keys,thetas)
  def add_Etheta(self):
    """
    add theta wrt nu direction
    """
    # Set keys, conditions, and values for add_col method
    keys = [
      'shw.Etheta2',
      'shw.truth.p.Etheta2',
      'trk.Etheta2',
      'trk.truth.p.Etheta2',
    ]
    self.add_key(keys)
    Ethetas = [
        self.data.shw.bestplane_energy*self.data.shw.theta**2,
        self.data.shw.truth.p.genE*self.data.shw.truth.p.theta**2,
        self.data.trk.bestenergy*self.data.trk.theta**2,
        self.data.trk.truth.p.genE*self.data.trk.truth.p.theta**2
    ]
    self.add_cols(keys,Ethetas)
    
  def add_stat(self,func,key,**funkwargs):
    """
    add relevant errors to df
    """
    keys = [
      # Key names for comparing true and reco
      #f'shw.stat.{key}.Etheta2',
      f'shw.stat.{key}.energy',
      f'shw.stat.{key}.theta',
      f'shw.stat.{key}.start.x',
      f'shw.stat.{key}.start.y',
      f'shw.stat.{key}.start.z',
      f'shw.stat.{key}.vtx',
      #f'trk.stat.{key}.Etheta2',
      f'trk.stat.{key}.energy',
      f'trk.stat.{key}.theta',
      f'trk.stat.{key}.start.x',
      f'trk.stat.{key}.start.y',
      f'trk.stat.{key}.start.z',
      f'trk.stat.{key}.vtx',
    ]
    
    stats = [
      #func(self.data.shw.Etheta2,self.data.shw.truth.p.Etheta2,**funkwargs),
      func(self.data.shw.bestplane_energy,self.data.shw.truth.p.genE,**funkwargs),
      func(self.data.shw.theta,self.data.shw.truth.p.theta,**funkwargs),
      func(self.data.shw.start.x,self.data.shw.truth.p.start.x,**funkwargs),
      func(self.data.shw.start.y,self.data.shw.truth.p.start.y,**funkwargs),
      func(self.data.shw.start.z,self.data.shw.truth.p.start.z,**funkwargs),
      func(self.data.shw.start.values,self.data.shw.truth.p.start.values,**funkwargs),
      #func(self.data.trk.Etheta2,self.data.trk.truth.p.Etheta2,**funkwargs),
      func(self.data.trk.bestenergy,self.data.trk.truth.p.genE,**funkwargs),
      func(self.data.trk.theta,self.data.trk.truth.p.theta,**funkwargs),
      func(self.data.trk.start.x,self.data.trk.truth.p.start.x,**funkwargs),
      func(self.data.trk.start.y,self.data.trk.truth.p.start.y,**funkwargs),
      func(self.data.trk.start.z,self.data.trk.truth.p.start.z,**funkwargs),
      func(self.data.trk.start.values,self.data.trk.truth.p.start.values,**funkwargs)
    ]
    
    self.add_cols(keys,stats)

    #If any stat is na, the others are also na
    self.data[self.data.trk.stat.isna().any(axis=1)].trk.stat = np.nan
    self.data[self.data.shw.stat.isna().any(axis=1)].shw.stat = np.nan
  
  def assign_costheta_bins(self,key='costheta',assign_key='costheta_bin',costheta_bins=None):
    """
    Assign costheta bins to dataframe
    
    costheta_bins: costheta bins set 
    """
    if costheta_bins is not None: self.set_costheta_bins(costheta_bins=costheta_bins)
    self.check_key(key)
    self.assign_bins(self.costheta_binning,key,df_comp=None,assign_key=assign_key,low_to_high=True)
  def assign_momentum_bins(self,key='p',assign_key='momentum_bin',momentum_bins=None):
    """
    Assign momentum bins to dataframe
    
    momentum_bins: momentum bins set 
    """
    if momentum_bins is not None: self.set_momentum_bins(momentum_bins=momentum_bins)
    self.check_key(key)
    self.assign_bins(self.momentum_binning,key,df_comp=None,assign_key=assign_key,low_to_high=True)
  def assign_prism_bins(self,key='prism_theta',assign_key='prism_bin',prism_bins=None):
    """
    Assign prism bins to dataframe
    
    prism_bins: prism bins set 
    """
    if prism_bins is not None: self.set_prism_bins(prism_bins=prism_bins)
    self.check_key(key)
    self.assign_bins(self.prism_binning,key,df_comp=None,assign_key=assign_key,low_to_high=True)
  

  #-------------------- getters --------------------#
  def get_best_muon(self,method='energy'):
    """
    Return best muon
    """
    pfp_muon = self.get_parts_from_pdg(13,remove_nan=True,use_reco=True)
    if method == 'energy':
      best_muon = pfp_muon.data.groupby(level=[0,1]).apply(lambda x: x.loc[x.trk.bestenergy.idxmax()])
    elif method == 'length':
      best_muon = pfp_muon.data.groupby(level=[0,1]).apply(lambda x: x.loc[x.trk.len.idxmax()])
    else:
      print(f'Method "{method}" not an option, choose from - "energy","length"')
      raise ValueError('Method not implemented')
    return PFP(best_muon
               ,prism_bins=self.prism_binning
               ,momentum_bins=self.momentum_binning
               ,costheta_bins=self.costheta_binning
               ,pot=self.pot)
  def get_total_reco_energy(self):
    """
    Return total reco energy from pfp
    """
    return get_row_vals(self.data.shw.bestplane_energy)+get_row_vals(self.data.trk.bestenergy) #Add tracks later
  def get_min_theta(self):
    """
    Return minimum theta from pfp
    """
    shw_min = get_row_vals(self.data.shw.theta,mode='min')
    trk_min = get_row_vals(self.data.trk.theta,mode='min')
    return np.min(shw_min,trk_min,axis=0)
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
  #-------------------- cutters --------------------#
  #-------------------- fixers --------------------#