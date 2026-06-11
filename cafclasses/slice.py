from sbnd.general import utils
from pyanalib import pandas_helpers
from .particlegroup import ParticleGroup
from sbnd.detector.volume import *
from sbnd.detector.definitions import *
from sbnd.constants import *
import numpy as np


PREPROCESSORS = {
    "det": lambda slc: slc.cut_is_cont(cut=False),
    "flux": lambda slc: slc.merge_flux_universes(),
}


def run_preprocess(slc, stages=("det",)):
    """Run registered preprocessors (column definition, no row filtering).

    Registered stages: ``det`` (containment columns), ``flux`` (merge flux universes).
    """
    for stage in stages:
        if stage not in PREPROCESSORS:
            raise KeyError(f"unknown preprocess stage: {stage}")
        PREPROCESSORS[stage](slc)


class CAFSlice(ParticleGroup):
    """
    CAF Slice class. Has functions for slc level (no pfps)
    
    Args:
        CAF : CAF dataframe with useful functions
    """
    #-------------------- constructor/rep --------------------#
    def __init__(self,data,**kwargs):
      super().__init__(data,**kwargs)
    def __getitem__(self, item):
      data = super().__getitem__(item)
      return CAFSlice(data
                      ,prism_bins=self.prism_binning
                      ,pot=self.pot
                      ,filter_univ=self.filter_univ)
    def copy(self, deep=True, duplicate_ok=False):
      return CAFSlice(
          self.data.copy(deep),
          pot=self.pot,
          prism_bins=self.prism_binning,
          duplicate_ok=duplicate_ok,
      )
    def load(fname, key='slice', filter_univ=False, cuts=None, categories=None, preprocess=False, **kwargs):
      """
      Load slice CAF from HDF5 file(s).

      Parameters
      ----------
      fname : str or list
          Path(s) to HDF5 file(s).
      key : str or list, optional
          HDF5 key(s) to read (default ``'slice'``).
      filter_univ : bool, optional
          Drop universe-weight columns after load (default False).
      cuts : str or sequence, optional
          Cut names applied per input file before concatenation (default
          None). Names may be short (``'fv'``) or full (``'cut.fv'``); each
          must exist as a boolean ``cut.*`` column. Use e.g. ``PAND_CUTS``
          from ``naming``. Entries named ``'precut'`` are skipped.
      categories : int or sequence of int, optional
          Truth ``event_type`` codes to keep after cuts (default None, load
          all). E.g. ``0`` for contained numu CC, ``[0, 1]`` for contained
          and uncontained signal.
      file_index_offset : int, optional
          Added to per-file ``__ntuple`` index offsets when loading multiple
          chunks sequentially (default 0).
      verbose : bool, optional
          Print load progress and per-cut row counts (default True).
      preprocess : bool, str, or sequence of str, optional
          If truthy, run :func:`run_preprocess` after load. Pass ``True`` for
          default stages ``("det",)``, a stage name (e.g. ``"flux"``) for one
          stage, or e.g. ``("det", "flux")`` to also merge flux universe
          weights into ``Flux_combine`` columns. Note: ``("flux")`` without a
          trailing comma is a ``str``, not a tuple; use ``("flux",)`` or
          ``"flux"``.
      **kwargs
          Passed to ``CAF._load_combined`` / ``CAFSlice`` (e.g. ``ncpu``).
      """
      from .parent import CAF
      from sbnd.general.utils import read_hdf_local
      thiscaf = CAF._load_combined(
        fname, key, read_hdf_local, CAFSlice,
        filter_univ=filter_univ, cuts=cuts, categories=categories, **kwargs
      )
      if preprocess:
        if isinstance(preprocess, str):
          stages = (preprocess,)
        elif isinstance(preprocess, (list, tuple)):
          stages = tuple(preprocess)
        else:
          stages = ("det",)
        run_preprocess(thiscaf, stages=stages)
      return thiscaf
    def merge_flux_universes(self):
      """
      Combine per-component truth ``*_Flux`` universe weights into ``Flux_combine``.

      For each universe index 0..99, multiplies clipped component weights per
      event, drops the source ``*_Flux`` columns, and adds ``Flux_combine`` cols.
      No-op if ``Flux_combine`` already present or no ``*_Flux`` columns exist.
      """
      if not hasattr(self.data.columns, "levels"):
        return self
      if any(col[1] == "Flux_combine" for col in self.data.columns):
        return self

      flux_keys = [col for col in self.data.columns if "_Flux" in col[1]]
      if not flux_keys:
        return self

      nuniv = 100
      n_events = len(self.data)
      flux_weights = np.ones((nuniv, n_events))

      for i in range(nuniv):
        use_cols = [k for k in flux_keys if f"univ_{i}" == k[2]]
        if not use_cols:
          continue
        with np.errstate(invalid="ignore"):
          product = np.clip(self.data.loc[:, use_cols].values, 0, 10).prod(axis=1)
        flux_weights[i] = np.clip(product, 0, 10)

      new_keys = []
      for col in flux_keys[:nuniv]:
        lst = list(col)
        lst[1] = "Flux_combine"
        new_keys.append(utils.col_tuple_to_key(lst))

      flux_src_cols = [
        col for col in self.data.columns
        if "_Flux" in col[1] and col[1] != "Flux_combine"
      ]
      self.data = self.data.drop(columns=flux_src_cols)
      self.add_cols(new_keys, list(flux_weights))
      return self
    #-------------------- setters --------------------#
    def set_mcnu_containment(self,mcnu,suffix=""):
      """
      Set mcnu containment
      """
      return super().set_mcnu_containment(mcnu,'pandora',suffix=suffix)
    #-------------------- cutters --------------------#
    def cut_is_cont(self,cut=False):
      """
      Cut to only contained muons.

      Always stores both reco and truth columns:
      cut.cont / cut.truth.cont: per-TPC (cont_tpc0 | cont_tpc1)
      cut.cont_full / cut.truth.cont_full: detector-wide is_contained
      """
      col_tpc0 = self.get_key(f'mu.pfp.trk.cont_tpc0')[0]
      col_tpc1 = self.get_key(f'mu.pfp.trk.cont_tpc1')[0]
      true_col_tpc0 = self.get_key(f'truth.mu.cont_tpc0')[0]
      true_col_tpc1 = self.get_key(f'truth.mu.cont_tpc1')[0]
      condition_tpc = (self.data.loc[:,col_tpc0] == True) | (self.data.loc[:,col_tpc1] == True)
      true_condition_tpc = (self.data.loc[:,true_col_tpc0] == True) | (self.data.loc[:,true_col_tpc1] == True)

      col = self.get_key(f'mu.pfp.trk.is_contained')[0]
      true_col = self.get_key(f'truth.mu.is_contained')[0]
      condition_full = self.data.loc[:,col] == True
      true_condition_full = self.data.loc[:,true_col] == True

      self.apply_cut('cut.cont', condition_tpc, cut=cut)
      self.apply_cut('cut.truth.cont', true_condition_tpc, cut=False) # Never cut on truth
      self.apply_cut('cut.cont_full', condition_full, cut=False) # Never cut on full containment
      self.apply_cut('cut.truth.cont_full', true_condition_full, cut=False) # Never cut on truth

    def cut_all_cont(self, cut=False):
      """
      Require all score-cut PFP tracks in the slice to be TPC-contained (slc.all_pfps_cont).
      """
      condition = self.data.slc.all_pfps_cont == True
      self.apply_cut('cut.all_cont', condition, cut)

    def cut_cosmic_isclear(self,use_isclearcosmic=False,cut=False):
      """
      Reject slices marked as clear cosmic
      """
      if use_isclearcosmic:
        condition = self.data.slc.is_clear_cosmic != 1
      else:
        condition = np.array([True]*len(self.data))
      self.apply_cut('cut.cosmic_isclear', condition, cut)
    def cut_cosmic_nuscore(self,nu_score=None,cut=False):
      """
      Require a minimum slice nu score
      """
      if nu_score is None:
        condition = np.array([True]*len(self.data))
      else:
        condition = self.data.slc.nu_score > nu_score
      self.apply_cut('cut.cosmic_nuscore', condition, cut)
    def cut_cosmic_fmatch(self,fmatch_score=None,use_opt0=False,cut=False):
      """
      Require a minimum fmatch or opt0-style score
      """
      if fmatch_score is None:
        condition = np.array([True]*len(self.data))
      elif use_opt0 == True:
        condition = self.data.slc.opt0.score > fmatch_score
      elif use_opt0 == 'highlevel':
        # R - H / R
        _score = (self.data.opt0.measPE - self.data.opt0.hypoPE)/self.data.opt0.measPE
        condition = _score > fmatch_score
      elif use_opt0 == 'barycenterFM':
        condition = self.data.slc.barycenterFM.score > fmatch_score
      else:
        raise ValueError('Fmatch score not supported for slice yet')
      self.apply_cut('cut.cosmic_fmatch', condition, cut)
    def cut_cosmic(self,fmatch_score=None,nu_score=None,use_opt0=False,cut=False,use_isclearcosmic=False):
      """
      Backward-compatible wrapper for split cosmic cuts
      """
      pieces = []
      if use_isclearcosmic:
        self.cut_cosmic_isclear(use_isclearcosmic=use_isclearcosmic,cut=False)
        pieces.append(self.data.cut.cosmic_isclear)
      if fmatch_score is not None:
        self.cut_cosmic_fmatch(fmatch_score=fmatch_score,use_opt0=use_opt0,cut=False)
        pieces.append(self.data.cut.cosmic_fmatch)
      if nu_score is not None:
        self.cut_cosmic_nuscore(nu_score=nu_score,cut=False)
        pieces.append(self.data.cut.cosmic_nuscore)
      if len(pieces) == 0:
        condition = np.array([True]*len(self.data))
      else:
        condition = pieces[0]
        for cond in pieces[1:]:
          condition &= cond
      self.apply_cut('cut.cosmic', condition, cut)
      self.apply_cut('cut.truth.cosmic', (self.data.slc.truth.pdg == -1) | (self.data.slc.truth.pdg.isna()), cut=False) # Never cut on truth
    def cut_flashmatch(self,cut=False,method='barycenterFM'):
      """
      Find each slice in an event and mark the one with the highest score as True
      """
      if method == 'opt0':
        scores = self.data.slc.opt0.score
      elif method == 'barycenterFM':
        scores = self.data.slc.barycenterFM.score
      else:
        raise ValueError(f'Invalid method: {method}')
      inds = scores.groupby(level=self.data.index.names[:-1]).idxmax(skipna=True) # ignore last level (slice)
      # Drop any NaN results (groups where all values were NA)
      inds = inds.dropna()
      
      # Set condition
      condition = pd.Series(False, index=self.data.index)
      condition.loc[inds.values] = True

      self.apply_cut('cut.flashmatch', condition, cut)
      
    def cut_fv(self,cut=False):
      """
      Cut to only in fv
      """
      self.apply_cut('cut.fv', self.data.vertex.fv == True, cut=cut)
      self.apply_cut('cut.truth.fv', self.data.slc.truth.fv == True, cut=False) # Never cut on truth
    def cut_muon(self,cut=False,min_ke=0.1,suffix=""):
      """
      Cut to only muon
      """
      cols = self.get_key([f'mu{suffix}.pfp.trk.P.p_muon',f'has_muon{suffix}'])
      min_p = utils.calc_momentum_from_ke(min_ke,PDG_TO_MASS_MAP[13])
      condition = (self.data.loc[:,cols[1]] == True) & (self.data.loc[:,cols[0]] > min_p)
      self.apply_cut('cut.muon', condition, cut)
      if min_ke == 0.1:
        self.apply_cut('cut.truth.muon', self.data.slc.truth.nmu_100MeV > 0, cut=False) # Never cut on truth
      elif min_ke == 0.027:
        self.apply_cut('cut.truth.muon', self.data.slc.truth.nmu_27MeV > 0, cut=False) # Never cut on truth
      else:
        raise ValueError(f'Invalid min_ke: {min_ke}')
    def cut_lowz(self,cut=False,z_max=6,include_start=True,suffix=""):
      """
      Cut if the start or end muon point is within
      """
      cols = self.get_key([f'mu{suffix}.pfp.trk.start.z',f'mu{suffix}.pfp.trk.end.z'])
      if include_start:
        condition = ~((self.data.loc[:,cols[0]] < z_max) | (self.data.loc[:,cols[1]] < z_max))
      else:
        condition = self.data.loc[:,cols[1]] > z_max
      self.apply_cut('cut.lowz', condition, cut)
    def cut_flashpe(self,cut=False,min_flashpe=2000,prescale=1.):
      """ Cut flash pe < min_flashpe. Prescale the flash pe by prescale factor.
      """
      condition = self.data.slc.barycenterFM.flashPEs.values*prescale > min_flashpe
      self.apply_cut('cut.flashpe', condition, cut)
    def cut_all(self,cont=True,cut=False,mode='reco'):
      """
      add a column that is true if all cuts are true
      """
      #print('WARNING: Ensure all cuts in this function are correct')
      if mode == 'reco':
        condition = self.data.cut.fv\
          & self.data.cut.muon\
          & self.data.cut.flashmatch\
          & self.data.cut.flashpe
        if cont:
          condition &= self.data.cut.cont
          condition &= self.data.cut.all_cont
        else:
          condition &= self.data.cut.lowz
      elif mode == 'truth':
        categories = [0] if cont else [0,1]
        condition = self.data.truth.event_type.isin(categories)
      else:
        raise ValueError(f'Invalid mode: {mode}')
      if cut:
        self.data = self.data[condition]
        return
      self.apply_cut('cut.all', condition, cut)
    #-------------------- adders --------------------#
    def add_isbroken(self):
      """
      Add is broken boolean. True if the track is broken into two or more tracks
      """
      pass
      
    def add_is_tpc_contained(self,ismc=False):
      """
      Add is contained boolean. True if the track is contained in the active volume
      """
      keys = [
        'mu.pfp.trk.cont_tpc0',
        'mu.pfp.trk.cont_tpc1',
        'truth.mu.cont_tpc0',
        'truth.mu.cont_tpc1',
        'mu.pfp.trk.truth.p.cont_tpc0',
        'mu.pfp.trk.truth.p.cont_tpc1',
      ]
      cols_start = self.get_key([f'mu.pfp.trk.start.x',f'mu.pfp.trk.start.y',f'mu.pfp.trk.start.z'])
      cols_end = self.get_key([f'mu.pfp.trk.end.x',f'mu.pfp.trk.end.y',f'mu.pfp.trk.end.z'])
      start = self.data.loc[:, cols_start]
      end = self.data.loc[:, cols_end]
      veto = enters(start, end, NOT_FV_HIGH_Z)

      values = [
        involume(start, volume=TPC0_BUFFER) & involume(end, volume=TPC0_BUFFER) & ~veto,
        involume(start, volume=TPC1_BUFFER) & involume(end, volume=TPC1_BUFFER) & ~veto,
        pd.Series(np.zeros(len(self.data), dtype=bool), index=self.data.index),
        pd.Series(np.zeros(len(self.data), dtype=bool), index=self.data.index),
        pd.Series(np.zeros(len(self.data), dtype=bool), index=self.data.index),
        pd.Series(np.zeros(len(self.data), dtype=bool), index=self.data.index),
      ]
      if ismc:
        cols_truth_start = self.get_key([f'truth.mu.start.x',f'truth.mu.start.y',f'truth.mu.start.z'])
        cols_truth_end = self.get_key([f'truth.mu.end.x',f'truth.mu.end.y',f'truth.mu.end.z'])
        cols_truth_match_start = self.get_key([f'mu.pfp.trk.truth.p.start.x',f'mu.pfp.trk.truth.p.start.y',f'mu.pfp.trk.truth.p.start.z'])
        cols_truth_match_end = self.get_key([f'mu.pfp.trk.truth.p.end.x',f'mu.pfp.trk.truth.p.end.y',f'mu.pfp.trk.truth.p.end.z'])
        truth_start = self.data.loc[:, cols_truth_start]
        truth_end = self.data.loc[:, cols_truth_end]
        match_start = self.data.loc[:, cols_truth_match_start]
        match_end = self.data.loc[:, cols_truth_match_end]
        truth_veto = enters(truth_start, truth_end, NOT_FV_HIGH_Z)
        match_veto = enters(match_start, match_end, NOT_FV_HIGH_Z)
        values[2:] = [
          involume(truth_start, volume=TPC0_BUFFER) & involume(truth_end, volume=TPC0_BUFFER) & ~truth_veto,
          involume(truth_start, volume=TPC1_BUFFER) & involume(truth_end, volume=TPC1_BUFFER) & ~truth_veto,
          involume(match_start, volume=TPC0_BUFFER) & involume(match_end, volume=TPC0_BUFFER) & ~match_veto,
          involume(match_start, volume=TPC1_BUFFER) & involume(match_end, volume=TPC1_BUFFER) & ~match_veto,
        ]
      self.add_cols(keys,values,fill=False,overwrite=True)
    def add_has_muon(self,suffix=""):
      """
      Check if there is a muon
      """

      col = self.get_key(f'mu{suffix}.pfp.trk.is_muon')
      # Set keys, conditions and values
      keys = [f'has_muon{suffix}']
      #Get slices with muons
      mask = (self.data.loc[:,col] == True).values.flatten()
      self.add_cols(keys,mask,fill=False)
    def add_2d_binning(self, costheta_bins=None, momentum_bins=None, momentum_bins_2d=None,
                              include_truth=True, include_reco=True):
      """
      Add 2D muon phase space binning for truth and/or reco.
      Uses costheta dependent momentum binning when momentum_bins_2d is provided.
      """
      from sbnd.numu.numu_constants import (
        DIFF_COSTHETA_BINS,
        DIFF_MOMENTUM_BINS_2D,
        N_DIFF_COSTHETA_BINS,
      )
      if costheta_bins is None:
        costheta_bins = DIFF_COSTHETA_BINS
      if momentum_bins_2d is None:
        momentum_bins_2d = DIFF_MOMENTUM_BINS_2D

      costheta_bins = np.asarray(costheta_bins)
      momentum_bins_2d = np.asarray(momentum_bins_2d)

      n_costheta_bins = len(costheta_bins) - 1
      if n_costheta_bins != N_DIFF_COSTHETA_BINS:
        raise ValueError(f'n_costheta_bins {n_costheta_bins} does not match N_DIFF_COSTHETA_BINS {N_DIFF_COSTHETA_BINS}')
      if momentum_bins_2d.shape[0] != n_costheta_bins:
        raise ValueError(f'momentum_bins_2d has {momentum_bins_2d.shape[0]} rows, expected {n_costheta_bins}')

      # Truth binning
      if include_truth and self.check_key('truth.mu.dir.z') and self.check_key('truth.mu.totp'):
        self.assign_bins(costheta_bins, 'truth.mu.dir.z', assign_key='true_bin.costheta')
        # Ensure momentum column exists even when no rows pass a cbin mask.
        self.add_key(['true_bin.momentum'], fill=-1.0)
        for cbin in range(n_costheta_bins):
          mask = (self.data.true_bin.costheta.values.astype(float) == float(cbin))
          if not np.any(mask):
            continue
          self.assign_bins(
            momentum_bins_2d[cbin],
            'truth.mu.totp',
            assign_key='true_bin.momentum',
            mask=mask,
          )
        cols = self.get_key(['true_bin.costheta','true_bin.momentum'])
        mask = self.data.loc[:,cols].values.astype(float) >= 0
        mask = np.all(mask, axis=1)
        differential_bins = (
          self.data.true_bin.costheta.values.astype(float)
          + self.data.true_bin.momentum.values.astype(float) * n_costheta_bins
        )
        self.add_cols('true_bin.differential', differential_bins[mask], conditions=mask, fill=-1.0)

      # Reco binning
      if include_reco and self.check_key('mu.pfp.trk.costheta') and self.check_key('mu.pfp.trk.P.p_muon'):
        self.assign_bins(costheta_bins, 'mu.pfp.trk.costheta', assign_key='bin.costheta')
        # Ensure momentum column exists even when no rows pass a cbin mask.
        self.add_key(['bin.momentum'], fill=-1.0)
        for cbin in range(n_costheta_bins):
          mask = (self.data.bin.costheta.values.astype(float) == float(cbin))
          if not np.any(mask):
            continue
          self.assign_bins(
            momentum_bins_2d[cbin],
            'mu.pfp.trk.P.p_muon',
            assign_key='bin.momentum',
            mask=mask,
          )
        cols = self.get_key(['bin.costheta','bin.momentum'])
        mask = self.data.loc[:,cols].values.astype(float) >= 0
        mask = np.all(mask, axis=1)
        differential_bins = (
          self.data.bin.costheta.values.astype(float)
          + self.data.bin.momentum.values.astype(float) * n_costheta_bins
        )
        self.add_cols('bin.differential', differential_bins[mask], conditions=mask, fill=-1.0)
    def add_in_av(self):
      """
      Add containment 1 or 0 for each pfp
      """
      #Set keys, values, conditions
      keys = [
        'slc.truth.av',
        'vertex.av'
      ]
      values = [
        involume(self.data.truth.position,volume=AV),
        involume(self.data.slc.vertex,volume=AV)
      ]
      self.add_cols(keys,values,fill=False)
    def add_phi(self,ismc=False):
      """
      CORRECT
      Add phi column
      Angle of the start direction of the track in the x-y plane
      """
      keys = [
        f'mu.pfp.trk.phi'
      ]
      values = [
        np.arctan2(self.data.mu.pfp.trk.dir.x, self.data.mu.pfp.trk.dir.y)
      ]
      if ismc:
        totp = self.data.truth.mu.totp
        keys += [
          'mu.pfp.trk.truth.p.phi',
          'truth.mu.phi'
        ]
        values += [
          np.arctan2(self.data.mu.pfp.trk.truth.p.genp.x/totp, self.data.mu.pfp.trk.truth.p.genp.y/totp),
          np.arctan2(self.data.truth.mu.dir.x, self.data.truth.mu.dir.y),
        ]
      self.add_cols(keys,values,fill=np.nan)

    def add_theta(self,ismc=False):
      """
      Add theta column
      """
      keys = [
        f'mu.pfp.trk.theta',
      ]
      values = [
        np.arccos(self.data.mu.pfp.trk.dir.z)
      ]
      if ismc:
        keys += [
          'mu.pfp.trk.truth.p.theta',
          'truth.mu.theta'
        ]
        totp = self.data.truth.mu.totp
        values += [
          np.arccos(self.data.mu.pfp.trk.truth.p.genp.z/totp),
          np.arccos(self.data.truth.mu.dir.z)
        ]
      self.add_cols(keys,values,fill=np.nan)
    def add_in_fv(self):
      """
      Add containment 1 or 0 for each pfp
      """
      #Set keys, values, conditions
      keys = [
        'slc.truth.fv',
        'vertex.fv'
      ]
      values = [
        involume_FV(self.data.truth.position),
        involume_FV(self.data.slc.vertex)
      ]
      self.add_cols(keys,values,fill=False) 
   
    def add_event_type(self,min_ke=0.1):
      """
      Add event type from genie type map. True event type

      Parameters
      ----------
      min_ke : float
        Minimum kinetic energy to be considered a muon [GeV]
      """
      iscc = self.data.slc.truth.iscc == 1
      isnumu = self.data.slc.truth.pdg == 14
      isanumu = self.data.slc.truth.pdg == -14
      isnue = self.data.slc.truth.pdg == 12
      isanue = self.data.slc.truth.pdg == -12
      iscosmic = (self.data.slc.truth.pdg == -1) | (self.data.slc.truth.pdg.isna()) #masked to 1
      istrueav = self.data.slc.truth.av #filled to bool
      istruefv = self.data.slc.truth.fv #filled to bool
      if min_ke == 0.1:
        ismuon = self.data.slc.truth.nmu_100MeV > 0 #KE requirement
      elif min_ke == 0.027:
        ismuon = self.data.slc.truth.nmu_27MeV > 0 #KE requirement
      else:
        raise ValueError(f'Invalid min_ke: {min_ke}')
      # Contained in either tpc0 or tpc1
      cols = self.get_key([
        'mu.pfp.trk.truth.p.cont_tpc0',
        'mu.pfp.trk.truth.p.cont_tpc1'
      ])
      iscont = (
        (self.data.loc[:,cols[0]] == 1) | (self.data.loc[:,cols[0]] == True) |
        (self.data.loc[:,cols[1]] == 1) | (self.data.loc[:,cols[1]] == True)
      ) # contained in either tpc0 or tpc1, but not both
      
      #aggregate true types
      isnumuccav = iscc & (isnumu | isanumu) & istrueav & ~iscosmic #numu cc av
      isnumuccfv = iscc & (isnumu | isanumu) & istruefv & ~iscosmic #numu cc fv
      isnumuccfv_cont = isnumuccfv & iscont & ismuon
      isnumuccfv_uncont = isnumuccfv & ~iscont & ismuon
      isnumuccoops = isnumuccav & ~isnumuccfv_cont & ~isnumuccfv_uncont #numu cc out of phase space (oops)
      isnueccav = iscc & (isnue | isanue) & istrueav & ~iscosmic #nue cc av
      isncav = ~iscc & istrueav & ~iscosmic #nc av
      iscosmicav = istrueav & iscosmic #cosmic av
      isdirt = ~istrueav & ~iscosmic #dirt
      
      #Consider adding exclusive channels??
      
      #Add to slice
      keys = [
        'truth.event_type'
      ]
      self.add_key(keys,fill=-1)
      cols = pandas_helpers.getcolumns(keys,depth=self.key_length())
      #assert total == len(self.data.loc[:,cols[0]]), f'total categorized ({total:,}) does not match the number of slices ({len(self.data.loc[:,cols[0]]):,}). Diff = {len(self.data.loc[:,cols[0]]) - total:,}.'
      self.data.loc[isnumuccfv_cont,cols[0]] = 0 #numu cc (contained)
      self.data.loc[isnumuccfv_uncont,cols[0]] = 1 #numu cc (uncontained)
      self.data.loc[isnumuccoops,cols[0]] = 2 #numu cc oops
      self.data.loc[isdirt,cols[0]] = 3 #dirt
      self.data.loc[isnueccav,cols[0]] = 4 #nue cc
      self.data.loc[isncav,cols[0]] = 5 #nc
      self.data.loc[iscosmic,cols[0]] = 6 #cosmic
      #self.data.loc[isnumuccfv_cont | isnumuccfv_uncont,cols[0]] = 2 #numu cc (either)
      #the rest are unknown
    
    #-------------------- getters --------------------#
    def get_pur_eff_f1(self,mcnu,cuts,categories=[0,1]):
      """
      Get pur eff f1
      """
      return super().get_pur_eff_f1('pandora',mcnu,cuts,categories)
    #-------------------- plotters --------------------#
      
       
       
        

      
  