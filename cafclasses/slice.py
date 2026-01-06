from sbnd.general import utils
from pyanalib import pandas_helpers
from .particlegroup import ParticleGroup
from sbnd.detector.volume import *
from sbnd.constants import *


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
    def copy(self,deep=True):
      return CAFSlice(self.data.copy(deep),pot=self.pot,prism_bins=self.prism_binning)
    def load(fname,key='slice',filter_univ=True,**kwargs):
      if isinstance(key,list):
        if len(key) == 0:
          raise ValueError(f'No keys provided for {fname}')
        for i,k in enumerate(key):
          if i == 0:
            thisslc = CAFSlice(pd.read_hdf(fname,key=k),**kwargs)
          else:
            thisslc.combine(CAFSlice(pd.read_hdf(fname,key=k),**kwargs))
      elif isinstance(key,str):
        thisslc = CAFSlice(pd.read_hdf(fname,key=key),**kwargs)
      else:
        raise ValueError(f'Invalid key: {key}')
      if filter_univ:
        from .parent import filter_univ_columns
        df = filter_univ_columns(thisslc.data)
        thisslc = CAFSlice(df,**kwargs)
      return thisslc
    #-------------------- setters --------------------#
    def set_mcnu_containment(self,mcnu,suffix=""):
      """
      Set mcnu containment
      """
      return super().set_mcnu_containment(mcnu,'pandora',suffix=suffix)
    #-------------------- cutters --------------------#
    def cut_is_cont(self,cut=True,suffix=""):
      """
      Cut to only contained
      """
      col = self.get_key(f'mu{suffix}.pfp.trk.is_contained')[0]
      condition = self.data.loc[:,col] == True
      self.apply_cut('cut.cont', condition, cut=cut)
      self.apply_cut('cut.truth.cont', self.data.slc.truth.mu.is_contained == True, cut=False) # Never cut on truth
    def cut_has_nuscore(self,cut=True):
      """
      Cut those that don't have a nu score. They didn't get any reco
      """
      condition = self.data.nu_score >= 0
      self.apply_cut('cut.has_nuscore', condition, cut)
    
    def cut_cosmic(self,crumbs_score=None,fmatch_score=None,nu_score=None,use_opt0=False,cut=False,use_isclearcosmic=False):
      """
      Cut to only cosmic tracks
      """
      #print(f'Number of clear cosmic: {len(self.data[self.data.slc.is_clear_cosmic == 1])}')
      if use_isclearcosmic:
        #print('Cutting is clear cosmic')
        condition = self.data.slc.is_clear_cosmic != 1
      else:
        #print('Skipping is clear cosmic cut')
        condition = np.array([True]*len(self.data))
      if crumbs_score is not None:
          #print(f'Cutting crumbs score: {crumbs_score}')
          raise ValueError('Crumbs score not supported for slice yet')
      if fmatch_score is not None:
          if use_opt0 == True:
              #print(f'Cutting opt0 score: {fmatch_score}')
              condition &= self.data.slc.opt0.score > fmatch_score
          elif use_opt0 == 'highlevel':
              #print(f'Cutting highlevel opt0 score: {fmatch_score} (R-H/R)')
              # R - H / R
              _score = (self.data.opt0.measPE - self.data.opt0.hypoPE)/self.data.opt0.measPE
              condition &= _score > fmatch_score    
          elif use_opt0 == 'barycenterFM':
              #print(f'Cutting barycenterFM score: {fmatch_score}')
              condition &= self.data.slc.barycenterFM.score > fmatch_score
          else:
              #print(f'Cutting fmatch score: {fmatch_score}')
              raise ValueError('Fmatch score not supported for slice yet')
              #condition &= self.data.fmatch.score < fmatch_score
      if nu_score is not None:
          #print(f'Cutting nu score: {nu_score}')
          condition &= self.data.slc.nu_score > nu_score

      self.apply_cut('cut.cosmic', condition, cut)
      self.apply_cut('cut.truth.cosmic', (self.data.slc.truth.pdg == -1) | (self.data.slc.truth.pdg.isna()), cut=False) # Never cut on truth
    def cut_flashmatch(self,cut=False,use_isclearcosmic=True,method='opt0'):
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
      
      if use_isclearcosmic:
        condition &= self.data.slc.is_clear_cosmic != 1
      self.apply_cut('cut.flashmatch', condition, cut)
      
    def cut_fv(self,volume=FV,cut=False):
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
    def cut_all(self,cont=False,cut=False,mode='reco',categories=[0,1]):
      """
      add a column that is true if all cuts are true
      """
      #print('WARNING: Ensure all cuts in this function are correct')
      if mode == 'reco':
        condition = self.data.cut.fv\
          & self.data.cut.cosmic\
          & self.data.cut.muon\
          & self.data.cut.flashmatch
        if cont:
          condition &= self.data.cut.cont
        else:
          condition &= self.data.cut.lowz
      elif mode == 'truth':
        condition = self.data.truth.event_type.isin(categories)
      else:
        raise ValueError(f'Invalid mode: {mode}')
      if cut:
        self.data = self.data[condition]
        return
      self.apply_cut('cut.all', condition, cut)
    #-------------------- adders --------------------#
    def add_track_flipping(self,suffix=""):
      """
      Add track flipping boolean. True if start and end are correctly matched
      """
      return super().add_track_flipping('pandora',suffix=suffix)
    def add_has_muon(self,suffix=""):
      """
      Check if there is a muon
      """

      col = self.get_key(f'mu{suffix}.pfp.trk.is_muon{suffix}')
      # Set keys, conditions and values
      keys = [f'has_muon{suffix}']
      #Get slices with muons
      inds = self.data[self.data.loc[:,col] == True].index
      #Set condition
      condition = pd.Series(False,index=self.data.index)
      condition.loc[inds] = True
      self.add_cols(keys,[True],conditions=condition,fill=False)
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
        involume(self.data.truth.position,volume=FV),
        involume(self.data.slc.vertex,volume=FV)
      ]
      self.add_cols(keys,values,fill=False) 
   
    def add_event_type(self,min_ke=0.1,suffix=""):
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
      cols = self.get_key([f'mu{suffix}.pfp.trk.truth.p.contained'])
      iscont = (self.data.loc[:,cols[0]] ==1) | (self.data.loc[:,cols[0]] == True) #contained, break down signal and background
      
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
      
       
       
        
      
      
      
      
  