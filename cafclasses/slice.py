from sbnd.general import utils
from pyanalib import panda_helpers
from .parent import CAF
from sbnd.volume import *


class CAFSlice(CAF):
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
                      ,pot=self.pot)
    def copy(self,deep=True):
      return CAFSlice(self.data.copy(deep),pot=self.pot,prism_bins=self.prism_binning)
    def load(fname,key='slice',**kwargs):
      df = pd.read_hdf(fname,key=key,**kwargs)
      return CAFSlice(df,**kwargs)
    #-------------------- cutters --------------------#
    def cut_has_nuscore(self,cut=True):
      """
      Cut those that don't have a nu score. They didn't get any reco
      """
      keys = [
        'cut.has_nuscore'
      ]
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[:,cols[0]] = self.data.nu_score >= 0
      if cut:
        self.data = self.data[self.data.cut.has_nuscore] 
    
    def cut_cosmic(self,crumbs_score=None,fmatch_score=None,nu_score=None,cut=False):
      """
      Cut to only cosmic tracks
      """
      slc_df = self.data[self.data.is_clear_cosmic != 1] #removed from nu score = -1
      if crumbs_score is not None:
        slc_df = slc_df[slc_df.crumbs_result.bestscore > crumbs_score]
      if fmatch_score is not None:
        slc_df = slc_df[slc_df.fmatch.score < fmatch_score]
      if nu_score is not None:
        slc_df = slc_df[slc_df.nu_score > nu_score]
      inds = slc_df.index.values
      #add the boolean to the data
      keys = [
        'cut.cosmic'
      ]
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[:,cols[0]] = False
      self.data.loc[inds,cols[0]] = True
      if cut:
        self.data = self.data[self.data.cuts.cosmic]
    def cut_fv(self,volume=FV,cut=False):
      """
      Cut to only in fv
      """
      keys = [
        'cut.fv',
        'truth.fv' #truth fv but not really a cut
      ]
      self.add_key(keys,fill=False)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[:,cols[0]] = involume(self.data.vertex, volume=volume) #based on pandora vertex
      self.data.loc[:,cols[1]] = involume(self.data.truth.position,volume=volume) #based on true vertex
      if cut: #add handle to cut on truth??
        self.data = self.data[self.data.cuts.fv]
    def cut_muon(self,cut=False):
      """
      Cut to only muon
      """
      keys = [
        'cut.has_muon'
      ]
      self.add_key(keys,fill=False)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[:,cols[0]] = self.data.has_muon
      if cut:
        self.data = self.data[self.data.cuts.muon]
    def cut_trk(self,cut=False):
      """
      Cut to only tracks
      """
      keys = [
        'cut.trk'
      ]
      self.add_key(keys,fill=False)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[:,cols[0]] = self.data.has_trk
      if cut:
        self.data = self.data[self.data.cuts.trk]
    def cut_all(self,cut=False):
      """
      add a column that is true if all cuts are true
      """
      keys = [
        'cut.total'
      ]
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      #vectorize this
      self.data.loc[:,cols[0]] = self.data.cut.fv\
        & self.data.cut.cosmic\
        & self.data.cut.has_muon
      if cut:
        self.data = self.data[self.data.cuts.all]
    #-------------------- adders --------------------#
    def add_has_muon(self,pfp):
      """
      Check if there is a muon
      """
      #has to be a track to be a muon (for now)
      muons = pfp.data[pfp.data.bestpdg == 13]
      inds = utils.get_sub_inds_from_inds(muons.index.values,self.data.index.values,self.index_depth)
      #print(inds)
      #now add the boolean to the data
      keys = [
        'has_muon'
      ]
      self.add_key(keys,fill=False)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[inds,cols[0]] = True
    def add_has_trk(self,pfp):
      """
      Check if there is a track
      """
      #has to be a track to be a muon (for now)
      trks = pfp.data[pfp.data.semantic_type == 0]
      inds = utils.get_sub_inds_from_inds(trks.index.values,self.data.index.values,self.index_depth)
      #now add the boolean to the data
      keys = [
        'has_trk'
      ]
      self.add_key(keys,fill=False)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[inds,cols[0]] = True
    def add_in_av(self):
      """
      Add containment 1 or 0 for each pfp
      """
      keys = [
        'truth.av',
        'vertex.av'
      ]
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[:,cols[0]] = involume(self.data.truth.position,volume=AV)
      self.data.loc[:,cols[1]] = involume(self.data.vertex,volume=AV)
    def add_shws_trks(self,pfp,energy_threshold=0.):
      """
      Count the number of showers and tracks
      
      energy_threshold: minimum energy to be observable (this probably needs some extra thinking)
      """
      #Get slice indices in common
      inds = utils.get_sub_inds_from_inds(pfp.data.index.values,self.data.index.values,self.index_depth)
      
      #Find number of showers and tracks per slice
      trk_counts = (pfp.data.semantic_type == 0).groupby(level=[0,1,2]).sum()
      shw_counts = (pfp.data.semantic_type == 1).groupby(level=[0,1,2]).sum()
      
      #Find true number of showers and tracks per slice
      observable = (pfp.data.trk.truth.p.startE-pfp.data.trk.truth.p.endE) >= energy_threshold
      true_trk_counts = ((abs(pfp.data.trk.truth.p.pdg) == 13) & observable).groupby(level=[0,1,2]).sum()\
                  + ((abs(pfp.data.trk.truth.p.pdg) == 211) & observable).groupby(level=[0,1,2]).sum()\
                  + ((abs(pfp.data.trk.truth.p.pdg) == 321) & observable).groupby(level=[0,1,2]).sum()\
                  + ((abs(pfp.data.trk.truth.p.pdg) == 2212) & observable).groupby(level=[0,1,2]).sum()
      true_shw_counts = ((abs(pfp.data.trk.truth.p.pdg) == 11) & observable).groupby(level=[0,1,2]).sum()\
                  + ((abs(pfp.data.trk.truth.p.pdg) == 22) & observable).groupby(level=[0,1,2]).sum()\
                  + 2*((abs(pfp.data.trk.truth.p.pdg) == 111) & observable).groupby(level=[0,1,2]).sum()
      
      #Add to slice
      keys = [
        'nshw',
        'ntrk',
        'npfp',
        'truth.nshw',
        'truth.ntrk',
        'truth.npfp'
      ]
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[:,cols[0]] = shw_counts
      self.data.loc[:,cols[1]] = trk_counts
      self.data.loc[:,cols[2]] = shw_counts + trk_counts
      self.data.loc[inds,cols[3]] = true_shw_counts
      self.data.loc[inds,cols[4]] = true_trk_counts
      self.data.loc[inds,cols[5]] = true_shw_counts + true_trk_counts
    def add_tot_visE(self):
      """
      Find total visible energy
      """
      visible = np.max([self.data.truth.plane.I0.I0.visE.values
        ,self.data.truth.plane.I0.I1.visE.values
        ,self.data.truth.plane.I0.I2.visE.values],axis=0)
      
      #Add to slice
      keys = [
        "truth.visE"
      ]
      self.add_key(keys)
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[:,cols[0]] = visible  
    
    def add_event_type(self):
      """
      Add event type from genie type map. True event type
      """
      iscc = self.data.truth.iscc == 1
      isnumu = self.data.truth.pdg == 14
      isanumu = self.data.truth.pdg == -14
      isnue = self.data.truth.pdg == 12
      isanue = self.data.truth.pdg == -12
      iscosmic = self.data.truth.pdg == -1 #masked to 1
      istrueav = self.data.truth.av #filled to bool
      
      #aggregate true types
      isnumuccav = iscc & (isnumu | isanumu) & istrueav & ~iscosmic #numu cc av
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
      cols = panda_helpers.getcolumns(keys,depth=self.key_length())
      self.data.loc[isnumuccav,cols[0]] = 0 #numu cc
      self.data.loc[isncav,cols[0]] = 1 #nc
      self.data.loc[isnueccav,cols[0]] = 2 #nue cc
      self.data.loc[iscosmic,cols[0]] = 3 #cosmic
      self.data.loc[isdirt,cols[0]] = 4 #dirt
      #the rest are unknown
      
    #-------------------- assigners --------------------#
    #-------------------- getters --------------------#
    def get_numevents(self):
      """
      Get number of events from gen weights
      """
      if self.check_key('genweight'):
        return self.data.genweight.sum()
      raise ValueError('genweight not in dataframe. Run scale_to_pot first')
    def get_pur_eff_f1(self,cuts=[]):
      """
      Get purity, efficiency, and f1 score from list of cuts applied
      
      Purity: signal / total events (precision)
      Efficiency: remaining signal / initial signal (recall)
      f1: 2*eff*purity/(eff+purity)
      
      returns list of purs,effs,f1s for each cut in order
      """
      
      pur = np.zeros(len(cuts)+1)
      eff = np.zeros(len(cuts)+1)
      f1 = np.zeros(len(cuts)+1)
      
      slc_cut_df = self.copy().data #apply cuts to a copy
      init_signal = slc_cut_df[slc_cut_df.truth.event_type == 0].genweight.sum() #initial number of events
      
      pur[0] = init_signal/slc_cut_df.genweight.sum()
      f1[0] = 1
      
      if len(cuts) != 0:
        assert isinstance(cuts,list), 'cuts must be a list'
        assert isinstance(cuts[0],str), 'cuts must be a list of strings'
        for i,cut in enumerate(cuts):
          slc_cut_df = slc_cut_df[slc_cut_df.cut[cut]] #apply cut
          pur[i+1] = slc_cut_df[slc_cut_df.truth.event_type == 0].genweight.sum()/slc_cut_df.genweight.sum() #purity
          eff[i+1] = slc_cut_df[slc_cut_df.truth.event_type == 0].genweight.sum()/init_signal
          f1[i+1] = 2*eff[i+1]*pur[i+1]/(eff[i+1]+pur[i+1])
      return pur,eff,f1
        
        
      
      
      
      
  