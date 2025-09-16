from .parent import CAF
from pyanalib import pandas_helpers
import numpy as np
import pandas as pd
from .nu import NU

class ParticleGroup(CAF):
    """
    Interface class to interactions from ML Reco and slices from Pandora
    """
    #-------------------- constructor/rep --------------------#
    def __init__(self,data,**kwargs):
      super().__init__(data,**kwargs)
    #-------------------- cutters --------------------#
    #-------------------- assigners --------------------#
    #-------------------- adders --------------------#
    #-------------------- setters --------------------#
    def set_mcnu_containment(self,mcnu,algo):
        """
        Since mcnu has not idea about particle propagation in g4, we
        need to set it from the slice

        Returns
        -------
        mcnu : NU
        mcnu with the containment set
        algo: str
        Algorithm to use for containment. Either 'pandora' or 'spine'
        """
        assert algo in ['pandora','spine'], f'Invalid algo: {algo}'
        #Get slice copy that matches the mcnu indices
        pgrp_df = self.data.copy()
        pgrp_df.dropna(subset=self.get_key('truth.ind'),inplace=True) # Get rid of any rows without truth info
        pgrp_df.index = pgrp_df.index.droplevel(-1) #Drop slice index
        pgrp_df = pgrp_df.set_index(pgrp_df.truth.ind,append=True) #Use nu index
        pgrp_df.index.names = [None] * len(pgrp_df.index.names) #Drop names of indices

        #Drop any indices that are not in the mcnu index
        pgrp_df = pgrp_df[pgrp_df.index.isin(mcnu.data.index)]
        #Drop duplicated pgrp_df indices
        pgrp_df = pgrp_df.loc[~pgrp_df.index.duplicated(keep='first')]

        #Add to mcnu
        keys = [f'mu.is_{algo}_contained']
        mcnu.add_key(keys,fill=False) #Fill to false by default
        cols = pandas_helpers.getcolumns(keys,depth=mcnu.key_length())
        if algo == 'pandora':
            is_contained = (pgrp_df.mu.pfp.trk.truth.p.contained == 1) | (pgrp_df.mu.pfp.trk.truth.p.contained == True)
        elif algo == 'spine':
            is_contained = (pgrp_df.mu.tpart.is_contained == 1) | (pgrp_df.mu.tpart.is_contained == True)
        mcnu.data.loc[pgrp_df.index,cols] = is_contained
        return mcnu
    #-------------------- getters ---------------------#
    def get_pur_eff_f1(self,algo,mcnu,cuts,categories=[0,1]):
        """
        Get purity, efficiency, and f1 score from list of cuts applied

        Parameters
        ----------
        algo: str
          Algorithm to use for event type. Either 'pandora' or 'spine'
        mcnu: mcnu object containing truth information
        cuts: list of cuts to apply
        categories: list of categories that are signal

        Returns
        -------
        Purity: signal / total events (precision)
        Efficiency: remaining signal / initial signal (recall)
        f1: 2*eff*purity/(eff+purity)
        """
        print(f'categories: {categories}')
        assert algo in ['pandora','spine'], f'Invalid algo: {algo}'
        algo_event_col = mcnu.get_key(f'event_type_{algo}')

        
        pur = np.zeros(len(cuts)+1)
        eff = np.zeros(len(cuts)+1)
        f1 = np.zeros(len(cuts)+1)
        
        #Set indices to be the truth indices
        partgrp_cut_df = self.copy().data #apply cuts to a copy
        partgrp_cut_df = partgrp_cut_df.droplevel(-1).set_index(partgrp_cut_df.truth.ind,append=True)

        #Get truth object for efficiency calculations
        mcnu.data.dropna(subset=mcnu.get_key('ind'),inplace=True)
        mcnu.data.droplevel(-1).set_index(mcnu.data.ind,append=True)
        mcnu.data.drop_duplicates(inplace=True)
        mcnu_df = mcnu.data

        #Drop names of indices
        mcnu_df.index.names = [None] * len(mcnu_df.index.names)
        partgrp_cut_df.index.names = [None] * len(partgrp_cut_df.index.names)

        init_truth = len(mcnu_df[np.isin(mcnu_df[algo_event_col],categories)].index.drop_duplicates())
        init_truth_total = len(mcnu_df)
        init_signal = len(partgrp_cut_df[np.isin(partgrp_cut_df.truth.event_type,categories)]) #initial number of events
        init_total = len(partgrp_cut_df)

        pur[0] = init_signal/init_total
        eff[0] = 1
        f1[0] = 1
        
        #print(f'init_truth: {init_truth}, init_truth_total: {init_truth_total}, init_signal: {init_signal}, init_total: {init_total}')

        if len(cuts) != 0:
            assert isinstance(cuts,list), 'cuts must be a list'
            assert isinstance(cuts[0],str), 'cuts must be a list of strings'
            for i,cut in enumerate(cuts):
                partgrp_cut_df = partgrp_cut_df[partgrp_cut_df.cut[cut]] #apply cut
                # Find indices of partgrp_cut_df that are in mcnu_df
                partgrp_inds = partgrp_cut_df.index.drop_duplicates()
                mcnu_inds = mcnu_df.index.drop_duplicates()
                inds = partgrp_inds.intersection(mcnu_inds)
                mcnu_df = mcnu_df.loc[inds]

                # Get number of truth and signal events
                truth_events = len(mcnu_df[np.isin(mcnu_df[algo_event_col],categories)])
                signal_events = len(partgrp_cut_df[np.isin(partgrp_cut_df.truth.event_type,categories)])
                total_events = len(partgrp_cut_df)
                #print(f'cut: {cut}, truth_events: {truth_events}, signal_events: {signal_events}, total_events: {total_events}')
                pur[i+1] = signal_events/total_events #purity
                eff[i+1] = truth_events/init_truth
                f1[i+1] = 2*eff[i+1]*pur[i+1]/(eff[i+1]+pur[i+1])
        return pur,eff,f1
    def get_events_cuts(self,cuts=[],normalize=True):
        """
        Get number of events from list of cuts applied
        
        returns dictionary of events for each cut in order
        """
        if 'precut' not in cuts: cuts = ['precut']+cuts #add precut if not there 
        _partgrp = self.copy()
        et_key = self.get_key('truth.event_type')
        precut_events = _partgrp.data.groupby(et_key).genweight.sum().to_dict()
        events = dict({c:precut_events for c in cuts})
        for i,c in enumerate(cuts):
            if i == 0: continue #skip precut
            _partgrp.apply_cut(c)
            events[c] = _partgrp.data.groupby(et_key).genweight.sum().to_dict()
        df = pd.DataFrame(events).T
        df.index.name = 'cut'
        df = df.reindex(cuts)
        df = df.reindex(sorted(df.columns), axis=1)
        df.fillna(0,inplace=True)
        if normalize:
            df = df.div(df.sum(axis=1), axis=0)
        return df
    def get_numevents(self):
        """
        Get number of events from gen weights
        """
        if self.check_key('genweight'):
            return self.data.genweight.sum()
        raise ValueError('genweight not in dataframe. Run scale_to_pot first')
        
        