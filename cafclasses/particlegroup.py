from .parent import CAF
from pyanalib import panda_helpers
import numpy as np
import pandas as pd

class ParticleGroup(CAF):
    """
    Interface class to interactions from ML Reco and slices from Pandora
    """
    #-------------------- constructor/rep --------------------#
    def __init__(self,data,**kwargs):
      super().__init__(data,**kwargs)
    #-------------------- cutters --------------------#
    def apply_cut(self, cut_name, condition=None, cut=True):
        """
        Apply a cut based on a specified condition.

        Parameters
        ----------
        cut_name : str
            The name of the cut to apply.
        condition : array_like
            Boolean array where True indicates the row passes the cut.
        cut : bool, optional
            Whether to actually apply the cut to the data. Default is True.
        """
        if 'cut.' not in cut_name: #Allow us to be lazy
            cut_name = 'cut.'+cut_name
        if cut and self.check_key(cut_name):
            orig_size = len(self.data)
            cut_col = self.get_key(cut_name)
            self.data = self.data[self.data[cut_col].values]
            new_size = len(self.data)
            print(f'Applied cut on key: {cut_name} ({orig_size:,} --> {new_size:,})')
            return
        elif condition is None:
            raise Exception(f'Attempting to cut on key not in data: {cut_name}')

        self.add_key([cut_name],fill=False)
        print(f'added key: {cut_name}')
        col = panda_helpers.getcolumns([cut_name], depth=self.key_length())[0]
        self.data.loc[:, col] = condition

        if cut:
            orig_size = len(self.data)
            self.data = self.data[self.data[col]]
            new_size = len(self.data)
            print(f'Applied cut on key: {cut_name} ({orig_size:,} --> {new_size:,})')
            return
    #-------------------- assigners --------------------#
    #TODO: Stop using these, they waste time
    def assign_costheta_bins(self,key='best_muon.costheta',assign_key='best_muon.costheta_bin',costheta_bins=None):
        """
        Assign costheta bins to dataframe
        
        costheta_bins: costheta bins set 
        """
        if costheta_bins is not None: self.set_costheta_bins(costheta_bins=costheta_bins)
        self.check_key(key)
        self.assign_bins(self.costheta_binning,key,df_comp=None,assign_key=assign_key,low_to_high=True)
    def assign_momentum_bins(self,key='best_muon.p',assign_key='best_muon.momentum_bin',momentum_bins=None):
        """
        Assign momentum bins to dataframe
        
        momentum_bins: momentum bins set 
        """
        if momentum_bins is not None: self.set_momentum_bins(momentum_bins=momentum_bins)
        self.check_key(key)
        self.assign_bins(self.momentum_binning,key,df_comp=None,assign_key=assign_key,low_to_high=True)
    def assign_prism_bins(self,key='best_muon.prism_theta',assign_key='best_muon.prism_bin',prism_bins=None):
        """
        Assign prism bins to dataframe
        
        prism_bins: prism bins set 
        """
        if prism_bins is not None: self.set_prism_bins(prism_bins=prism_bins)
        self.check_key(key)
        self.assign_bins(self.prism_binning,key,df_comp=None,assign_key=assign_key,low_to_high=True)
    #-------------------- adders --------------------#
    #-------------------- getters ---------------------#
    def get_pur_eff_f1(self,cuts=[],categories=[0,1]):
        """
        Get purity, efficiency, and f1 score from list of cuts applied

        categories: list of categories that are signal
        
        Purity: signal / total events (precision)
        Efficiency: remaining signal / initial signal (recall)
        f1: 2*eff*purity/(eff+purity)
        
        returns list of purs,effs,f1s for each cut in order
        """
        
        pur = np.zeros(len(cuts)+1)
        eff = np.zeros(len(cuts)+1)
        f1 = np.zeros(len(cuts)+1)
        
        partgrp_cut_df = self.copy().data #apply cuts to a copy
        init_signal = partgrp_cut_df[np.isin(partgrp_cut_df.truth.event_type,categories)].genweight.sum() #initial number of events
        init_total = partgrp_cut_df.genweight.sum()

        pur[0] = init_signal/init_total
        eff[0] = 1
        f1[0] = 1
        
        print(f'init_signal: {init_signal}, init_total: {init_total}')

        if len(cuts) != 0:
            assert isinstance(cuts,list), 'cuts must be a list'
            assert isinstance(cuts[0],str), 'cuts must be a list of strings'
            for i,cut in enumerate(cuts):
                partgrp_cut_df = partgrp_cut_df[partgrp_cut_df.cut[cut]] #apply cut
                signal_events = partgrp_cut_df[np.isin(partgrp_cut_df.truth.event_type,categories)].genweight.sum()
                total_events = partgrp_cut_df.genweight.sum()
                pur[i+1] = signal_events/partgrp_cut_df.genweight.sum() #purity
                eff[i+1] = signal_events/init_signal
                f1[i+1] = 2*eff[i+1]*pur[i+1]/(eff[i+1]+pur[i+1])
                print(f'cut: {cut}, signal_events: {signal_events}, total_events: {total_events}')
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
        
        