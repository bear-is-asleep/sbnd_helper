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
    def add_track_flipping(self,algo,suffix=""):
        """
        Add track flipping boolean. True if start and end are correctly matched

        Parameters
        ----------
        algo: str
          Algorithm to use for event type. Either 'pandora' or 'spine'
        """
        def _get_vector(prefix):
            axes = ['x','y','z']
            cols = self.get_key([f'{prefix}.{axis}' for axis in axes])
            return np.column_stack([self.data.loc[:,col].values for col in cols])

        if algo == 'pandora':
            start = _get_vector(f'mu{suffix}.pfp.trk.start')
            true_start = _get_vector(f'mu{suffix}.pfp.trk.truth.p.start')
            true_end = _get_vector(f'mu{suffix}.pfp.trk.truth.p.end')
            keys = [f'mu.pfp.trk.is_flipped']
        elif algo == 'spine':
            start = _get_vector(f'mu{suffix}.start_point')
            true_start = _get_vector(f'mu{suffix}.tpart.start_point')
            true_end = _get_vector(f'mu{suffix}.tpart.end_point')
            keys = [f'mu.is_flipped']
        else:
            raise ValueError(f'Invalid algo: {algo}')
        #Find distance between start and true start
        diff_s2s = np.linalg.norm(start-true_start,axis=1)
        diff_s2e = np.linalg.norm(start-true_end,axis=1)

        #Set keys, values, and conditions
        self.add_cols(keys,[np.array(diff_s2s > diff_s2e).astype(np.float64)])
    def add_stat_unc(self,nuniv=100, progress_bar=False):
        """
        Add statistical uncertainty to the dataframe
        
        Parameters
        ----------
        nuniv : int
            Number of statistical uncertainty universes to generate (default: 100)
        """
        from tqdm import tqdm
        
        # Generate universe keys
        univ_keys = [f"truth.stat.univ_{i}" for i in range(nuniv)]
        
        # Create unique seeds based on event metadata
        n_events = len(self.data)
        meta_seeds = []
        seen_seeds = set()
        
        for i in tqdm(range(n_events), desc="Generating event seeds", disable=not progress_bar):
            # Get event metadata from the dataframe index
            index_tuple = self.data.index[i]
            base_seed = hash(str(index_tuple)) % (2**32)
            
            # Ensure uniqueness by adding event index if needed
            unique_seed = base_seed
            counter = 0
            while unique_seed in seen_seeds:
                unique_seed = (base_seed + counter) % (2**32)
                counter += 1
                if counter > 1000:  # Safety check to avoid infinite loop
                    raise ValueError(f"Could not generate unique seed for event {i}")
            
            seen_seeds.add(unique_seed)
            meta_seeds.append(unique_seed)
        
        # Verify all seeds are unique
        assert len(meta_seeds) == len(set(meta_seeds)), f"Non-unique seeds detected! Found {len(meta_seeds)} seeds but only {len(set(meta_seeds))} unique ones"
        #print(f"Successfully generated {len(meta_seeds)} unique seeds")
        
        # Generate Poisson weights for each universe
        poisson_mean = 1.0
        all_poisson_weights = []
        
        for uidx in tqdm(range(nuniv), desc="Generating universes", disable=not progress_bar):
            universe_seed = hash(f"universe_{uidx}") % (2**32)
            
            poisson_weights = []
            for sidx, meta_seed in enumerate(meta_seeds):
                # Combine universe seed with event seed for unique randomness
                combined_seed = (universe_seed + meta_seed) % (2**32)
                np.random.seed(combined_seed)
                
                poisson_val = np.random.poisson(poisson_mean)
                poisson_weights.append(float(poisson_val))  # Convert to float
            
            all_poisson_weights.append(np.array(poisson_weights, dtype=float))
        self.add_cols(univ_keys, all_poisson_weights)

    #-------------------- setters --------------------#
    def set_mcnu_containment(self,mcnu,algo,suffix=""):
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
        # Optimize memory: extract only needed columns before copying
        truth_ind_key = self.get_key('truth.ind')[0]
        mask = ~self.data[truth_ind_key].isna()
        
        # Extract only the columns we need using attribute-style access (works with MultiIndex)
        # Get the masked subset first
        masked_data = self.data.loc[mask]
        truth_ind_series = masked_data[truth_ind_key]
        if algo == 'pandora':
            contained_col = self.get_key(f'mu{suffix}.pfp.trk.truth.p.contained')[0]
            contained_series = masked_data.loc[:, contained_col]
        elif algo == 'spine':
            contained_col = self.get_key(f'mu{suffix}.tpart.is_contained')[0]
            contained_series = masked_data.loc[:, contained_col]
        index_vals = masked_data.index
        
        # Create minimal dataframe with just what we need
        pgrp_df = pd.DataFrame({
            truth_ind_key: truth_ind_series.values,
            'contained': contained_series.values
        }, index=index_vals)
        
        # Clean up intermediate objects immediately
        del masked_data, truth_ind_series, contained_series, index_vals
        
        pgrp_df.index = pgrp_df.index.droplevel(-1) #Drop slice index
        pgrp_df = pgrp_df.set_index(truth_ind_key,append=True) #Use nu index
        pgrp_df.index.names = [None] * len(pgrp_df.index.names) #Drop names of indices

        #Drop any indices that are not in the mcnu index
        pgrp_df = pgrp_df[pgrp_df.index.isin(mcnu.data.index)]
        #Drop duplicated pgrp_df indices
        pgrp_df = pgrp_df.loc[~pgrp_df.index.duplicated(keep='first')]

        #Add to mcnu
        keys = [f'mu{suffix}.is_{algo}_contained']
        mcnu.add_key(keys,fill=False) #Fill to false by default
        cols = pandas_helpers.getcolumns(keys,depth=mcnu.key_length())
        is_contained = (pgrp_df['contained'] == 1) | (pgrp_df['contained'] == True)
        mcnu.data.loc[pgrp_df.index,cols] = is_contained
        
        # Explicitly delete the copy to free memory immediately
        del pgrp_df, is_contained
        return mcnu
    #-------------------- getters ---------------------#
    def get_reference_mcnu(self,mcnu):
        """
        Get reference mcnu object
        """

        #Set indices to be the truth indices
        partgrp_cut_df = self.copy().data #apply cuts to a copy
        partgrp_cut_df = partgrp_cut_df.droplevel(-1).set_index(partgrp_cut_df.truth.ind,append=True)

        #Get truth object for efficiency calculations
        _mcnu = mcnu.copy()
        _mcnu.data.dropna(subset=_mcnu.get_key('ind'),inplace=True)
        _mcnu.data.droplevel(-1).set_index(_mcnu.data.ind,append=True)
        _mcnu.data.drop_duplicates(inplace=True)
        
        # Get truth indices from filtered data
        truth_index_tuples = partgrp_cut_df.index.unique()
        
        # Filter mcnu.data using index intersection - more robust than isin()
        _mcnu.data = _mcnu.data.loc[_mcnu.data.index.intersection(truth_index_tuples)]
        
        return _mcnu
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
        assert algo in ['pandora','spine'], f'Invalid algo: {algo}'
        algo_event_col = mcnu.get_key(f'event_type_{algo}')
        event_col = self.get_key('truth.event_type')

        
        pur = np.zeros(len(cuts)+1)
        eff = np.zeros(len(cuts)+1)
        f1 = np.zeros(len(cuts)+1)
        true_events= pur.copy()
        reco_events= pur.copy()
        event_types = np.sort(np.unique(self.data[event_col].values))
        events_by_cat = np.zeros((len(cuts)+1, len(event_types)))

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

        truth_inds = mcnu_df[np.isin(mcnu_df[algo_event_col],categories)].index.drop_duplicates()
        init_truth = mcnu_df.loc[truth_inds].genweight.sum()
        init_truth_total = mcnu_df.genweight.sum()

        reco_inds = partgrp_cut_df[np.isin(partgrp_cut_df.truth.event_type,categories)].index.drop_duplicates()
        init_signal = partgrp_cut_df.loc[reco_inds].genweight.sum()
        init_total = partgrp_cut_df.genweight.sum()

        # Get no cut efficiency and purity
        partgrp_inds = partgrp_cut_df.index.drop_duplicates()
        mcnu_inds = mcnu_df.index.drop_duplicates()
        inds = partgrp_inds.intersection(mcnu_inds)
        mcnu_df = mcnu_df.loc[inds]
        # Get number of truth and signal events
        truth_inds = mcnu_df[np.isin(mcnu_df[algo_event_col],categories)].index.drop_duplicates()
        truth_events = mcnu_df.loc[truth_inds].genweight.sum()

        pur[0] = init_signal/init_total
        eff[0] = truth_events/init_truth
        f1[0] = 2*eff[0]*pur[0]/(eff[0]+pur[0])
        true_events[0] = init_truth
        reco_events[0] = init_total
        
        events_by_cat[0] = partgrp_cut_df.groupby(event_col).genweight.sum().reindex(event_types, fill_value=0).values.tolist()
        #print(f'events_by_cat[0]: {events_by_cat[0]}')
        #print(f'init_truth: {init_truth}, init_truth_total: {init_truth_total}, init_signal: {init_signal}, init_total: {init_total}')
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
            truth_inds = mcnu_df[np.isin(mcnu_df[algo_event_col],categories)].index.drop_duplicates()
            truth_events = mcnu_df.loc[truth_inds].genweight.sum()
            total_events = mcnu_df.genweight.sum()
            truth_events = mcnu_df.loc[truth_inds].genweight.sum()

            reco_inds = partgrp_cut_df[np.isin(partgrp_cut_df.truth.event_type,categories)].index.drop_duplicates()
            signal_events = partgrp_cut_df.loc[reco_inds].genweight.sum()
            total_events = partgrp_cut_df.genweight.sum()
            #print(f'-cut: {cut}, truth_events: {truth_events}, signal_events: {signal_events}, total_events: {total_events}')
            with np.errstate(divide='ignore', invalid='ignore'):
                eff[i+1] = truth_events/init_truth
                pur[i+1] = signal_events/total_events
                f1[i+1] = 2*eff[i+1]*pur[i+1]/(eff[i+1]+pur[i+1])
            true_events[i+1] = truth_events
            reco_events[i+1] = total_events
            events_by_cat[i+1] = partgrp_cut_df.groupby(event_col).genweight.sum().reindex(event_types, fill_value=0).values.tolist()
            #print(f'events_by_cat[{i+1}]: {events_by_cat[i+1]}')
        return pur,eff,f1,true_events,reco_events,events_by_cat
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
        
        