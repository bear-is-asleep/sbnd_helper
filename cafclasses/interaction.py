from .particlegroup import ParticleGroup
import pandas as pd
from sbnd.detector.volume import *
from pyanalib import pandas_helpers

class CAFInteraction(ParticleGroup):
    """
    CAF Interaction class. Has functions for interaction level (pfps)
    """
    #-------------------- constructor/rep --------------------#
    def __init__(self,data,**kwargs):
        super().__init__(data,**kwargs)
    def __getitem__(self, item):
        data = super().__getitem__(item)
        return CAFInteraction(data)
    def copy(self,deep=True):
        return CAFInteraction(self.data.copy(deep))
    def load(fname,key='evt_0',**kwargs):
        df = pd.read_hdf(fname,key=key,**kwargs)
        return CAFInteraction(df,**kwargs)
    #-------------------- setters --------------------#
    def set_mcnu_containment(self,mcnu):
        """
        Set mcnu containment
        """
        return super().set_mcnu_containment(mcnu,'spine')
    #-------------------- cutters --------------------#
    def cut_muon(self,cut=True,min_ke=0.1):
        """
        Cut muon column
        """
        #Unfortunately, the ke column is in MeV, so we need to convert to MeV for the cut
        ke = self.data.mu.ke/1000.
        min_ke = min_ke
        self.apply_cut('cut.muon', (self.data.mu.pid == 2) & (ke > min_ke) & (self.data.mu.is_primary), cut=cut)
        if min_ke == 0.1:
            self.apply_cut('cut.truth.muon', self.data.truth.nmu_100MeV > 0, cut=False) # Never cut on truth
        elif min_ke == 0.027:
            self.apply_cut('cut.truth.muon', self.data.truth.nmu_27MeV > 0, cut=False) # Never cut on truth
        else:
            raise ValueError(f'Invalid min_ke: {min_ke}')
    def cut_cosmic(self,cut=True):
        """
        Cut cosmic column
        """
        self.apply_cut('cut.cosmic', self.data.is_flash_matched == 1, cut=cut)
        self.apply_cut('cut.truth.cosmic', (self.data.truth.pdg == -1) | (self.data.truth.pdg.isna()), cut=False) # Never cut on truth
    def cut_cosmic_score(self,cut=True,score=102.35):
        """
        Cut cosmic score column
        """
        self.apply_cut('cut.cosmic_score', self.data.flash_scores > score, cut=cut)
    def cut_fv(self,cut=True):
        """
        Cut fv column
        """
        self.apply_cut('cut.fv', self.data.fv == True, cut=cut)
        self.apply_cut('cut.truth.fv', self.data.truth.fv == True, cut=False) # Never cut on truth
    def cut_start_dedx(self,cut=True,dedx=4.17):
        """
        Cut start dedx column
        """
        #Only cut if not contained
        self.apply_cut('cut.start_dedx', ((self.data.mu.start_dedx < dedx) & ~(self.data.mu.is_contained.values.astype(bool))) | (self.data.mu.is_contained), cut=cut)
    def cut_is_cont(self,cut=True):
        """
        Cut is contained column
        """
        self.apply_cut('cut.cont', (self.data.mu.is_contained == 1) | (self.data.mu.is_contained == True), cut=cut)
        #TODO: Fix upstream replacement for is_contained_y->is_contained. Not sure why it's renamed
        self.apply_cut('cut.truth.cont', (self.data.truth.mu.is_contained_y == True) | (self.data.truth.mu.is_contained_y == 1), cut=False) # Never cut on truth

    #-------------------- adders --------------------#
    def add_cont(self):
        """
        Add in cont column
        """
        # Set keys
        keys = [
            'truth.cont'
        ]
        values = [
            involume(self.data.mu.start,volume=AV_BUFFER) & involume(self.data.mu.end,volume=AV_BUFFER)
        ]
        self.add_cols(keys,values,fill=False)
    def add_in_fv(self):
        """
        Add in fv column
        """
        #Set keys, values, conditions
        keys = [
            'fv',
            'truth.fv'
        ]
        values = [
            involume(self.data.vertex,volume=FV),
            involume(self.data.truth.position,volume=FV)
        ]
        self.add_cols(keys,values,fill=False)
    def add_in_av(self):
        """
        Add in av column
        """
        #Set keys, values, conditions
        keys = [
            'av',
            'truth.av'
        ]
        values = [
            involume(self.data.vertex,volume=AV),
            involume(self.data.truth.position,volume=AV)
        ]
        self.add_cols(keys,values,fill=False)
    def add_event_type(self,min_ke=0.1):
        """
        Add true event type

        Parameters
        ----------
        min_ke : float
            Minimum kinetic energy to be considered a muon [GeV]
        """
        iscc = self.data.truth.iscc == 1
        isnumu = abs(self.data.truth.pdg) == 14
        isnue = abs(self.data.truth.pdg) == 12
        iscosmic = (self.data.truth.pdg == -1) | (self.data.truth.pdg.isna())
        istrueav = self.data.truth.av
        istruefv = self.data.truth.fv
        #TODO: Fix upstream replacement for is_contained_y->is_contained. Not sure why it's renamed
        #iscont = (self.data.truth.mu.is_contained_y == 1) | (self.data.truth.mu.is_contained_y == True)
        iscont = (self.data.mu.tpart.is_contained == 1) | (self.data.mu.tpart.is_contained == True) #FIXME: Replace with genie defintion
        if min_ke == 0.1:
            ismuon = self.data.truth.nmu_100MeV > 0
        elif min_ke == 0.027:
            ismuon = self.data.truth.nmu_27MeV > 0
        else:
            raise ValueError(f'Invalid min_ke: {min_ke}')
        isnumuccav = iscc & isnumu & istrueav & ~iscosmic #numu cc av
        isnumuccfv = iscc & isnumu & istruefv & ~iscosmic #numu cc fv
        isnumuccfv_cont = isnumuccfv & iscont & ismuon
        isnumuccfv_uncont = isnumuccfv & ~iscont & ismuon
        isnumuccoops = isnumuccav & ~isnumuccfv_cont & ~isnumuccfv_uncont #numu cc out of phase space (oops)
        isnueccav = iscc & isnue & istrueav & ~iscosmic #nue cc av
        isncav = ~iscc & istrueav & ~iscosmic #nc av
        iscosmicav = istrueav & iscosmic #cosmic av
        isdirt = ~istrueav & ~iscosmic #dirt

        #Add to dataframe
        keys = [
            'truth.event_type'
        ]
        self.add_key(keys,fill=-1)
        cols = pandas_helpers.getcolumns(keys,depth=self.key_length())
        self.data.loc[isnumuccfv_cont,cols[0]] = 0 #numu cc (contained)
        self.data.loc[isnumuccfv_uncont,cols[0]] = 1 #numu cc (uncontained)
        self.data.loc[isnumuccoops,cols[0]] = 2 #numu cc oops
        self.data.loc[isdirt,cols[0]] = 3 #dirt
        self.data.loc[isnueccav,cols[0]] = 4 #nue cc
        self.data.loc[isncav,cols[0]] = 5 #nc
        self.data.loc[iscosmic,cols[0]] = 6 #cosmic
        #self.data.loc[isnumuccfv_cont | isnumuccfv_uncont,cols[0]] = 7 #numu cc (either)
        #the rest are unknown
    #-------------------- getters --------------------#
    def get_pur_eff_f1(self,mcnu,cuts,categories=[0,1]):
        """
        Get pur eff f1
        """
        return super().get_pur_eff_f1('spine',mcnu,cuts,categories)
    
    