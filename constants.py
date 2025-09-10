import numpy as np
from .detector.volume import FV

DUMMY_INT = np.iinfo(np.int32).max
DUMMY_FLOAT = np.nan

GeV2perm2 = 2.56819e31 #GeV^2 to m^2
GeV2percm2 = GeV2perm2*1e-4 #GeV^2 to cm^2


kElectronMass   =  5.109989461e-04        # GeV
kMuonMass       =  1.056583745e-01         # GeV
kTauMass        =  1.77686e+00             # GeV
kPionMass       =  1.3957018e-01          # GeV
kPi0Mass        =  1.349766e-01           # GeV
kProtonMass     =  9.38272081e-01           # GeV
kNeutronMass    =  9.39565413e-01           # GeV
kPhotonMass     =  0           # GeV

PDG_TO_MASS_MAP = {
  11 : kElectronMass,
  13 : kMuonMass,
  22 : kPhotonMass,
  111 : kPi0Mass,
  211 : kPionMass,
  2112 : kNeutronMass,
  2212 : kProtonMass,
}

#GENIE enums
GENIE_INTERACTION_MAP = {
    -np.iinfo(np.int32).max : 'kUnknownInteractionMode', #cosmics??
    -1 : 'kUnknownInteractionMode',
    0  : 'kQE',
    1  : 'kRes',
    2  : 'kDIS',
    3  : 'kCoh',
    4  : 'kCohElastic',
    5  : 'kElectronScattering',
    6  : 'kIMDAnnihilation',
    7  : 'kInverseBetaDecay',
    8  : 'kGlashowResonance',
    9  : 'kAMNuGamma',
    10 : 'kMEC',
    11 : 'kDiffractive',
    12 : 'kEM',
    13 : 'kWeakMix'
}

#Genie interaction type
kNuanceOffset = 1000
GENIE_TYPE_MAP =\
  {
    -1 : 'kUnknownInteractionType'                    , 
    kNuanceOffset +  1 : 'kCCQE'                      ,  # charged current quasi-elastic
    kNuanceOffset +  2 : 'kNCQE'                      ,  # neutral current quasi-elastic
    kNuanceOffset +  3 : 'kResCCNuProtonPiPlus'       ,  # resonant charged current, nu p -> l- p pi+
    kNuanceOffset +  4 : 'kResCCNuNeutronPi0'         ,  # resonant charged current, nu n -> l- p pi0
    kNuanceOffset +  5 : 'kResCCNuNeutronPiPlus'      ,  # resonant charged current, nu n -> l- n pi+
    kNuanceOffset +  6 : 'kResNCNuProtonPi0'          ,  # resonant neutral current, nu p -> nu p pi0
    kNuanceOffset +  7 : 'kResNCNuProtonPiPlus'       ,  # resonant neutral current, nu p -> nu p pi+
    kNuanceOffset +  8 : 'kResNCNuNeutronPi0'         ,  # resonant neutral current, nu n -> nu n pi0
    kNuanceOffset +  9 : 'kResNCNuNeutronPiMinus'     ,  # resonant neutral current, nu n -> nu p pi-
    kNuanceOffset + 10 : 'kResCCNuBarNeutronPiMinus'  ,  # resonant charged current, nubar n -> l+ n pi-
    kNuanceOffset + 11 : 'kResCCNuBarProtonPi0'       ,  # resonant charged current, nubar p -> l+ n pi0
    kNuanceOffset + 12 : 'kResCCNuBarProtonPiMinus'   ,  # resonant charged current, nubar p -> l+ p pi-
    kNuanceOffset + 13 : 'kResNCNuBarProtonPi0'       ,  # resonant charged current, nubar p -> nubar p pi0
    kNuanceOffset + 14 : 'kResNCNuBarProtonPiPlus'    ,  # resonant charged current, nubar p -> nubar n pi+
    kNuanceOffset + 15 : 'kResNCNuBarNeutronPi0'      ,  # resonant charged current, nubar n -> nubar n pi0
    kNuanceOffset + 16 : 'kResNCNuBarNeutronPiMinus'  ,  # resonant charged current, nubar n -> nubar p pi-
    kNuanceOffset + 17 : 'kResCCNuDeltaPlusPiPlus'    ,
    kNuanceOffset + 21 : 'kResCCNuDelta2PlusPiMinus'  ,
    kNuanceOffset + 28 : 'kResCCNuBarDelta0PiMinus'   ,
    kNuanceOffset + 32 : 'kResCCNuBarDeltaMinusPiPlus' ,
    kNuanceOffset + 39 : 'kResCCNuProtonRhoPlus'      ,
    kNuanceOffset + 41 : 'kResCCNuNeutronRhoPlus'     ,
    kNuanceOffset + 46 : 'kResCCNuBarNeutronRhoMinus' ,
    kNuanceOffset + 48 : 'kResCCNuBarNeutronRho0'     ,
    kNuanceOffset + 53 : 'kResCCNuSigmaPlusKaonPlus'  ,
    kNuanceOffset + 55 : 'kResCCNuSigmaPlusKaon0'     ,
    kNuanceOffset + 60 : 'kResCCNuBarSigmaMinusKaon0' ,
    kNuanceOffset + 62 : 'kResCCNuBarSigma0Kaon0'     ,
    kNuanceOffset + 67 : 'kResCCNuProtonEta'          ,
    kNuanceOffset + 70 : 'kResCCNuBarNeutronEta'      ,
    kNuanceOffset + 73 : 'kResCCNuKaonPlusLambda0'    ,
    kNuanceOffset + 76 : 'kResCCNuBarKaon0Lambda0'    ,
    kNuanceOffset + 79 : 'kResCCNuProtonPiPlusPiMinus' ,
    kNuanceOffset + 80 : 'kResCCNuProtonPi0Pi0'       ,
    kNuanceOffset + 85 : 'kResCCNuBarNeutronPiPlusPiMinus' ,
    kNuanceOffset + 86 : 'kResCCNuBarNeutronPi0Pi0'   ,
    kNuanceOffset + 90 : 'kResCCNuBarProtonPi0Pi0'    ,
    kNuanceOffset + 91 : 'kCCDIS'                     ,  # charged current deep inelastic scatter
    kNuanceOffset + 92 : 'kNCDIS'                     ,  # charged current deep inelastic scatter
    kNuanceOffset + 93 : 'kUnUsed1'                   ,
    kNuanceOffset + 94 : 'kUnUsed2'                   ,
    kNuanceOffset + 95 : 'kCCQEHyperon'               ,
    kNuanceOffset + 96 : 'kNCCOH'                     ,
    kNuanceOffset + 97 : 'kCCCOH'                     ,  # charged current coherent pion
    kNuanceOffset + 98 : 'kNuElectronElastic'         ,  # neutrino electron elastic scatter
    kNuanceOffset + 99 : 'kInverseMuDecay'            ,  # inverse muon decay
    kNuanceOffset + 100 : 'kMEC2p2h'                  ,   # extension of nuance encoding for MEC / 2p2h
  }
  
#GIBUU enums
GIBUU_INTERACTION_MAP = {
  0 : 'Other',
  1 : 'QE',
  32 : 'Pi + n',
  33 : 'Pi0 + p',
  34 : 'DIS',
  35 : '2p2h QE',
  36 : '2p2h Delta',
  37 : '2Pi'
}
GIBUU_INTERACTION_MAP.update({i:'Res (s=0)' for i in range(2,32)})

#Event types
# EVENT_TYPE_LIST= {
#   -1 : ["Unknown","unk","black"],
#   0 : [r"$\nu_\mu$ CC","numucc","goldenrod"],
#   1 : [r"$\nu$ NC","nc","darkgreen"],
#   2 : [r"$\nu_e$ CC","nuecc","navy"],
#   3 : ["Cosmic","cosmic","red"],
#   4 : [r"Non AV $\nu$","dirt","sienna"],
# }
#'$\\nu_\\mu$ CC (cont.)', '$\\nu_\\mu$ CC (exit)', '$\\nu_\\mu$ CC (OOPS)', 'Non-AV', '$\\nu_e$ CC', 'NC', 'Cosmic'
#['gold','goldenrod','purple','sienna','blue','green','red']
EVENT_TYPE_LIST= {
  -1 : ["Unknown","unk","black"],
  0 : [r"$\nu_\mu$ CC (cont.)","numucc_cont","gold"],
  1 : [r"$\nu_\mu$ CC (exit.)","numucc_exit","goldenrod"],
  2 : [r"$\nu_\mu$ CC (OOPS)","numucc_oops","purple"],
  3 : [r"Non-AV","dirt","sienna"],
  4 : [r"$\nu_e$ CC","nuecc","blue"],
  5 : [r"$\nu$ NC","nc","green"],
  6 : ["Cosmic","cosmic","red"],
}

EVENT_NAMES = [i[0] for i in EVENT_TYPE_LIST.values()]
EVENT_IDS = list(EVENT_TYPE_LIST.keys())
EVENT_COLORS = [i[2] for i in EVENT_TYPE_LIST.values()]
EVENT_LABELS = [i[1] for i in EVENT_TYPE_LIST.values()]

#Semantic types for pandora
SEMANITC_TYPE_MAP = {
  -1 : "Unknown", #probably cosmic
  0 : "Track",
  1 : "Shower"
}

#PDGs
PID_PDG_MAP = {
  0 : "unknown",
  11 : "electron",
  13 : "muon",
  22 : "photon",
  111 : "pi0",
  211 : "pi",
  2112 : "neutron",
  2212 : "proton",
}
  