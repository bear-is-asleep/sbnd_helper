NOM_POT = 10e20

#SBND volume
FACE_AREA = 400**2 #cm^2
SBND_THICKNESS = 500 #cm
SBND_VOLUME = FACE_AREA*SBND_THICKNESS #cm^3
ARGON_MASS_DENSITY = 1.3954 #g/cm^3
AVOGADRO_CONSTANT = 6.02214076e23 #mol^-1
ARGON_MOLAR_MASS = 39.95 #g/mol
ARGON_DENSITY = ARGON_MASS_DENSITY/ARGON_MOLAR_MASS*AVOGADRO_CONSTANT #n_argon/cm^3
NUMBER_TARGETS = ARGON_DENSITY*SBND_VOLUME #Number of argon targets in active volume


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

#GENIE enums
GENIE_INTERACTION_MAP = {
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