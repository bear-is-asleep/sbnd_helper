import numpy as np

#Volume defs
AV = [
  [-200,200],#x
  [-200,200],#y
  [0,500],#z
  ]

FV = [
  [-199.15+10,199.15-10],#x
  [-200+10,200-10],#y
  [0+30,500-50],#z
  ]
CONT_VOL = [
  [-200,200],#x
  [-200,200],#y
  [30,500],#z - based on backup slide in https://sbn-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=35996
]

#SBND volume
FACE_AREA = 400**2 #cm^2
SBND_THICKNESS = 500 #cm
SBND_VOLUME = FACE_AREA*SBND_THICKNESS #cm^3
ARGON_MASS_DENSITY = 1.3954 #g/cm^3
AVOGADRO_CONSTANT = 6.02214076e23 #mol^-1
ARGON_MOLAR_MASS = 39.95 #g/mol
NUMBER_NUCLEONS = 40 #nucleons in argon
ARGON_DENSITY = ARGON_MASS_DENSITY/ARGON_MOLAR_MASS*AVOGADRO_CONSTANT #n_argon/cm^3
NUMBER_TARGETS = ARGON_DENSITY*SBND_VOLUME*NUMBER_NUCLEONS #Number of argon targets in active volume
FV_VOLUME = np.prod([i[1]-i[0] for i in FV]) #cm^3
FV_AREA = np.prod([i[1]-i[0] for i in FV[1:]]) #cm^2
NUMBER_TARGETS_FV = ARGON_DENSITY*FV_VOLUME*NUMBER_NUCLEONS #Number of argon targets in fiducial volume