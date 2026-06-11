import numpy as np

#Volume defs
AV = [
  [-200,200],#x
  [-200,200],#y
  [0,500],#z
  ]

AV_BUFFER =[ #Used to check if a track is contained
  [-195,195],#x
  [-195,195],#y
  [5,495],#z
  ]

FV = [
  [-190,190],#x
  [-190,190],#y
  [30,450],#z
  ]
CONT_VOL = [
  [-200,200],#x
  [-200,200],#y
  [30,500],#z - based on backup slide in https://sbn-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=35996
]

# Reject events in this region of the fiducial volume
NOT_FV_HIGH_Z = [
  [-190,0],#x
  [100,190],#y
  [250,450],#z
]

CATHODE_REGION = [
  [-5,5],#x
  [-200,200],#y
  [0,500],#z
]

TPC0 = [
  [-200,0],#x
  [-200,200],#y
  [0,500],#z
]

TPC1 = [
  [0,200],#x
  [-200,200],#y
  [0,500],#z
]

# 8 equal octants of AV (split at x=0, y=0, z=250)
# top/bottom = +/- y, left/right = -/+ x, low/high z = [0,250]/[250,500]
OCTANT0 = [  # top right, low z
  [0,200],#x
  [0,200],#y
  [0,250],#z
]
OCTANT1 = [  # bottom right, low z
  [0,200],#x
  [-200,0],#y
  [0,250],#z
]
OCTANT2 = [  # top left, low z
  [-200,0],#x
  [0,200],#y
  [0,250],#z
]
OCTANT3 = [  # bottom left, low z
  [-200,0],#x
  [-200,0],#y
  [0,250],#z
]
OCTANT4 = [  # top right, high z
  [0,200],#x
  [0,200],#y
  [250,500],#z
]
OCTANT5 = [  # bottom right, high z
  [0,200],#x
  [-200,0],#y
  [250,500],#z
]
OCTANT6 = [  # top left, high z
  [-200,0],#x
  [0,200],#y
  [250,500],#z
]
OCTANT7 = [  # bottom left, high z
  [-200,0],#x
  [-200,0],#y
  [250,500],#z
]

OCTANTS = [OCTANT0, OCTANT1, OCTANT2, OCTANT3, OCTANT4, OCTANT5, OCTANT6, OCTANT7]
OCTANT_LABELS = ['top_right_low_z', 'bottom_right_low_z', 'top_left_low_z', 'bottom_left_low_z', 'top_right_high_z', 'bottom_right_high_z', 'top_left_high_z', 'bottom_left_high_z']
OCTANT_NAMES = ['x > 0, y > 0, z < 250', 'x > 0, y < 0, z < 250', 'x < 0, y > 0, z < 250', 'x < 0, y < 0, z < 250', 'x > 0, y > 0, z > 250', 'x > 0, y < 0, z > 250', 'x < 0, y > 0, z > 250', 'x < 0, y < 0, z > 250']

TPC0_BUFFER = [
  [-195,-5],#x
  [-195,195],#y
  [5,495],#z
]
TPC1_BUFFER = [
  [5,195],#x
  [-195,195],#y
  [5,495],#z
]

F_SCALE = 0.0725 #https://sbn-docdb.fnal.gov/cgi-bin/sso/RetrieveFile?docid=41013&filename=2025-05-08_Absolute-Normalization-Data-MC.pdf&version=4

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
#FV, subtract off the veto region, which include the TPC0 and TPC1 buffer from the cathode, add back the high yz sliver to avoid double counting the buffer region
FV_VOLUME = np.prod([i[1]-i[0] for i in FV])\
    - np.prod([i[1]-i[0] for i in NOT_FV_HIGH_Z])\
    - np.prod([TPC0_BUFFER[0][1]-TPC0_BUFFER[0][0],FV[1][1]-FV[1][0],FV[2][1]-FV[2][0]])\
    + np.prod([TPC0_BUFFER[0][1]-TPC0_BUFFER[0][0],NOT_FV_HIGH_Z[1][1]-NOT_FV_HIGH_Z[1][0],NOT_FV_HIGH_Z[2][1]-NOT_FV_HIGH_Z[2][0]])#cm^3
FV_AREA = np.prod([i[1]-i[0] for i in FV[1:]]) #cm^2
NUMBER_TARGETS_FV = ARGON_DENSITY*FV_VOLUME*NUMBER_NUCLEONS #Number of nucleon targets in fiducial volume