import numpy as np

#Neutrino bins - nominal
PRISM_BINS = np.arange(0,1.8,0.2)
#Muon phase space bins - nominal
COSTHETA_BINS = np.array([-1,-0.5,0,0.27,0.45,0.62,0.76,0.86,0.94,1])
THETA_BINS = np.arccos(COSTHETA_BINS)*180/np.pi
MOMENTUM_BINS = np.array([0,0.3,0.5,0.7,0.9,1.1,1.3,1.5,2,3])