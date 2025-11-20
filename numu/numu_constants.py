import numpy as np

#Selection
NOM_POT = 0.6e20

#Muon phase space bins - nominal
COSTHETA_BINS = np.array([-1,-0.5,0,0.27,0.45,0.62,0.76,0.86,0.94,1])
COSTHETA_BIN_LABELS = [f'{COSTHETA_BINS[i]:.2f} - {COSTHETA_BINS[i+1]:.2f}' for i in range(len(COSTHETA_BINS)-1)]

THETA_BINS = np.arccos(COSTHETA_BINS)*180/np.pi
THETA_BIN_LABELS = [f'{THETA_BINS[i]:.2f} - {THETA_BINS[i+1]:.2f}' for i in range(len(THETA_BINS)-1)]

MOMENTUM_BINS = np.array([0,0.3,0.5,0.7,0.9,1.1,1.3,1.5,2,3,1e10])
MOMENTUM_BIN_LABELS = [f'{MOMENTUM_BINS[i]:.2f} - {MOMENTUM_BINS[i+1]:.2f}' if i < len(MOMENTUM_BINS)-2 else f'> {MOMENTUM_BINS[i]:.2f}' for i in range(len(MOMENTUM_BINS)-1)]
