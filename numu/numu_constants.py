import numpy as np

#Selection
NOM_POT = 0.6e20

#Muon phase space bins - nominal
COSTHETA_BINS = np.array([-1,-0.5,0,0.27,0.45,0.62,0.76,0.86,0.94,1])
COSTHETA_BIN_LABELS = [f'{COSTHETA_BINS[i]:.2f} - {COSTHETA_BINS[i+1]:.2f}' for i in range(len(COSTHETA_BINS)-1)]
COSTHETA_CENTERS = (COSTHETA_BINS[:-1] + COSTHETA_BINS[1:])/2.

THETA_BINS = np.arccos(COSTHETA_BINS)*180/np.pi
THETA_BIN_LABELS = [f'{THETA_BINS[i]:.2f} - {THETA_BINS[i+1]:.2f}' for i in range(len(THETA_BINS)-1)]
THETA_CENTERS = (THETA_BINS[:-1] + THETA_BINS[1:])/2.

MOMENTUM_BINS = np.array([0,0.3,0.5,0.7,0.9,1.1,1.3,1.5,2,3,1e10])
MOMENTUM_BIN_LABELS = [f'{MOMENTUM_BINS[i]:.2f} - {MOMENTUM_BINS[i+1]:.2f}' if i < len(MOMENTUM_BINS)-2 else f'> {MOMENTUM_BINS[i]:.2f}' for i in range(len(MOMENTUM_BINS)-1)]
MOMENTUM_CENTERS = (MOMENTUM_BINS[:-1] + MOMENTUM_BINS[1:])/2.

#Used for 2D binning
DIFF_COSTHETA_BINS = np.array([-1,-0.5,0,0.27,0.45,0.62,0.76,0.86,0.94,1])
DIFF_COSTHETA_BIN_LABELS = [f'{DIFF_COSTHETA_BINS[i]:.2f} - {DIFF_COSTHETA_BINS[i+1]:.2f}' for i in range(len(DIFF_COSTHETA_BINS)-1)]
DIFF_COSTHETA_CENTERS = (DIFF_COSTHETA_BINS[:-1] + DIFF_COSTHETA_BINS[1:])/2.

DIFF_THETA_BINS = np.arccos(DIFF_COSTHETA_BINS)*180/np.pi
DIFF_THETA_BIN_LABELS = [f'{DIFF_THETA_BINS[i]:.2f} - {DIFF_THETA_BINS[i+1]:.2f}' for i in range(len(DIFF_THETA_BINS)-1)]
DIFF_THETA_CENTERS = (DIFF_THETA_BINS[:-1] + DIFF_THETA_BINS[1:])/2.

DIFF_MOMENTUM_BINS = np.array([0,0.3,0.5,0.7,1.1,1e10])
DIFF_MOMENTUM_BIN_LABELS = [f'{DIFF_MOMENTUM_BINS[i]:.2f} - {DIFF_MOMENTUM_BINS[i+1]:.2f}' if i < len(DIFF_MOMENTUM_BINS)-2 else f'> {DIFF_MOMENTUM_BINS[i]:.2f}' for i in range(len(DIFF_MOMENTUM_BINS)-1)]
DIFF_MOMENTUM_CENTERS = (DIFF_MOMENTUM_BINS[:-1] + DIFF_MOMENTUM_BINS[1:])/2.

DIFFERENTIAL_BINS = (len(DIFF_COSTHETA_BINS)-2) + (len(DIFF_MOMENTUM_BINS)-2)*(len(DIFF_COSTHETA_BINS)-1)
DIFFERENTIAL_EDGES = np.arange(-1.5,np.max(DIFFERENTIAL_BINS)+1.5,1)
DIFFERENTIAL_CENTERS = (DIFFERENTIAL_EDGES[:-1] + DIFFERENTIAL_EDGES[1:])/2.

diff_dict_template = {
  #Differential bin id is the key
  'costheta_bin': -1,
  'momentum_bin': -1,
  'costheta_edges': [-np.inf,np.inf],
  'momentum_edges': [-np.inf,np.inf],
  'momentum_center': -np.inf,
  'costheta_center': -np.inf
}

DIFFERENTIAL_DICTS = {c:diff_dict_template.copy() for c in sorted(DIFFERENTIAL_CENTERS)}
for c in DIFFERENTIAL_DICTS:
  c = int(c)
  if c >= 0:
    cbin = np.mod(c,len(DIFF_COSTHETA_BINS)-1)
    pbin = c//(len(DIFF_COSTHETA_BINS)-1)
  else:
    cbin = -1
    pbin = -1
  DIFFERENTIAL_DICTS[c]['costheta_bin'] = cbin
  DIFFERENTIAL_DICTS[c]['momentum_bin'] = pbin
  if cbin >= 0 and pbin >= 0:
    DIFFERENTIAL_DICTS[c]['costheta_edges'] = DIFF_COSTHETA_BINS[cbin:cbin+2]
    DIFFERENTIAL_DICTS[c]['momentum_edges'] = DIFF_MOMENTUM_BINS[pbin:pbin+2]
    DIFFERENTIAL_DICTS[c]['momentum_center'] = DIFF_MOMENTUM_CENTERS[pbin]
    DIFFERENTIAL_DICTS[c]['costheta_center'] = DIFF_COSTHETA_CENTERS[cbin]