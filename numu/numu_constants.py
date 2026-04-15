import numpy as np

#Muon phase space bins - nominal
COSTHETA_BINS = np.array([-1,-0.5,0,0.27,0.45,0.62,0.76,0.86,0.94,1])
#COSTHETA_BINS = np.arange(-1,1.1,0.1)
COSTHETA_BIN_LABELS = [f'{COSTHETA_BINS[i]:.2f} - {COSTHETA_BINS[i+1]:.2f}' for i in range(len(COSTHETA_BINS)-1)]
COSTHETA_CENTERS = (COSTHETA_BINS[:-1] + COSTHETA_BINS[1:])/2.

THETA_BINS = np.arccos(COSTHETA_BINS)*180/np.pi
THETA_BIN_LABELS = [f'{THETA_BINS[i]:.2f} - {THETA_BINS[i+1]:.2f}' for i in range(len(THETA_BINS)-1)]
THETA_CENTERS = (THETA_BINS[:-1] + THETA_BINS[1:])/2.

MOMENTUM_BINS = np.array([0,0.3,0.4,0.5,0.65,0.8,1.,1.4,1e10])
#MOMENTUM_BINS = np.concatenate([np.arange(0,3.1,0.1),[1e10]])
MOMENTUM_BIN_LABELS = [f'{MOMENTUM_BINS[i]:.2f} - {MOMENTUM_BINS[i+1]:.2f}' if i < len(MOMENTUM_BINS)-2 else f'> {MOMENTUM_BINS[i]:.2f}' for i in range(len(MOMENTUM_BINS)-1)]
MOMENTUM_CENTERS = (MOMENTUM_BINS[:-1] + MOMENTUM_BINS[1:])/2.

#Used for 2D binning
DIFF_COSTHETA_BINS = np.array([-1,-0.5,0,0.27,0.45,0.62,0.76,0.86,0.94,1])
DIFF_COSTHETA_BIN_LABELS = [f'{DIFF_COSTHETA_BINS[i]:.2f} - {DIFF_COSTHETA_BINS[i+1]:.2f}' for i in range(len(DIFF_COSTHETA_BINS)-1)]
DIFF_COSTHETA_CENTERS = (DIFF_COSTHETA_BINS[:-1] + DIFF_COSTHETA_BINS[1:])/2.
N_DIFF_COSTHETA_BINS = len(DIFF_COSTHETA_BINS) - 1

DIFF_THETA_BINS = np.arccos(DIFF_COSTHETA_BINS)*180/np.pi
DIFF_THETA_BIN_LABELS = [f'{DIFF_THETA_BINS[i]:.2f} - {DIFF_THETA_BINS[i+1]:.2f}' for i in range(len(DIFF_THETA_BINS)-1)]
DIFF_THETA_CENTERS = (DIFF_THETA_BINS[:-1] + DIFF_THETA_BINS[1:])/2.

#Maximum display momentum, rest is overflow. Use this for xsec
MAX_PMOM = 2.

# Default 2D momentum grid: one row per costheta bin
DIFF_MOMENTUM_BINS_2D = np.array([
  [0,0.2,0.3,0.4,1e10],
  [0,0.3,0.4,0.6,1e10],
  [0,0.3,0.4,0.6,1e10],
  [0,0.4,0.6,0.8,1e10],
  [0,0.4,0.6,0.8,1e10],
  [0,0.4,0.6,0.8,1e10],
  [0,0.6,1.,1.4,1e10],
  [0,0.6,1.,1.4,1e10],
  [0,0.6,1.,1.4,1e10],
  ])
N_DIFF_MOMENTUM_BINS_2D = DIFF_MOMENTUM_BINS_2D.shape[1]

def get_diff_momentum_bins_for_costheta_bin(costheta_bin_index):
    """
    Return the 1D momentum bin edges for a given costheta bin index.
    """
    if costheta_bin_index < 0 or costheta_bin_index >= N_DIFF_COSTHETA_BINS:
        raise IndexError(f'costheta_bin_index {costheta_bin_index} out of range 0..{N_DIFF_COSTHETA_BINS-1}')
    return DIFF_MOMENTUM_BINS_2D[costheta_bin_index]

def verify_differential_mapping(n_costheta_bins=None, n_momentum_bins=None):
    """
    Simple consistency check for the differential bin index mapping
    diff_index = costheta_bin + momentum_bin * n_costheta_bins.
    """
    if n_costheta_bins is None:
        n_costheta_bins = N_DIFF_COSTHETA_BINS
    if n_momentum_bins is None:
        n_momentum_bins = N_DIFF_MOMENTUM_BINS_2D
    indices = set()
    for p in range(n_momentum_bins):
        for c in range(n_costheta_bins):
            idx = c + p * n_costheta_bins
            if idx in indices:
                raise ValueError('Duplicate differential index encountered')
            indices.add(idx)
    expected = n_costheta_bins * n_momentum_bins
    if len(indices) != expected:
        raise ValueError(f'Got {len(indices)} unique indices, expected {expected}')
    return True

DIFFERENTIAL_BINS = (len(DIFF_COSTHETA_BINS)-2) + (len(DIFF_MOMENTUM_BINS_2D)-2)*(len(DIFF_COSTHETA_BINS)-1)
DIFFERENTIAL_EDGES = np.arange(-1.5,np.max(DIFFERENTIAL_BINS)+1.5,1)
DIFFERENTIAL_CENTERS = (DIFFERENTIAL_EDGES[:-1] + DIFFERENTIAL_EDGES[1:])/2.