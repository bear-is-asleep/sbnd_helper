import numpy as np
import sys
sys.path.append('/sbnd/app/users/brindenc/mysbnana_v09_75_03/srcs/sbnana/sbnana/SBNAna/pyana')
from sbnd.detector.volume import involume

def get_numucc_mask(mcnu):
  """
  Returns a mask for numu CC events
  """
  numu_mask = (np.abs(mcnu.pdg) == 14)
  cc_mask = (mcnu.iscc == 1)
  
  return numu_mask.values & cc_mask.values

def get_av_mask(mcnu):
  """
  Returns a mask for events in the active volume
  """
  return involume(mcnu.position)