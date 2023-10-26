import numpy as np

#PRISM info
prism_centroid =  [-74.,0] #x,y [cm]
distance_from_bnb = 110e2 #z [cm]

def calc_rf(theta):
  return np.tan(theta*np.pi/180)*distance_from_bnb
