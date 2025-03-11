import numpy as np
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import pandas as pd

#SBND imports
from sbnd.detector import volume
from sbnd.detector.definitions import *
from sbnd.general import plotters
from sbnd.constants import *

#PRISM info
PRISM_CENTROID =  [-74.,0] #x,y [cm]
DISTANCE_FROM_BNB = 110e2 #z [cm]
PRISM_BINS = np.arange(0,1.8,0.2)
#Precomputed prism areas
PRISM_AREAS = [4618.376215337481, #0-0.2
 13915.604904740983, #0.2 - 0.4
 23178.371027036217, #0.4 - 0.6
 29145.16260931502,  #0.6 - 0.8
 31647.70653212055,  #0.8 - 1.0
 26480.702880074743, #1.0 - 1.2
 17927.07922989864,  #1.2 - 1.4
 10753.82499354881,  #1.4 - 1.6
 2286.5909266944673  #1.6 - 1.8
 ]
PRISM_VOLUME = np.sum(PRISM_AREAS[:-1])*SBND_THICKNESS

def calc_rf(theta):
  return np.tan(theta*np.pi/180)*DISTANCE_FROM_BNB
def calc_theta(rf):
  return np.arctan(rf/DISTANCE_FROM_BNB)*180/np.pi
def make_prism_rings(theta,ax,**pltkwargs):
  center = PRISM_CENTROID
  radius = calc_rf(theta)
  [ax.add_patch(Circle(center,radius=r1,**pltkwargs)) for r1 in radius]
  return ax
def make_prism_plot(pos_df,weights=None,**pltkwargs):
  if weights is None:
    weights = np.ones(len(pos_df))
  fig,ax = plt.subplots(figsize=(10,8))
  im = ax.hist2d(pos_df.x,pos_df.y,weights=weights,**pltkwargs)#,norm=colors.LogNorm())
  cbar = fig.colorbar(im[3],ax=ax)
  cbar.ax.tick_params(labelsize=16)
  
  #prism lines
  ax.scatter(-74,0,s=200,c='red',marker='x')
  ax = make_prism_rings(PRISM_BINS,ax,fill=False,ls='--',lw=2,color='red',alpha=0.4)
  
  
  ax.set_xlabel('x [cm]')
  ax.set_ylabel('y [cm]')
  ax.set_title(rf'{round(np.sum(weights)):,} $\nu_\mu CC$ events')
  return fig,ax

def calc_prism_area(r1,r2,n=int(1e7),return_ring_area=False,show_plot=False):
  """
  Calc area of prism ring with inner radius r1 and outer radius r2.
  Account for detector edges by using monte carlo integration.
  """
  assert r1 < r2, 'r1 must be less than r2'
  #Get random points in detector
  x = np.random.uniform(-500,500,n)
  y = np.random.uniform(-500,500,n)
  z = np.full(n,250)
  
  df = pd.DataFrame({'x':x,'y':y,'z':z})
  outside_det = volume.involume(df)
  
  #Get points in detector
  x = x[outside_det]
  y = y[outside_det]
  n_in_det = len(x)
  r = np.sqrt((x-PRISM_CENTROID[0])**2 + (y-PRISM_CENTROID[1])**2)
  
  #Get points in ring
  in_ring = np.logical_and(r >= r1,r <= r2)
  n_in_ring = np.sum(in_ring)
  
  total_area = 400**2 #cm^2
  prism_area = n_in_ring/n_in_det*total_area
  
  if show_plot:
    x_in_ring = x[in_ring]
    y_in_ring = y[in_ring]
    
    x_out_ring = x[~in_ring]
    y_out_ring = y[~in_ring]
    
    #Get prism rings
    thetas = calc_theta(np.array([r1,r2]))
    
    fig,ax = plt.subplots(figsize=(8,6))
    ax.scatter(x_in_ring,y_in_ring,color='blue',label='In Ring',s=2)
    ax.scatter(x_out_ring,y_out_ring,color='red',label='Out of Ring',s=2)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_xlim(-200,200)
    ax.set_ylim(-200,200)
    ax.legend()
    
    #Add prism rings
    ax = make_prism_rings(thetas,ax,fill=False,ls='-',lw=2,color='black',alpha=0.8)
    
    plotters.set_style(ax,legend_loc='upper right')
    plotters.add_label(ax,f'Area = {prism_area:.2f} cm^2',where='bottomright',fontsize=20)
    
  
  if return_ring_area:
    return prism_area,np.abs(np.pi*(r2**2 - r1**2))
  return prism_area
