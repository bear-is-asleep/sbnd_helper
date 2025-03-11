
import numpy as np
from ..constants import *

#Bins from MicroBooNE_CCInc_XSec_2DPcos_nu - https://github.com/NUISANCEMC/nuisance/blob/master/src/MicroBooNE/MicroBooNE_CCInc_XSec_2DPcos_nu.cxx
#Momentum bin edges - 
EdgesP = [
    [0.00, 0.18, 0.30, 0.45, 0.77, 2.50], # -1 to -0.5
    [0.00, 0.18, 0.30, 0.45, 0.77, 2.50], # -0.5 to 0
    [0.00, 0.18, 0.30, 0.45, 0.77, 2.50], # 0 to 0.27
    [0.00, 0.30, 0.45, 0.77, 2.50], # 0.27 to 0.45 
    [0.00, 0.30, 0.45, 0.77, 2.50], #  0.45 to 0.62 
    [0.00, 0.30, 0.45, 0.77, 2.50], # 0.62 to 0.76 
    [0.00, 0.30, 0.45, 0.77, 1.28, 2.50], # 0.76 to 0.86 
    [0.00, 0.30, 0.45, 0.77, 1.28, 2.50], # 0.86 to 0.94
    [0.00, 0.30, 0.45, 0.77, 1.28, 2.50], # 0.94 to 1
    ]

UBOONE_MOMENTUM_BINS = np.array([0.,0.2,0.3,0.5,0.8,1.3,2.5])