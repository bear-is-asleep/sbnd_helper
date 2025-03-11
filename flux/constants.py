from ..detector.definitions import *

#Flux constants
HENRY_TO_BEAR_RATIO = 126e3/FV_AREA #cm^2
INTEGRATED_FLUX = 1.664e13*HENRY_TO_BEAR_RATIO/12 #cm^-2 https://sbn-docdb.fnal.gov/cgi-bin/sso/RetrieveFile?docid=35364&filename=NC%26%23960%3B0%20Analysis%20Update%20-%20Physics%20Meeting%2014th%20March.pdf&version=1
FRAC_NUMU = 0.936 #https://sbn-docdb.fnal.gov/cgi-bin/sso/RetrieveFile?docid=35609&filename=NuINT%202024%20practice%20talk.pdf
NUMU_INTEGRATED_FLUX = INTEGRATED_FLUX*FRAC_NUMU #cm^-2