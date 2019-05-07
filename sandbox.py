import bluepyopt
import efel
import h5py
import numpy as np
from numpy import genfromtxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# salloc -N 1 -q debug -C haswell -t 00:30:00 -L SCRATCH
# salloc -N 1 -q interactive -C haswell -t 03:00:00 -L SCRATCH
# module load python/3.6-anaconda-4.4
# export HDF5_USE_FILE_LOCKING=FALSE

path= '/global/cscratch1/sd/vbaratha/izhi/32traces/'
files = os.listdir(path)
print(files)

for i in range(len(files)):
    print(files[i])
    h5 = read_data_hdf5(path + files[i])
