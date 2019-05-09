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

def read_data_hdf5(inpF):
        print('read data from hdf5:',inpF)
        h5f = h5py.File(inpF, 'r')
        objD={}
        for x in h5f.keys():
            obj=h5f[x][:]
            print('read ',x,obj.shape)
            objD[x]=obj

        h5f.close()
        return objD

paths = ['/global/homes/b/balewski/prj/roy-neuron-sim-data/hh_ball_stick_4par_easy_v3/raw/', '/global/homes/b/balewski/prj/roy-neuron-sim-data/hh_ball_stick_4par_hard_v3/raw/', '/global/homes/b/balewski/prj/roy-neuron-sim-data/hh_ball_stick_7par_v3/raw/', '/global/homes/b/balewski/prj/roy-neuron-sim-data/hh_two_dend_10par_v2/raw/', '/global/homes/b/balewski/prj/roy-neuron-sim-data/izhi_v6c/raw/', '/global/homes/b/balewski/prj/roy-neuron-sim-data/mainen_4par-v29/raw/', '/global/homes/b/balewski/prj/roy-neuron-sim-data/mainen_7par-v31/raw/', '/global/homes/b/balewski/prj/roy-neuron-sim-data/mainen_10par-v32/raw/']
for i in range(len(paths)):
    if 'izhi' in paths[i]:
        files = os.listdir(paths[i])
        print(files)
        # h5 = read_data_hdf5(paths[i] + files[0])
        # if 'izhi' in paths[i]:
        #     print('qa', h5['qa'])
        # else:
        #     print('binQA', h5['binQA'])
        print()
# path = '/global/cscratch1/sd/vbaratha/izhi/32traces/'
# files = os.listdir(path)
# for f in files:
#     h5 = read_data_hdf5(path + f)
# print(files)

comps = genfromtxt('/global/cscratch1/sd/kyoungh/NLeFEL/baseline/mainen_4p/comps/mcomp_spike_half_width.csv', delimiter=' ')
for i in range(len(comps)):
    if comps[i] < -1000:
        print(i, comps[i])
