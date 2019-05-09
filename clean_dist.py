import bluepyopt
import efel
import h5py
import numpy as np
from numpy import genfromtxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os,sys
import time

# Terminal Commands
# salloc -N 1 -q debug -C haswell -t 00:30:00 -L SCRATCH
# salloc -N 1 -q interactive -C haswell -t 03:00:00 -L SCRATCH
# After successfully allocating nodes, enter these lines in terminal before running script
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

def plot(comps, features, comps_path, plots_path):
    units_dict = {}
    units_dict['mean_frequency'] = 'Hz'
    units_dict['time_to_first_spike'] = 'ms'
    units_dict['mean_AP_amplitude'] = 'mV'
    units_dict['AHP_depth'] = 'mV'
    units_dict['spike_half_width'] = 'mV'

    fig, ax = plt.subplots(len(features), figsize=(20, 50))
    i= 0
    for feature in features:
        comp = comps[i]
        comp_mean = np.mean(comp)
        comp_std = np.std(comp)
        np.savetxt(comps_path + 'cleaned_' + feature + '.csv', comp, delimiter = ' ')
        max_bin = max(ax[i].hist(comp, bins=100)[0])
        ax[i].plot([comp_mean - comp_std, comp_mean + comp_std], [max_bin/3, max_bin/3], linewidth=4, label='std = ' + str(comp_std))
        ax[i].plot([comp_mean, comp_mean], [0, max_bin], linewidth=4, label='mean = ' + str(comp_mean))
        ax[i].plot([], label='# of accepted values = ' + str(len(comp)))
        ax[i].legend()
        ax[i].set(xlabel='Difference ' + feature + ' (' + units_dict[feature] + ')', ylabel='# of traces', title=feature)
        i += 1
    plt.plot()
    plt.savefig(plots_path + 'cleaned_histograms')

def main():
    features = ['mean_frequency', 'time_to_first_spike', 'mean_AP_amplitude', 'AHP_depth', 'spike_half_width']
    paths = ['/global/cscratch1/sd/kyoungh/NLeFEL/baseline/mainen_4p/comps/', '/global/cscratch1/sd/kyoungh/NLeFEL/baseline/mainen_7p/comps/','/global/cscratch1/sd/kyoungh/NLeFEL/baseline/mainen_10p/comps/','/global/cscratch1/sd/kyoungh/NLeFEL/baseline/izhi/comps/', '/global/cscratch1/sd/kyoungh/NLeFEL/baseline/hh_4p_easy/comps/', '/global/cscratch1/sd/kyoungh/NLeFEL/baseline/hh_4p_hard/comps/', '/global/cscratch1/sd/kyoungh/NLeFEL/baseline/hh_7p/comps/', '/global/cscratch1/sd/kyoungh/NLeFEL/baseline/hh_10p/comps/']
    model_names = ['mainen_4p', 'mainen_7p', 'mainen_10p', 'izhi', 'hh_4p_easy', 'hh_4p_hard', 'hh_7p', 'hh_10p']
    for i in range(len(paths)):
        path = paths[i]
        model_name = model_names[i]
        files = os.listdir(path)
        comps_path = './baseline/' + model_name + '/cleaned/mcomp_'
        plots_path = './baseline/' + model_name + '/cleaned/mcomp_'
        comps = []
        for f in files:
            pre_comp = genfromtxt(path + f, delimiter=' ')
            std = np.std(pre_comp)
            mean = np.mean(pre_comp)
            print(mean, std)
            comp = []
            for a in pre_comp:
                low_threshold = mean - (10*std)
                high_threshold = mean + (10*std)
                if a < high_threshold and a > low_threshold:
                    comp += [a]
            comps += [comp]
        plot(comps, features, comps_path, plots_path)






if __name__ == '__main__':
    main()
