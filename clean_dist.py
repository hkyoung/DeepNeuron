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
import csv

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
    out = []
    for feature in features:
        print(i)
        comp = comps[i]
        comp_mean = np.mean(comp)
        comp_std = np.std(comp)
        np.savetxt(comps_path + 'cleaned_' + feature + '.csv', comp, delimiter = ' ')
        max_bin = max(ax[i].hist(comp, bins=100)[0])
        ax[i].plot([comp_mean - comp_std, comp_mean + comp_std], [max_bin/3, max_bin/3], linewidth=4, label='std = ' + str(comp_std))
        ax[i].plot([comp_mean, comp_mean], [0, max_bin], linewidth=4, label='mean = ' + str(comp_mean))
        ax[i].plot([], label='# of accepted values = ' + str(len(comp)))
        # print(feature)
        # print(len(comp), comp_mean, comp_std)
        out += [feature, len(comp), comp_mean, comp_std]
        ax[i].legend()
        ax[i].set(xlabel='Difference ' + feature + ' (' + units_dict[feature] + ')', ylabel='# of traces', title=feature)
        i += 1
    np.savetxt(comps_path + 'cleaned_out.csv', out, delimiter = ' ', fmt='%s')
    plt.plot()
    plt.savefig(plots_path + 'cleaned_histograms')

def main():
    features = ['mean_frequency', 'adaptation_index', 'time_to_first_spike', 'mean_AP_amplitude', 'AHP_depth', 'spike_half_width']
    paths = ['/global/cscratch1/sd/kyoungh/NLeFEL/baseline/mainen_4p/comps/', '/global/cscratch1/sd/kyoungh/NLeFEL/baseline/mainen_7p/comps/','/global/cscratch1/sd/kyoungh/NLeFEL/baseline/mainen_10p/comps/','/global/cscratch1/sd/kyoungh/NLeFEL/baseline/izhi/comps/', '/global/cscratch1/sd/kyoungh/NLeFEL/baseline/hh_4p_easy/comps/', '/global/cscratch1/sd/kyoungh/NLeFEL/baseline/hh_4p_hard/comps/', '/global/cscratch1/sd/kyoungh/NLeFEL/baseline/hh_7p/comps/', '/global/cscratch1/sd/kyoungh/NLeFEL/baseline/hh_10p/comps/']
    model_names = ['mainen_4p', 'mainen_7p', 'mainen_10p', 'izhi', 'hh_4p_easy', 'hh_4p_hard', 'hh_7p', 'hh_10p']
    out_path = ['/global/cscratch1/sd/kyoungh/NLeFEL/baseline/raw_vs_filtered.csv']
    for i in range(len(paths)):
        path = paths[i]
        model_name = model_names[i]
        out_data = [[model_name], ['feature', 'num comps (before)', 'mean (before)', 'std (before)', 'num comps (after)', 'mean (after)', 'std (after)']]
        # print(model_name)
        pre_files = os.listdir(path)
        files = []
        for feature in features:
            for f in pre_files:
                if feature in f:
                    files += [f]
        comps_path = './baseline/' + model_name + '/cleaned/mcomp_'
        plots_path = './baseline/' + model_name + '/cleaned/mcomp_'
        comps = []
        print('before')
        for f in files:

            feat = ''
            for feature in features:
                if feature in f:
                    feat = feature
            pre_comp = genfromtxt(path + f, delimiter=' ')
            std = np.std(pre_comp)
            mean = np.mean(pre_comp)

            comp = []
            outliers = []
            plt.figure(figsize=(20,10))
            for a in pre_comp:
                low_threshold = mean - (10*std)
                high_threshold = mean + (10*std)
                if a < high_threshold and a > low_threshold:
                    comp += [a]
                else:
                    outliers += [a]
            # if model_name == 'mainen_7p' and feat == 'spike_half_width':
            #     print(len(outliers))
            #     plt.hist(outliers, bins=100)
            #     plt.savefig('./baseline/mainen7p_spike_half_width_outliers')
            comps += [comp]

            print(f, len(pre_comp), mean, std, len(comp), np.mean(comp), np.std(comp))
            out_data += [[feat, str(len(pre_comp)), str(np.mean(pre_comp)), str(np.std(pre_comp)), str(len(comp)), str(np.mean(comp)), str(np.std(comp))]]
        with open('/global/cscratch1/sd/kyoungh/NLeFEL/baseline/baseline_analysis_' + model_name + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(out_data)

        # plot(comps, features, comps_path, plots_path)
        print()
    print(out_data)





if __name__ == '__main__':
    main()
