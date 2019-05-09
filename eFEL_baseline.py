#!/usr/bin/env python

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

def extract_features(traces, features):
    #extract eFEL features for each trace
    #input: traces=list of traces, features=list of eFEL feature names
    #output: traces_results=list of dictionaries (feature->values) for each trace (a dictionary for each trace)
    time_arr = [0.02*i for i in range(len(traces[0]))]
    stim_start = 1
    stim_end = 179

    traces_dicts = []
    for t in traces:
        trace = {}
        trace['T'] = time_arr
        trace['V'] = t
        trace['stim_start'] = [stim_start]
        trace['stim_end'] = [stim_end]
        traces_dicts += [trace]
    print('traces_dicts created')

    start = time.time()
    traces_results = efel.getFeatureValues(traces_dicts, features)
    end = time.time()
    time_taken = (end - start) / 60
    print('Feature extraction runtime (min):', time_taken)

    return traces_results

def organize_features(traces_results, features):
    #combine the ouputted dictionaries from extract_features organized by feature as keys
    #input: traces_results=list of dictionaries (feature->values) for each trace, features=list of eFEL feature names
    #output: features_results=dictionary (feature->list of values for each trace)
    features_results = {}
    for feature in features:
        features_results[feature] = []
    for trace_results in traces_results:
        for feature, feature_values in trace_results.items():
            features_results[feature] += [[x for x in feature_values]]
    return features_results

def compare_features(features_results, features, comps_path, plots_path):
    #for each feature, compute the difference of feature values between two consecutive traces
    #input: features_results=dictionary (feature->list of values for each trace), features=list of eFEL feature names
    #output: N/A
    #saved files: comparisons (list of differences of consec. traces), plot of distribution of differences
    units_dict = {}
    units_dict['mean_frequency'] = 'Hz'
    units_dict['time_to_first_spike'] = 'ms'
    units_dict['mean_AP_amplitude'] = 'mV'
    units_dict['AHP_depth'] = 'mV'
    units_dict['spike_half_width'] = 'mV'
    fig, ax = plt.subplots(nrows=2, ncols=(len(features) // 2) + (len(features) % 2))
    for feature in features:
        comp = [np.mean(features_results[feature][i]) - np.mean(features_results[feature][i + 1]) for i in range(0, len(features_results[feature]), 2)]

        print(feature, len(comp))
        np.savetxt(comps_path + feature + '.csv', comp, delimiter = ' ')
        comp_mean = np.mean(comp)
        comp_std = np.std(comp)
        max_bin = max(plt.hist(comp, bins=100)[0])
        plt.plot([comp_mean - comp_std, comp_mean + comp_std], [max_bin/3, max_bin/3], linewidth=4, label='std = ' + str(comp_std))
        plt.plot([comp_mean, comp_mean], [0, max_bin], linewidth=4, label='mean = ' + str(comp_mean))
        plt.legend()
        plt.ylabel('# of traces')
        plt.xlabel('Difference ' + feature + ' (' + units_dict[feature] + ')')
        plt.title(feature)
        plt.plot()
        plt.savefig(plots_path + feature)

def hist_RMS(comp):
    return np.sqrt(np.sum([x**2 for x in comp])/len(comp))

def compare(traces, features, comps_path, plots_path):
    units_dict = {}
    units_dict['mean_frequency'] = 'Hz'
    units_dict['time_to_first_spike'] = 'ms'
    units_dict['mean_AP_amplitude'] = 'mV'
    units_dict['AHP_depth'] = 'mV'
    units_dict['spike_half_width'] = 'mV'

    time_arr = [0.02*i for i in range(len(traces[0]))]
    stim_start = 1
    stim_end = 179

    traces_dicts = []
    for t in traces:
        trace = {}
        trace['T'] = time_arr
        trace['V'] = t
        trace['stim_start'] = [stim_start]
        trace['stim_end'] = [stim_end]
        traces_dicts += [trace]
    print('traces_dicts created')
    fig, ax = plt.subplots(len(features), figsize=(20, 50))
    prelim_comp = [compute_eFEL_diff(traces_dicts[i], traces_dicts[i+1], features) for i in range(0, len(traces_dicts), 2)]
    i= 0
    for feature in features:

        comp = []
        for diff in [x[i] for x in prelim_comp]:
            if diff > -1000 and diff < 1000:
                comp += [diff]
        print(feature, len(comp))
        np.savetxt(comps_path + feature + '.csv', comp, delimiter = ' ')
        print('histogram RMS', hist_RMS(comp))
        comp_mean = np.mean(comp)
        comp_std = np.std(comp)
        max_bin = max(ax[i].hist(comp, bins=100)[0])
        ax[i].plot([comp_mean - comp_std, comp_mean + comp_std], [max_bin/3, max_bin/3], linewidth=4, label='std = ' + str(comp_std))
        ax[i].plot([comp_mean, comp_mean], [0, max_bin], linewidth=4, label='mean = ' + str(comp_mean))
        ax[i].legend()
        ax[i].set(xlabel='Difference ' + feature + ' (' + units_dict[feature] + ')', ylabel='# of traces', title=feature)
        i += 1
    plt.plot()
    plt.savefig(plots_path + 'histograms')

def compute_eFEL_diff(trace1, trace2, features):
    traces_results = efel.getFeatureValues([trace1, trace2], features)
    comp = []
    for feature in features:
        try:
            comp += [np.mean(traces_results[0][feature]) - np.mean(traces_results[1][feature])]
        except Exception as e:
            comp += [-9999]
    return comp

def main():
    print ('Number of arguments:', len(sys.argv)-1)
    print ('Argument List:', str(sys.argv[1:]))

    assert len(sys.argv)==2
    if sys.argv[1]=='hh_4p_easy':
        dir_path = '/global/homes/b/balewski/prj/roy-neuron-sim-data/hh_ball_stick_4par_easy_v3/raw/'
    elif sys.argv[1]=='hh_4p_hard':
        dir_path = '/global/homes/b/balewski/prj/roy-neuron-sim-data/hh_ball_stick_4par_hard_v3/raw/'
    elif sys.argv[1]=='hh_7p':
        dir_path = '/global/homes/b/balewski/prj/roy-neuron-sim-data/hh_ball_stick_7par_v3/raw/'
    elif sys.argv[1]=='hh_10p':
        dir_path = '/global/homes/b/balewski/prj/roy-neuron-sim-data/hh_two_dend_10par_v2/raw/'
    elif sys.argv[1]=='izhi':
        dir_path = '/global/homes/b/balewski/prj/roy-neuron-sim-data/izhi_v6c/raw/'
    elif sys.argv[1]=='mainen_4p':
        dir_path = '/global/homes/b/balewski/prj/roy-neuron-sim-data/mainen_4par-v29/raw/'
    elif sys.argv[1]=='mainen_7p':
        dir_path = '/global/homes/b/balewski/prj/roy-neuron-sim-data/mainen_7par-v31/raw/'
    elif sys.argv[1]=='mainen_10p':
        dir_path = '/global/homes/b/balewski/prj/roy-neuron-sim-data/mainen_10par-v32/raw/'
    features = ['mean_frequency', 'time_to_first_spike', 'mean_AP_amplitude', 'AHP_depth', 'spike_half_width']
    # features = ['time_to_first_spike', 'AHP_depth']
    files = os.listdir(dir_path)
    model_name = sys.argv[1]
    comps_path = './baseline/' + model_name + '/comps/mcomp_'
    plots_path = './baseline/' + model_name + '/plots/mcomp_hist_'

    num_traces = 50000
    traces = []
    index = 0
    while len(traces) < num_traces and index < len(files):
        if '.h5' in files[index]:
            h5 = read_data_hdf5(dir_path + files[index])
            prelim_traces = [v[5500:14500] for v in h5['voltages']]
            for i in range(len(prelim_traces)):
                if 'izhi' in dir_path:
                    if h5['qa'][i] == 1:
                        traces += [prelim_traces[i]]
                else:
                    if h5['binQA'][i] == 1:
                        traces += [prelim_traces[i]]
            print(len(traces))
        index += 1
    traces = traces[:num_traces]
    if len(traces) % 2 == 1:
        traces = traces[:-1]
    print('final number of traces:', len(traces))
    # traces_results = extract_features(traces, features)
    # features_results = organize_features(traces_results, features)
    # print([len(features_results[feature]) for feature in features])
    # compare_features(features_results, features, comps_path, plots_path)

    compare(traces, features, comps_path, plots_path)



if __name__ == '__main__':
    main()
