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

def extract_features(traces, features):
    time = [0.02*i for i in range(len(traces[0]))]
    stim_start = 1
    stim_end = 179

    traces_dicts = []
    for t in traces:
        trace = {}
        trace['T'] = time
        trace['V'] = t
        trace['stim_start'] = [stim_start]
        trace['stim_end'] = [stim_end]
        traces_dicts += [trace]

    traces_results = efel.getFeatureValues(traces_dicts, features)

    return traces_results

def organize_features(traces_results, features):
    features_results = {}
    for feature in features:
        features_results[feature] = []
    for trace_results in traces_results:
        for feature, feature_values in trace_results.items():
            features_results[feature] += [[x for x in feature_values]]
    return features_results

def diff_lists(lis1, lis2):
    if lis1 is None and lis2 is None:
       return 0
    if lis1 is None:
       lis1 = [0]
    if lis2 is None:
       lis2 = [0]
    len1, len2 = len(lis1), len(lis2)
    if len1 > len2:
       lis2 = np.concatenate((lis2, np.zeros(len1 - len2)), axis=0)
    if len2 > len1:
       lis1 = np.concatenate((lis1, np.zeros(len2 - len1)), axis=0)
    return np.sqrt(((np.array(lis1) - np.array(lis2))**2).mean())

def compare_features(features_results, features):
    for feature in features:
        truth = features_results[feature][0]
        comp = [diff_lists(truth, features_results[feature][i]) for i in range(1, len(features_results[feature]))]
        print(feature, len(comp))
        np.savetxt('./mainen/1pset/7p/comps/comp_' + feature + '.csv', comp, delimiter = ' ')
        comp_mean = np.mean(comp)
        comp_std = np.std(comp)
        plt.figure(figsize=(20,10))
        max_bin = max(plt.hist(comp, bins=50)[0])
        plt.plot([comp_mean - comp_std, comp_mean + comp_std], [max_bin/3, max_bin/3], linewidth=4, label='std = ' + str(comp_std))
        plt.plot([comp_mean, comp_mean], [0, max_bin], linewidth=4, label='mean = ' + str(comp_mean))
        plt.legend()
        plt.ylabel('# of traces')
        plt.xlabel('distance from truth')
        plt.title(feature)
        plt.plot()
        plt.savefig('./mainen/1pset/7p/plots/comp_hist_' + feature)

def main():
    file_path = '/global/cscratch1/sd/asranjan/efel_pred/efel_data_7v31.h5'
    h5 = read_data_hdf5(file_path)
    true_param = h5['true_param']
    true_trace = h5['true_trace']
    pred_param = h5['pred_params']
    pred_trace = h5['pred_volts']

    param_loss = []
    for p in pred_param:
        param_loss += [diff_lists(true_param, p)]
    print(len(pred_param[0]), len(true_param))
    print(len(param_loss))
    print(np.mean(param_loss))
    print(np.std(param_loss))
    # features = ['mean_frequency', 'time_to_first_spike', 'mean_AP_amplitude']
    # features = ['mean_frequency', 'time_to_first_spike', 'mean_AP_amplitude', 'AHP_depth', 'spike_half_width']
    # traces = [true_trace]
    # traces += [trace[5500:14500] for trace in pred_trace]
    # plt.figure(figsize=(20,10))
    # for trace in traces:
    #     plt.plot(trace)
    # plt.plot()
    # plt.savefig('./mainen/1pset/7p/plots/traces')
    # traces_results = extract_features(traces, features)
    # features_results = organize_features(traces_results, features)
    # print([len(features_results[feature]) for feature in features])
    # compare_features(features_results, features)



if __name__ == '__main__':
    main()
