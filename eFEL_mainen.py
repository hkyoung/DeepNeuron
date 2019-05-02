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

def extract_truth(h5_file, features):
    truth = h5_file['trace2D']
    time = [0.02*i for i in range(len(truth[0]))]
    stim_start = 1
    stim_end = 179
    traces = []


    for i in range(len(truth)):
        trace = {}
        trace['T'] = time
        trace['V'] = truth[i]
        trace['stim_start'] = [stim_start]
        trace['stim_end'] = [stim_end]
        traces += [trace]


    # features = ['mean_frequency', 'time_to_first_spike', 'mean_AP_amplitude', 'AHP_depth', 'spike_half_width']
    print('extracting features from traces')
    traces_results = efel.getFeatureValues(traces, features)
    print('done extracting')
    i = 0
    feature_results = {}
    feature_means = {}
    feature_stds = {}
    # for feature in features:
    #     feature_results[feature] = []
    for trace_results in traces_results:
        # trace_result is a dictionary, with as keys the requested features

        # for feature_name, feature_values in trace_results.items():
        #     feature_results[feature_name] += [x for x in feature_values]

        print()
        print('Trace', i)
        i += 1
        for feature_name, feature_values in trace_results.items():
            print ("Feature %s has the following values: %s" % \
                (feature_name, ', '.join([str(x) for x in feature_values])))
        print()

    return traces_results

def extract_model(mat, features):
    time = [0.02*i for i in range(len(mat[0]))]
    stim_start = 1
    stim_end = 179
    traces = []

    for i in range(len(mat)):
        trace = {}
        trace['T'] = time
        trace['V'] = mat[i]
        trace['stim_start'] = [stim_start]
        trace['stim_end'] = [stim_end]
        traces += [trace]

    print('extracting features from traces')
    traces_results = efel.getFeatureValues(traces, features)
    print('done extracting')
    i = 0
    feature_results = {}
    feature_means = {}
    feature_stds = {}
    # for feature in features:
    #     feature_results[feature] = []
    for trace_results in traces_results:
        # trace_result is a dictionary, with as keys the requested features

        # for feature_name, feature_values in trace_results.items():
        #     feature_results[feature_name] += [x for x in feature_values]

        print()
        print('Trace', i)
        i += 1
        for feature_name, feature_values in trace_results.items():
            print ("Feature %s has the following values: %s" % \
                (feature_name, ', '.join([str(x) for x in feature_values])))
        print()

    return traces_results

def plot_truth(h5_file):
    truth = h5_file['trace2D']
    stim = h5_file['stim0']
    time = [0.02*i for i in range(len(stim))]
    plt.figure(figsize=(20, 10))
    plt.plot(time, stim)
    plt.plot(time, truth[0])
    plt.savefig('./mainen/4p/plots/truth_volts/0')

def make_comp(features, truth_features, model_features):
    for feature_name in features:
        comp = [truth_features[feature_name][i] - model_features[feature_name][i] for i in range(len(model_features[feature_name]))]
        np.savetxt('./mainen/4p/comps/comp_' + feature_name + '.csv', comp, delimiter = ' ')
        comp_mean = np.mean(comp)
        comp_std = np.std(comp)
        plt.figure(figsize=(20,10))
        max_bin = max(plt.hist(comp, bins=60)[0])
        plt.plot([comp_mean - comp_std, comp_mean + comp_std], [max_bin/3, max_bin/3], linewidth=4, label='std = ' + str(comp_std))
        plt.plot([comp_mean, comp_mean], [0, max_bin], linewidth=4, label='mean = ' + str(comp_mean))
        plt.legend()
        plt.ylabel('# of traces')
        plt.xlabel('distance from truth')
        plt.title(feature_name)
        plt.plot()
        plt.savefig('./mainen/4p/plots/comp_hist_' + feature_name)
        print('done with comp', feature_name)


def main():
    truth_path = '/project/projectdirs/mpccc/balewski/roy-neuron-sim-data/paper1/april25_pred/mainen_4pv29-ML693-mainen_4PV29/0/cellRegr.sim.pred.h5'
    features = ['mean_frequency', 'time_to_first_spike', 'mean_AP_amplitude']

    truth_h5 = read_data_hdf5(truth_path)
    # plot_truth(truth_h5)
    truth_results = extract_truth(truth_h5, features)
    truth_features = {}
    for feature in features:
        truth_features[feature] = []
    for truth_result in truth_results:
        for feature_name, feature_values in truth_result.items():
            truth_features[feature_name] += [x for x in feature_values]
    print('done moving to truth_features')


    model_path = '/global/cscratch1/sd/asranjan/mainen4v29_pred/data/'
    trace_files = ['mainen_4paramsv29_pred_40000_1chirp23a.h5', 'mainen_4paramsv29_pred_40000_2chirp23a.h5', 'mainen_4paramsv29_pred_40000_3chirp23a.h5', 'mainen_4paramsv29_pred_40000_4chirp23a.h5']
    models_traces = []
    for trace_file in trace_files:
        model_h5 = read_data_hdf5(model_path + trace_file)
        mat = model_h5['voltages']
        for i in range(8):
            model_traces_raw = mat[5000*i: 5000*(i+1)]
            model_traces_trim = [model_traces_raw[i][5500:14500] for i in range(len(model_traces_raw))]
            models_traces += [model_traces_trim]
        print('done loading matrix', trace_file)
    print('number of models', len(models_traces))
    model_features = {}
    for i in range(len(models_traces)):
        model_results = extract_model(models_traces[i], features)

        for feature in features:
            model_features[feature] = []
        for model_result in model_results:
            for feature_name, feature_values in model_result.items():
                model_features[feature_name] += [x for x in feature_values]
        print('done moving over to feature, model', i)
    make_comp(features, truth_features, model_features)












if __name__ == '__main__':
    main()
