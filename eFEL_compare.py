import bluepyopt
import efel
import h5py
import numpy as np
from numpy import genfromtxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# module load python/3.6-anaconda-4.4

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

def main():
    A_2 = read_data_hdf5('/project/projectdirs/mpccc/balewski/roy-neuron-sim-data/experiment_2019-04-10/packed/041019A_2.exp.h5')

    efel_time = [0.02*i for i in range(len(A_2['stim']))]

    traces = []
    stim_start = 5
    stim_end = 150

    for i in range(len(A_2['sweep2D'])):
        trace = {}
        trace['T'] = efel_time
        trace['V'] = A_2['sweep2D'][i]
        trace['stim_start'] = [stim_start]
        trace['stim_end'] = [stim_end]
        traces += [trace]

    features = ['mean_frequency', 'time_to_first_spike', 'mean_AP_amplitude']
    # features = ['mean_frequency', 'time_to_first_spike', 'mean_AP_amplitude', 'AHP_depth', 'spike_half_width']
    traces_results = efel.getFeatureValues(traces, features)
    i = 0
    feature_results = {}
    feature_means = {}
    feature_stds = {}
    for feature in features:
        feature_results[feature] = []
    for trace_results in traces_results:
        # trace_result is a dictionary, with as keys the requested features

        for feature_name, feature_values in trace_results.items():
            feature_results[feature_name] += [x for x in feature_values]

        # print()
        # print('Trace', i)
        # i += 1
        # for feature_name, feature_values in trace_results.items():
        #     print ("Feature %s has the following values: %s" % \
        #         (feature_name, ', '.join([str(x) for x in feature_values])))
        # print()

    for feature in features:
        mean = np.mean(feature_results[feature])
        std = np.std(feature_results[feature])
        feature_means[feature] = mean
        feature_stds[feature] = std
        # plt.figure(figsize=(20, 10))
        # max_val = max(plt.hist(feature_results[feature], bins=20)[0])
        # plt.plot([mean - std, mean + std], [max_val/3, max_val/3], linewidth=4, label='std = ' + str(std))
        # plt.plot([mean, mean], [0, max_val], linewidth=4, label='mean = ' + str(mean))
        # plt.legend()
        # plt.ylabel('# of traces')
        # plt.title(feature)
        # plt.savefig('./plots/hist_' + feature)


    score_distr = []

    for i in range(len(traces_results) - 1):
        for j in range(i + 1, len(traces_results)):
            tr1, tr2 = traces_results[i], traces_results[j]
            score = 0
            for feature in tr1.keys():
                tr1_feature_val = tr1[feature][0]
                tr2_feature_val = tr2[feature][0]
                score += ((tr1_feature_val - tr2_feature_val)**2)/feature_stds[feature]
            # print((score))
            score_distr.append(score)
    score_mean = np.mean(score_distr)
    score_std = np.std(score_distr)
    print(max(score_distr))
    plt.figure(figsize=(20,10))
    max_bin = max(plt.hist(score_distr, bins=20)[0])
    plt.plot([score_mean - std, score_mean + std], [max_bin/3, max_bin/3], linewidth=4, label='std = ' + str(score_std))
    plt.plot([score_mean, score_mean], [0, max_bin], linewidth=4, label='mean = ' + str(score_mean))
    plt.legend()
    plt.ylabel('# of scores')
    plt.xlabel('chi square score')
    plt.plot()
    plt.savefig('./plots/scores_hist')

    # for i in range(len(A_2['sweep2D'])):
    #     plt.figure(figsize=(20, 10))
    #     plt.plot(efel_time, A_2['stim'])
    #     plt.plot(efel_time, A_2['sweep2D'][1])
    #     plt.savefig('./plots/A_2_sweep' + str(i))



if __name__ == '__main__':
    main()
