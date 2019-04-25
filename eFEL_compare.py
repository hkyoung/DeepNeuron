import bluepyopt
import efel
import h5py
import numpy as np
from numpy import genfromtxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    traces_results = efel.getFeatureValues(traces, ['mean_frequency', 'time_to_first_spike', 'mean_AP_amplitude', 'AHP_depth', 'spike_half_width'])
    i = 0
    for trace_results in traces_results:
        # trace_result is a dictionary, with as keys the requested features
        print()
        print('Trace', i)
        i += 1
        for feature_name, feature_values in trace_results.items():
            print ("Feature %s has the following values: %s" % \
                (feature_name, ', '.join([str(x) for x in feature_values])))
        print()


    # for i in range(len(A_2['sweep2D'])):
    #     plt.figure(figsize=(20, 10))
    #     plt.plot(efel_time, A_2['stim'])
    #     plt.plot(efel_time, A_2['sweep2D'][1])
    #     plt.savefig('./plots/A_2_sweep' + str(i))



if __name__ == '__main__':
    main()
