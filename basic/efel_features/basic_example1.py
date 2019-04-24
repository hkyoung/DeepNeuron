"""Basic example 1 for eFEL"""

import efel

import numpy
from numpy import genfromtxt


def main():
    """Main"""

    # Use numpy to read the trace data from the txt file

    # Time is the first column
    time = genfromtxt('./times.csv')
    # Voltage is the second column
    voltage = genfromtxt('./he_1_1_15.csv')

    # Now we will construct the datastructure that will be passed to eFEL

    # A 'trace' is a dictionary
    trace1 = {}

    # Set the 'T' (=time) key of the trace
    trace1['T'] = time

    # Set the 'V' (=voltage) key of the trace
    trace1['V'] = voltage

    # Set the 'stim_start' (time at which a stimulus starts, in ms)
    # key of the trace
    # Warning: this need to be a list (with one element)
    trace1['stim_start'] = [51]

    # Set the 'stim_end' (time at which a stimulus end) key of the trace
    # Warning: this need to be a list (with one element)
    trace1['stim_end'] = [7551]

    # Multiple traces can be passed to the eFEL at the same time, so the
    # argument should be a list
    traces = [trace1]

    # Now we pass 'traces' to the efel and ask it to calculate the feature
    # values
    traces_results = efel.getFeatureValues(traces,
                                           ['mean_frequency', 'adaptation_index', 'time_to_first_spike', 'mean_AP_amplitude', 'AHP_depth', 'spike_half_width'])

    # The return value is a list of trace_results, every trace_results
    # corresponds to one trace in the 'traces' list above (in same order)
    for trace_results in traces_results:
        # trace_result is a dictionary, with as keys the requested features
        for feature_name, feature_values in trace_results.items():
            print ("Feature %s has the following values: %s" % \
                (feature_name, ', '.join([str(x) for x in feature_values])))

    # for feature in efel.getFeatureNames():
    #     print(feature)


if __name__ == '__main__':
    main()
