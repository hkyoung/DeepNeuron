import numpy as np
from numpy import genfromtxt

time = genfromtxt('./times10k0_02.csv')

cumul_time = [sum(time[:i]) for i in range(len(time))]
np.savetxt('times.csv', cumul_time, delimiter = "\n")
