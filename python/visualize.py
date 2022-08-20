from numpy import genfromtxt
from matplotlib import pyplot as plt
per_data=genfromtxt('./python/error.csv',delimiter=',')
plt.plot(per_data)