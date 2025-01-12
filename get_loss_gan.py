import numpy as np
from matplotlib import pyplot
import glob

folder = 'trial_spiral_2023-04-13_1_52'

logfile = glob.glob("{}/train_log*".format(folder))[0]


data = np.loadtxt(logfile,skiprows=1,delimiter=',')

pyplot.plot(data[:,0],lw=2.5,alpha=0.7,color='blue',label='G')
pyplot.plot(data[:,1],lw=2.5,alpha=0.7,color='red',label='D')

pyplot.legend()
pyplot.savefig("loss_spiral.pdf")
