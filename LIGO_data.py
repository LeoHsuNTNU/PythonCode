import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import h5py 
import json

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def dump_info(name, obj):
    print("{0} :".format(name))
    try:
        print("   .value: {0}".format(obj.value))
        for key in obj.attrs.keys():
            print("     .attrs[{0}]:  {1}".format(key, obj.attrs[key]))
    except:
        pass

fh = h5py.File('/home/leo830227/LOSC_Event_tutorial/LOSC_Event_tutorial/H-H1_LOSC_4_V2-1126259446-32.hdf5', 'r')
fl = h5py.File('/home/leo830227/LOSC_Event_tutorial/LOSC_Event_tutorial/L-L1_LOSC_4_V2-1126259446-32.hdf5', 'r')

for key in fh.keys():
    print(key) 

for key in fh.keys():
    print(key) 

fh.visititems(dump_info)
fl.visititems(dump_info)

strain_H = fh['strain']['Strain'].value
strain_L = fl['strain']['Strain'].value


