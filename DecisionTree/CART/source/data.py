import numpy as np
def loadData(filename):
    dataSet=np.loadtxt(fname=filename,dtype=np.float32)
    return dataSet
