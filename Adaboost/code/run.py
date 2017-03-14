from __future__ import print_function,division
import numpy as np
import data
import adaboost

dataMat,labels=data.loadSimpleData()
print("dataMat:",dataMat)
print("labels:",labels)

adaboost.train(dataMat,labels)



