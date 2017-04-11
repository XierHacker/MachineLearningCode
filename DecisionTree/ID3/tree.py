from __future__ import print_function,division
import numpy as np
import math
import data

#compute the entropy
def getEntropy(dataSet):
    entropy=0.0
    #amount of samples
    numOfSamples=dataSet.shape[0]
    #build the dictionary of labels to do some compute
    labelsDict={}
    for label in dataSet[:,-1]:
        if label not in labelsDict:
            labelsDict[label]=0
        labelsDict[label]+=1

    print("labelDict:",labelsDict)
    #compute the entropy
    for key in labelsDict.keys():
        prob=float(labelsDict[key])/numOfSamples
        entropy+=prob*math.log2(prob)

    return entropy*-1

samples,cluts=data.createData()
print("entropy:",getEntropy(samples))


'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

print(splitDataSet(samples,0,1))
'''
