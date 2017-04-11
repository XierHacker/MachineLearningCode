import numpy as np
import matplotlib.pyplot as plt

#split dataSet trough featureIndex and value
def splitDataSet(dataset,featureIndex,value):
    subDataSet0=dataset[dataset[:,featureIndex]<=value,:]
    subDataSet1=dataset[dataset[:,featureIndex]>value,:]
    return subDataSet0,subDataSet1

#compute the regression Error in a data Set
def getError(dataSet):
    error=np.var(dataSet[:,-1])*dataSet.shape[0]
    return error

#choose the best featureIndex and value in dataSet
def chooseBestSplit(dataSet,leastErrorDescent,leastNumOfSplit):
    rows,cols=np.shape(dataSet)

    #error in dataSet
    Error=getError(dataSet)

    #init some important value we want get
    bestError=np.inf
    bestFeatureIndex=0
    bestValue=0

    #search process
    #every feature index
    for featureIndex in range(cols-1):
        #every value in dataSet of specific index
        for value in set(dataSet[:,featureIndex]):
            subDataSet0,subDataSet1=splitDataSet(dataSet,featureIndex,value)
            #print("sub0",subDataSet0.shape[0])
            #print("sub1", subDataSet1.shape[0])

          #  print(subDataSet0)
            if (subDataSet0.shape[0]<leastNumOfSplit) or (subDataSet1.shape[0]<leastNumOfSplit):
                continue
            #compute error
            tempError=getError(subDataSet0)+getError(subDataSet1)
            #print("tempError:",tempError)
            if tempError<bestError:
                bestError=tempError
                bestFeatureIndex=featureIndex
                bestValue=value

           # print("BestError:", bestError)
           # print("BestIndex:", bestFeatureIndex)
           # print("BestValue:", bestValue)
    if Error-bestError<leastErrorDescent:
        return None,np.mean(dataSet[:,-1])
    mat0,mat1=splitDataSet(dataSet,bestFeatureIndex,bestValue)
    if (mat0.shape[0]<leastNumOfSplit) or (mat1.shape[0]<leastNumOfSplit):
        return None,np.mean(dataSet[:,-1])

    return bestFeatureIndex,bestValue


#build tree
def buildTree(dataSet,leastErrorDescent=1,leastNumOfSplit=4):
    bestFeatureIndex,bestValue=chooseBestSplit(dataSet,leastErrorDescent,leastNumOfSplit)

    #recursion termination
    if bestFeatureIndex==None:
        return bestValue

    Tree={}
    Tree["featureIndex"]=bestFeatureIndex
    Tree["value"]=bestValue
    #get subset
    leftSet,rightSet=splitDataSet(dataSet,bestFeatureIndex,bestValue)

    #recursive function
    Tree["left"]=buildTree(leftSet,leastErrorDescent,leastNumOfSplit)
    Tree["right"] = buildTree(rightSet, leastErrorDescent, leastNumOfSplit)

    return Tree

def isTree(tree):
    return (type(tree).__name__=='dict')


def predict(tree,x):
    if x[tree["featureIndex"]]<tree["value"]:
        if isTree(tree["left"]):
            return predict(tree["left"],x)
        else:
            return tree["left"]

    else:
        if isTree(tree["right"]):
            return predict(tree["right"],x)
        else:
            return tree["right"]




