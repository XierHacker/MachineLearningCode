from __future__ import print_function,division
import numpy as np
import matplotlib.pyplot as plt
import data

#init the weight of dataset
def initWeight(num_of_samples):
    weights=np.ones(shape=(num_of_samples,))/num_of_samples
    return weights

#like a basic stump fuction,can give every sample a label
def stumpClassify(dataSet,featureIndex,threshold,order):
    result=np.ones(shape=(dataSet.shape[0],))
    if order=='lt':
        result[dataSet[:,featureIndex]<=threshold]=-1
    else:
        result[dataSet[:, featureIndex] > threshold] = -1
    return result


#find the best stump
def getBestStump(dataSet,labels,weights):
    num_of_samples=dataSet.shape[0]
    num_of_features=dataSet.shape[1]
    #num of steps,defined by yourself
    num_of_steps=10
    minError=np.inf
    #a dict that record the best choice
    bestStump={}
    #every feature
    for i in range(num_of_features):
        print("feature:",i)
        min=dataSet[:,i].min()
        max=dataSet[:,i].max()
        step_interval=(max-min)/num_of_steps
        #print(min,max)
        for j in range(num_of_steps):
            print("step:",j)
            for order in ['lt','gt']:
                weights_temp=weights.copy()
                threshold=min+float(j)*step_interval
                predict=stumpClassify(dataSet,i,threshold,order)
                print("predict:",predict)
                weights_temp[predict==labels]=0
                print("weight_temp:",weights_temp)
                error=weights_temp.sum()
                print(error)
                if error<minError:
                    minError=error
                    bestResult=predict.copy()
                    bestStump['featureIndex']=i
                    bestStump['threshold'] = threshold
                    bestStump['order']=order

    return bestStump,minError,bestResult

def train(dataSet,labels,times=40):
    #init weights
    weights = initWeight(dataSet.shape[0])
    #function,we can store the all the bestStump(final function)
    G=[]
    #accumulate result
    accuResult=np.zeros(shape=(dataSet.shape[0],))
    for i in range(times):
        print("epoch:",i)
        print("weights:",weights)
        #get the best stump
        bestStump, minError, result = getBestStump(dataSet, labels, weights)
        print("minError:",minError)
        print("result:",result)

        #compute the coefficient
        coefficient=0.5*np.log((1-minError)/max(minError,1e-16))
        print("coefficient:",coefficient)
        bestStump['coefficient']=coefficient

        #add the bestStump(weak classifier) to the function
        G.append(bestStump)

        #update the weights
        weights=weights*np.exp((-1*coefficient)*labels*result)
        weights=weights/weights.sum()

        #compute the accumulate function error
        accuResult+=coefficient*result
        print("acuuResult:",accuResult)
        temp=np.ones(shape=(dataSet.shape[0],))
        temp[np.sign(accuResult)==labels]=0.0
        print("temp:",temp)
        if temp.sum()==0.0:
            break


