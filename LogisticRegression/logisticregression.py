import numpy as np
import math
 
#sigmoid function
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

#classifier
def classifier(test_data,test_labels,weights):
	correct=0
	total=test_data.shape[0]
	#print(total)
	prob=sigmoid(np.dot(test_data,weights))
	#print(prob)
	for i in range(total):
		if prob[i]>0.5:
			prob[i]=1
		else:
			prob[i]=0
		if prob[i]==test_labels[i]:
			correct+=1

	percent=(correct/total)*100
	return percent

#gradiant descent and updata the weights
def gradDescent(training_data,training_labels,learning_rate,epochs):
	#get the dimension of data set
	m,n=np.shape(training_data)

	#weights we want to get(the number of features)
	w=np.ones((n,1))     
	#Iteration
	for i in range(epochs):
		print("epochs:",i)
		hypothesis=sigmoid(np.dot(training_data,w))
		loss=hypothesis-training_labels
		cost=np.sum(loss**2)/(2*m)
		print("Cost:",cost)
		gradient=np.dot(training_data.transpose(),loss)/m
		print("gradient:")
		print(gradient)

		#update w
		w=w-learning_rate*gradient
		print("weight:")
		print("w:",w)
	return w


def SGD(training_data,training_labels,learning_rate,epochs):
    pass