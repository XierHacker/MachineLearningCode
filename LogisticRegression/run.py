import data
import logisticregression
import numpy as np
import matplotlib.pyplot as plt 

training_data,test_data,training_labels,test_labels=data.createdata("testSet.txt")

w=logisticregression.gradDescent(training_data,training_labels,0.05,30)

print("test rate:")
p=logisticregression.classifier(test_data,test_labels,w)
print(p,"%")
#draw scatter
for i in range(training_data.shape[0]):
	plt.plot()
	if(training_labels[i][0]==1):
		plt.plot(training_data[i,1],training_data[i,2],'r*')
	else:
		plt.plot(training_data[i,1],training_data[i,2],'b+')
#draw line
x=np.linspace(-5,5,60)
y=-(x*w[1]+w[0])/w[2]
plt.plot(x,y)
plt.show()

