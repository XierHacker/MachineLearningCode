import matplotlib.pyplot as plt
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import data 

def drawPlot(dataset,labels):
	fig=plt.figure(1)
	ax=fig.add_subplot(111,projection='3d')
	for i in range(dataset.shape[0]):
		x=dataset[i][0]
		y=dataset[i][1]
		z=dataset[i][2]
		if labels[i]=='largeDoses':
			ax.scatter(x,y,z,c='b',marker='o')
		elif labels[i]=='smallDoses':
			ax.scatter(x,y,z,c='r',marker='s')
		else:
			ax.scatter(x,y,z,c='g',marker='^')
	plt.show()