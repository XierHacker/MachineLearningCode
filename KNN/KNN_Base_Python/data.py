import numpy as np 
import operator

#创建数据集的函数：生成数据 group和数据对应的标签labels
def createDataset():
	group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels=['A','A','B','B']
	return group,labels