import numpy as np 
import operator

#定义函数，参数分别是待分类数据，数据集，数据集标签，k值
def classify(testdata,dataset,labels,k):
	#知道数据集大小有两个方法：
	#1从标签集中看长度，
	#从数据集中看行数
	datasetSize=dataset.shape[0]
	#扩展待测试数据的数量，以便和数据集中的每个元素做运算。
	testdata=np.tile(testdata,(datasetSize,1))

	#做运算（这里是用的欧式距离）
	distance=(((testdata-dataset)**2).sum(axis=1))**0.5
	#得到按照从小到大的顺序排列的元素的下表
	index=distance.argsort()
	
	#建立一个关于标签以及标签出现次数的字典
	classCount={}
	for i in range(k):
		label=labels[index[i]]
		classCount[label]=classCount.get(label,0)+1


	#先将字典转化为键值对的列表，按照出现次数进行排序
	#排序返回的也是键值对的列表
	sortedclassCount=sorted(list(classCount.items()),
		key=lambda d:d[1],reverse=True)

	#返回标签
	return sortedclassCount[0][0]
