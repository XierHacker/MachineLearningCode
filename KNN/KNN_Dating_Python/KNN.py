import numpy as np 
#分类函数(核心)
def classify(testdata,dataset,labels,k):
	dataSize=dataset.shape[0]
	testdata=np.tile(testdata,(dataSize,1))
	#计算距离并且按照返回排序后的下标值列表
	distance=(((testdata-dataset)**2).sum(axis=1))**0.5
	index=distance.argsort()

	classCount={}
	for i in range(k):
		label=labels[index[i]]
		classCount[label]=classCount.get(label,0)+1

	sortedClassCount=sorted(list(classCount.items()),
		key=lambda d:d[1],reverse=True)

	return sortedClassCount[0][0]


#归一化函数(传入的都是处理好的只带数据的矩阵)
def norm(dataset):
	#sum/min/max函数传入0轴表示每列,得到单行M列的数组
	minValue=dataset.min(0)
	maxValue=dataset.max(0)

	m=dataset.shape[0]
	return (dataset-np.tile(minValue,(m,1)))/np.tile(maxValue-minValue,(m,1))


#测试函数
def classifyTest(testdataset,dataset,dataset_labels,
				testdataset_labels,k):
	sampleAmount=testdataset.shape[0]

	#归一化测试集合和训练集合
	testdataset=norm(testdataset)
	dataset=norm(dataset)
	#测试
	numOfWrong=0
	for i in range(sampleAmount):
		print("the real kind is:",testdataset_labels[i])
		print("the result kind is:",
			classify(testdataset[i],dataset,dataset_labels,k))
		if testdataset_labels[i]==classify(testdataset[i],
									dataset,dataset_labels,k):
			print("correct!!")
		else:
			print("Wrong!!")
			numOfWrong+=1
		print()

	print(numOfWrong)



