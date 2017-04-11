import numpy as np 

#计算距离:其中centerset是所有中心点的集合
def distance(dataset,centerset):
	#K个中心点
	k=centerset.shape[0]
	rows=dataset.shape[0]
	
	#返回row行K列的矩阵
	#其中每行是一个数据离所有中心点的距离
	distanceMatrix=np.zeros((rows,k))
	for row in range(rows):
		temp=dataset[row,:]
		temp=np.tile(temp,(k,1))
		#print("现在在计算距离的函数中:",row)
		#print(temp)
		dis=((temp-centerset)**2).sum(1)
		#print(dis)
		distanceMatrix[row,:]=dis[:]
	return distanceMatrix

#随机初始化中心点(-5到5之间)
def createRandomCenter(k):
	centerset=(np.random.rand(k,2)-0.5)*10
	return centerset


#更新中心点
def updateCenter(dataset,labels,k):
	#其中第一列存x坐标,第二列存y坐标
	centerset=np.zeros((k,2))
	#找到label的所有索引
	#第K行表示一类中dataset的索引
	index=[]
	for i in range(k):
		index_temp=np.where(labels==i)
		#因为np.where返回一个元组
		#取索引为0的元素就是array数组了
		index.append(index_temp[0])
	#print(index)
	#计算均值
	for i in range(k):
		xsums=0
		ysums=0
		for j in range(index[i].shape[0]):
			xsums+=dataset[index[i][j]][0]
			ysums+=dataset[index[i][j]][1]
		centerset[i,0]=xsums/index[i].shape[0]
		centerset[i,1]=ysums/index[i].shape[0]
	return centerset

#KMeans核心算法
def kMeans(dataset,k):
	#初始化过程
	#判断每个数据类别是否改变的flag
	changed=True
	rows=dataset.shape[0]
	centerset=createRandomCenter(k)
	#print(centerset)
	#类别,索引i表示数据点i
	#而里面的内容就是数据点i相应的类别
	labels=np.zeros(rows)
	#迭代次数
	times=0
	while(changed):
		times+=1
		print(" 第 ",times," 次迭代:")
		dis=distance(dataset,centerset)
		#把距离值按照索引来排序
		#索引在前面的距离最小
		index=dis.argsort()
		#把所有距离最小的索引赋给labels_temp,
		#这里把索引值作为类别
		labels_temp=index[:,0]
		#所有类别都想等，意味不再改变
		if (labels==labels_temp).all():
			changed=False
		else:
			labels[:]=labels_temp[:]
			centerset=updateCenter(dataset,labels,k)
		
	return centerset,labels





