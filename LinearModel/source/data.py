import numpy as np 
def createdata(filename,k):
	file=open(filename)
	lines=file.readlines()
	rows=len(lines)

	#数据的矩阵
	X_Matrix=np.zeros((rows,k+1))
	#结果的矩阵
	T_Matrix=np.zeros((rows,1))
	
	#从文本读数据
	row=0
	for line in lines:
		line=line.strip().split('\t')
		T_Matrix[row]=float(line[0])
		for i in range(k+1):
			X_Matrix[row,i]=float(line[1])**i
		row+=1

	#分离出训练集
	trainset=X_Matrix[:17,:]
	trainresultset=T_Matrix[:17]
	#分离出验证集
	validationset=X_Matrix[17:,:]
	varesultset=T_Matrix[17:]

	#返回训练集,和验证集的东西
	return trainset,trainresultset,validationset,varesultset
	