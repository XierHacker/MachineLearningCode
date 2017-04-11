import numpy as np 
def createData(filename):
	file=open(filename)
	#一次性读入所有行
	lines=file.readlines()
	rows=len(lines)
	#创建数据集
	dataSet=np.zeros((rows,2))

	#数据读入数据集
	row=0
	for line in lines:
		#分割当前行返回列表
		line=line.strip().split('\t')
		dataSet[row,:]=line[:]
		row+=1
	return dataSet