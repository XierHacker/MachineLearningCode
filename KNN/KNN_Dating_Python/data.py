import numpy as np 

def creatData(filename):
	#打开文件,并且读入整个文件到一个字符串里面
	file=open(filename)
	lines=file.readlines()
	sizeOfRecord=len(lines)

	#开始初始化数据集矩阵和标签
	group=np.zeros((sizeOfRecord,3))
	labels=[]
	row=0
	#这里从文件读取存到二维数组的手法记住
	for line in lines:
		line=line.strip()
		tempList=line.split('\t')
		group[row,:]=tempList[:3]
		
		labels.append(tempList[-1])
		row+=1
	return group,labels