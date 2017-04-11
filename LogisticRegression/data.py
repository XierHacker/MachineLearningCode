import numpy as np 
def createdata(filename):
	#open file
	file=open(filename)
	lines=file.readlines()
	rows=len(lines)

	#matrix of data(3 means 2 features+1))
	data=np.zeros((rows,3))
	data[:,0]=1
	training_data=np.zeros((60,3))
	test_data=np.zeros((40,3))
	#matrix of labels(labels need only one colume)
	labels=np.zeros((rows,1))
	training_labels=np.zeros((60,1))
	test_labels=np.zeros((40,1))

	#handle datas
	row=0
	for line in lines:
		line=line.strip().split('\t')

		data[row,1:]=line[:2]
		labels[row,:]=line[-1]
		row+=1
		
	training_data[:,:]=data[:60,:]
	test_data[:,:]=data[60:,:]

	training_labels[:,:]=labels[:60,:]
	test_labels[:,:]=labels[60:,:]

	return training_data,test_data,training_labels,test_labels

