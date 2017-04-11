import matplotlib.pyplot as plt 

def drawplot(testdata,dataset,kind):
	plt.plot(dataset[ :,0],dataset[:,1],'bo',testdata[0],testdata[1],'ro')
	plt.show()