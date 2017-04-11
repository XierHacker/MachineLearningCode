import matplotlib.pyplot as plt 
import matplotlib.animation as animation

def drawplot(dataset,labels,centerset,k):
	markers=["*","+","^","o","s"]
	colors=["b","c","r","g","k"]
	fig=plt.figure(1)
	ax=fig.add_subplot(111)

	#画中心点
	for i in range(k):
		ax.plot(centerset[i,0],centerset[i,1],
			marker=markers[i],color=colors[i],
			markersize=10.0)

	#画数据点
	row=0
	for i in labels:
		ax.plot(dataset[row,0],dataset[row,1],
			marker=markers[3],color=colors[i])
		row+=1
	
	plt.show()
