import data
import fitting
import matplotlib.pyplot as plt
import numpy as np 

#生成阶数和损失
scala=[i+1 for i in range(3)]
train_loss=[]
validation_loss=[]

#对单一变量的k阶拟合,不知道为什么,只能够算到第三阶.
for i in scala:
	trainset,trainresultset,validationset,\
	varesultset=data.createdata("Olympic.txt",i)
	arg=fitting.fit(trainset,trainresultset)
	#训练预测的结果
	predictresult=trainset.dot(arg)
	train_loss.append(((trainresultset-predictresult)**2).mean())
	#验证预测的结果
	predictresult=validationset.dot(arg)
	validation_loss.append(((varesultset-predictresult)**2).mean())

print("trainning loss:\n",train_loss,'\n')
print("validation loss:\n",validation_loss,'\n')



#设置一个画图的类型列表
#kind=[['b','o'],['r','*'],['g','^'],['k','+']]
#画图部分
fig=plt.figure(1)
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
#man
#数据集的散点图
#ax.scatter(trainset[:,1],trainresultset,c='b',marker='o')
#拟合直线图
ax1.plot(scala,train_loss,'b')
ax2.plot(scala,validation_loss,'r')

plt.show()