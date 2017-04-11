import data
import KNN 
import drawer

#这里测试数据集和训练数据集都是采用的同一个数据集
dataset,labels=data.creatData("datingTestSet.txt")
testdata_set,testdataset_labels=data.creatData("datingTestSet.txt")

print(type(dataset[0][0]))
#测试分类效果。K取得是10
KNN.classifyTest(testdata_set,dataset,labels,testdataset_labels,10)

#画出训练集的分布
drawer.drawPlot(dataset,labels)