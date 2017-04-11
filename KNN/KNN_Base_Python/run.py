import data
import KNN
import draw

testdata=[0.1,0.5]
group,labels=data.createDataset()
a=KNN.classify(testdata,group,labels,3)
print(a)
draw.drawplot(testdata,group,1)