import data
import KMeans
import drawer
import numpy as np
dataset=data.createData("testSet.txt")
centerset,labels=KMeans.kMeans(dataset,4)
labels=labels.astype(np.int64)
print(centerset)
print(labels)
print(labels.dtype)

drawer.drawplot(dataset,labels,centerset,4)