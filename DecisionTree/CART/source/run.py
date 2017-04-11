import numpy as np
import data
import CART

dataMat1=data.loadData("../data/ex00.txt")
dataMat2=data.loadData("../data/ex0.txt")

'''
print(dataMat.shape)
print(np.shape(dataMat))
e=CART.getError(dataMat)
print(e)
print(CART.getError(mat0))
print(CART.getError(mat1))

mat0,mat1=CART.splitDataSet(dataMat,0,0.5)
print(mat0)
print(mat1)
print(mat0.shape)
'''



#bestIndex,bestValue=CART.chooseBestSplit(dataMat)
#print(bestIndex,bestValue)

#tree1
tree1=CART.buildTree(dataMat1)
print(tree1)

#tree2
tree2=CART.buildTree(dataMat2)
print(tree2)

x=[1.0,0.559009]
print(CART.predict(tree2,x))