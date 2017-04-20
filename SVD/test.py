import numpy as np
import matplotlib.pyplot as plt
'''
mat1=np.array([[1,2],[0,0],[0,0]])
U,sigma,V_T=np.linalg.svd(mat1)
print("U:",U)
print("sigma:",sigma)
print("V_T:",V_T)
'''
x=np.array([0,1,1,0,2,0,1])
#x=np.array([[0,1,0],[0,1,2],[1,1,1]])
#x=np.mat(x)
print(x)
index=np.nonzero(x)
print(index)
print(index[0])
#print(index.shape)
#print(x.A)
