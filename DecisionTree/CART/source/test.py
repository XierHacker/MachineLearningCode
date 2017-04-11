import numpy as np
a=np.array([0,1,2,3,4,5])
print(a)
n=np.nonzero(a)
print(n)
print(n[0])

b = np.array([[0,0,3],[0,0,0],[0,0,9]])
n2 = np.nonzero(b)
print(b)
print(n2)
#print(np.transpose(np.nonzero(a)))
print(b[b[:,-1]>3])