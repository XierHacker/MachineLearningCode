import numpy as np
import matplotlib.pyplot as plt

mat1=np.array([[1,2],[0,0],[0,0]])
U,sigma,V_T=np.linalg.svd(mat1)
print("U:",U)
print("sigma:",sigma)
print("V_T:",V_T)