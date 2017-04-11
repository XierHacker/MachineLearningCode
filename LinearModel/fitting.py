import numpy as np 
def fit(X_Matrix,T_Matrix):
	#直接套退出来的矩阵公式就行
	X_T=np.transpose(X_Matrix)
	arg=(np.linalg.inv(X_T.dot(X_Matrix)).\
		dot(X_T)).dot(T_Matrix)
	return arg
