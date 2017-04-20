import data
import numpy as np

#reconstruct the shape of SVD matrix
#u:matrix u
#sigma:matrix sigma
#v_t:matrix v_t
#num:the shape of new sigma matrix we want to take
def reshapeMatrix(u,sigma,v_t,num):
    sigma_matrix=np.zeros(shape=(num,num))
    for i in range(num):
        sigma_matrix[i][i]=sigma[i]
    return u[:,:num],sigma_matrix,v_t[:num,:]

def EuclidSimilarity(vecA,vecB):
    return 1.0/(1.0+np.linalg.norm(vecA-vecB))


def predict()

dataSet1=data.createData()
#print(dataSet1)
#print(dataSet1.shape)
U,sigma,V_T=np.linalg.svd(dataSet1)
'''
print("U:\n",U)
print(U.shape)
print(type(U))
print("sigma:\n",sigma)
print("V_T:\n",V_T)
print("\n\n\n\n\n")
'''

U_matrix,sigma_matrix,V_T_matrix=reshapeMatrix(U,sigma,V_T,3)
'''
print("U:\n",U_matrix)
print(U_matrix.shape)
print(type(U_matrix))
print("sigma:\n",sigma_matrix)
print("V_T:\n",V_T_matrix)
'''

result=(U_matrix.dot(sigma_matrix)).dot(V_T_matrix)
#print(result)
#print(EuclidSimilarity(dataSet1[:,0],dataSet1[:,0]))
#print(EuclidSimilarity(dataSet1[:,0],dataSet1[:,4]))
