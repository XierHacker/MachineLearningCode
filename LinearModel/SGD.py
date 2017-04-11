import numpy as np
import matplotlib.pyplot as plt

#load data
data=np.loadtxt("Olympic.txt",usecols=(1,),ndmin=2)
y=np.loadtxt("Olympic.txt",usecols=(0,),ndmin=2)
#test
#print("type of x:",type(x))
#print("x:",x)
#print("shape of x:",x.shape)
#print("type of y:",type(y))
#print("y:",y)
#print("shape of y:",y.shape)
#print("element type of y:",y.dtype)
#print(y)

#num of variables
n=data.shape[1]
#num of examples
m=data.shape[0]
#learning rate
rate=0.01


#create X matrix and weight matrix
x=np.ones((m,n+1))
x[:,0]=100
x[:,1:]=data[:,0:]
x=x/100
print(x)
weight=np.ones((n+1,1))

#epoch
epoch=10
for i in range(epoch):
    print("epoch: ",i)
    h=x.dot(weight)
    print(h)
    error=h-y
    print("error:",error)
    grad=error.T.dot(x)
    print("grad:\n",grad)
    grad=grad.T
    print(grad)
    #print(rate*grad)
    weight=(weight-rate*grad)/m
    print("weight:\n",weight)
 
    loss=1/(2*m)*((error*error).sum())
    print("loss:",loss)
   
    
#plt.plot(data,y,"+")
#plt.plot(data,X.dot(weight))
#plt.show()


    








