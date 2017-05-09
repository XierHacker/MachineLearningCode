import numpy as np
import matplotlib.pyplot as plt

def elu(x,a):
    y=x.copy()
    for i in range(y.shape[0]):
        if y[i]<0:
            y[i]=a*(np.exp(y[i])-1)
    return y

x=np.linspace(start=-10,stop=10,num=100)
y=elu(x,0.5)
print(x)
print(y)
plt.plot(x,y)
plt.grid(True)
plt.show()