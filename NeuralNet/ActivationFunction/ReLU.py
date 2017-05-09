import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    y=x[:]
    y[y<0]=0
    return y

x=np.linspace(start=-10,stop=10,num=100)
y=relu(x)
print(y)
plt.plot(x,y,"*")
plt.grid(True)
plt.show()