import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    y=1.0/(1.0+np.exp(-x))
    return y

x=np.linspace(start=-10,stop=10,num=100)
y=sigmoid(x)

plt.plot(x,y)
plt.grid(True)
plt.show()