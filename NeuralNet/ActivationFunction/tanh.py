import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    y=(1.0-np.exp(-2*x))/(1.0+np.exp(-2*x))
    return y



x=np.linspace(start=-10,stop=10,num=100)
y=tanh(x)

plt.plot(x,y)
plt.grid(True)
plt.show()