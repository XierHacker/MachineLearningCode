import numpy as np
import matplotlib.pyplot as plt

def softplus(x):
    y=np.log(np.exp(x)+1)
    return y

x=np.linspace(start=-10,stop=10,num=100)
y=softplus(x)

plt.plot(x,y)
plt.grid(True)
plt.show()