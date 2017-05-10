import numpy as np
import matplotlib.pyplot as plt

def softsign(x):
    y=x/(np.abs(x)+1)
    return y

x=np.linspace(start=-10,stop=10,num=100)
y=softsign(x)

plt.plot(x,y)
plt.grid(True)
plt.show()