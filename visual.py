import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import style
import torch

print(19, flush=True)
rho = 8.92
sigma = .092
K = .95
L = 80
c = (K/(sigma*rho))**.5

def lam(n):
    return c * n * torch.pi / L

def u(x,t):
    return 100 * torch.e**(-lam(1)**2*t) * torch.sin((torch.pi/L)*x)



fig, ax = plt.subplots()
ax.set_xlim(0,100)
ax.set_ylim(0,100)

line, = ax.plot([],[],lw=2)


def init():
    line.set_data([],[])
    return line,




def update(frame):
    x = torch.linspace(0,L,100)
    y = u(x,frame)
    line.set_data(x,y)
    return line,



ani = FuncAnimation(fig, update, frames = 1500, init_func=init, blit=True, interval=1)
plt.show()