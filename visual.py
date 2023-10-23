from core_1d import rho, sigma, K, L, c
from core_1d import lam, u
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch



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