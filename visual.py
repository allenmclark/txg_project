from core_1d import rho, sigma, K, L, c
from core_1d import lam, u
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from net_1d import net
import core_1d


fig, ax = plt.subplots()
ax.set_xlim(0,100)
ax.set_ylim(0,100)

line1, = ax.plot([],[],color='orange',lw=2)
line2, = ax.plot([],[],color='blue',lw=2)

def init():
    line1.set_data([],[])
    line2.set_data([],[])
    return line1,line2



def update(frame):


    x = torch.linspace(0,L,81)
    y1 = u(x,frame)
    y1 = y1.detach().numpy()
    line1.set_data(x,y1)
    y2 = net(core_1d.train_input[frame])
    y2 = y2.detach().numpy()
    line2.set_data(x,y2)
    return line1,line2


                                #correct frames to core_1d.data_range
ani = FuncAnimation(fig, update, frames = core_1d.data_range, blit=True, interval=1)
plt.show()